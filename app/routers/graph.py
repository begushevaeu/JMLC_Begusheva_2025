import json
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from neo4j import GraphDatabase, Driver
from .. import schemas

# --- Конфигурация и подключение к Neo4j ---

CONFIG_PATH = 'config.json'

def get_neo4j_config() -> Dict[str, Any]:
    """Загружает конфигурацию Neo4j из файла.

    Returns:
        Словарь с параметрами подключения.
    """
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Файл конфигурации '{CONFIG_PATH}' не найден.")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Ошибка при чтении файла конфигурации '{CONFIG_PATH}'.")


def get_neo4j_driver() -> Driver:
    """Создает и возвращает драйвер для подключения к Neo4j."""
    config = get_neo4j_config()
    uri = config.get("NEO4J_URI", "")
    # Принудительно используем bolt:// для прямого подключения к одиночному инстансу
    if uri.startswith("neo4j://"):
        uri = uri.replace("neo4j://", "bolt://", 1)
        
    return GraphDatabase.driver(
        uri, 
        auth=(config.get("NEO4J_USER"), config.get("NEO4J_PASSWORD"))
    )

router = APIRouter(
    prefix="/graphs",
    tags=["Graphs"],
    responses={404: {"description": "Not found"}},
)

# --- Модели данных (Pydantic) ---

class GraphQuery(BaseModel):
    account_ids: List[str]
    depth: int = 1

# --- Эндпоинты роутера ---

@router.post("/", response_model=schemas.GraphResponse)
def get_graph_data(request: schemas.GraphRequest, driver: Driver = Depends(get_neo4j_driver)):
    """Получает данные для графа, используя модель (Account)-[:SENT]->(Transaction)-[:TO]->(Account)."""
    nodes = set()
    edges = []
    
    # search_depth в отношениях. 1 транзакция = 2 отношения (SENT, TO).
    # MATCH p = (a)-[*..2]->(b) означает путь длиной до 2 отношений.
    # Поэтому для глубины 1 нам нужен путь *..2, для глубины 2 -> *..4 и т.д.
    search_depth = request.depth * 2

    # Этот запрос ищет пути, соответствующие модели (Account)-[:SENT]->(Transaction)-[:TO]->(Account).
    # Он начинается с указанных аккаунтов и движется на заданную глубину.
    cypher_query = """
        MATCH (start_node:Account)
        WHERE start_node.name IN $account_ids
        // Находим все транзакции в пределах заданной глубины
        CALL apoc.path.expandConfig(start_node, {
            relationshipFilter: 'SENT>|<TO',
            labelFilter: '+Account|Transaction',
            minLevel: 1,
            maxLevel: %s
        }) YIELD path
        // Извлекаем из пути только узлы-транзакции
        WITH [node IN nodes(path) WHERE node:Transaction] AS transactions
        UNWIND transactions AS t
        // Для каждой уникальной транзакции находим ее отправителя и получателя
        WITH DISTINCT t
        MATCH (sender:Account)-[:SENT]->(t)-[:TO]->(receiver:Account)
        RETURN sender, receiver, t
    """ % search_depth

    with driver.session() as session:
        # Используем .read_transaction для гарантии корректной работы в кластере
        records = session.read_transaction(lambda tx: list(tx.run(cypher_query, account_ids=request.account_ids)))
        
        for record in records:
            sender_node = record["sender"]
            receiver_node = record["receiver"]
            tx_node = record["t"]

            # Добавляем узлы-счета в множество, чтобы избежать дубликатов
            nodes.add((sender_node['name'], sender_node['name']))
            nodes.add((receiver_node['name'], receiver_node['name']))

            # Обрабатываем и форматируем дату и время
            dt = tx_node.get('datetime')
            date_str = dt.to_native().strftime('%Y-%m-%d') if dt else 'N/A'
            time_str = dt.to_native().strftime('%H:%M:%S') if dt else 'N/A'

            # Формируем ребро графа
            edge_data = {
                "source": sender_node['name'],
                "target": receiver_node['name'],
                "type": "TRANSACTION",
                "amount": tx_node.get('amount'),
                "is_fraud": tx_node.get('is_fraud'),
                "date": date_str,
                "time": time_str,
                "currency": tx_node.get('currency', '')
            }
            edges.append(edge_data)

    # Преобразуем множество узлов в список словарей для ответа API
    nodes_list = [{"id": id, "name": label, "label": label} for id, label in nodes]
    return {"nodes": nodes_list, "edges": edges}


@router.get("/max_depth/{account_id}", response_model=int)
def get_max_depth(account_id: str, driver: Driver = Depends(get_neo4j_driver)):
    """Вычисляет максимальную глубину графа (в транзакциях) для счета."""
    # Этот запрос находит самый длинный путь от заданного счета и делит его длину на 2,
    # так как одна транзакция представлена двумя отношениями (SENT и TO).
    # Этот запрос находит самый длинный путь от заданного счета и делит его длину на 2,
    # так как одна транзакция (Account -> Transaction -> Account) имеет длину 2.
    cypher_query = """
        MATCH (a:Account {name: $account_id})
        CALL apoc.path.expandConfig(a, {
            relationshipFilter: 'SENT>|<TO',
            labelFilter: '+Account|Transaction',
            uniqueness: 'NODE_PATH' // Уникальность узлов в пути
        }) YIELD path
        WITH length(path) AS path_length
        RETURN toInteger(ceil(max(path_length) / 2.0)) AS max_depth
    """
    
    with driver.session() as session:
        result = session.run(cypher_query, account_id=account_id).single()
        max_depth = result["max_depth"] if result and result["max_depth"] is not None else 1
    
    return int(max_depth)
