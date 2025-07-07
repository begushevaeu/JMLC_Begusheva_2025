from neo4j import GraphDatabase
from . import config

class Neo4jConnection:

    def __init__(self, uri, user, password):
        self.__uri = uri
        self.__user = user
        self.__password = password
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__password))
        except Exception as e:
            print(f"Failed to create the driver: {e}")

    def close(self):
        if self.__driver is not None:
            self.__driver.close()

    def query(self, query, parameters=None, db=None):
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try:
            session = self.__driver.session(database=db) if db is not None else self.__driver.session()
            response = list(session.run(query, parameters))
        except Exception as e:
            print(f"Query failed: {e}")
        finally:
            if session is not None:
                session.close()
        return response

_neo4j_conn = None

def get_neo4j_connection():
    """
    Возвращает синглтон-экземпляр соединения с Neo4j.
    Создает соединение при первом вызове.
    """
    global _neo4j_conn
    if _neo4j_conn is None:
        _neo4j_conn = Neo4jConnection(
            uri=config.NEO4J_URI,
            user=config.NEO4J_USER,
            password=config.NEO4J_PASSWORD
        )
    return _neo4j_conn

def add_transaction_to_graph(tx_data):
    """
    Добавляет транзакцию в граф Neo4j.
    Создает узлы для счетов (если они не существуют) и ребро транзакции между ними.
    """
    conn = get_neo4j_connection()
    query = (
        "MERGE (s:Account {id: $source_id}) "
        "MERGE (t:Account {id: $target_id}) "
        "CREATE (s)-[r:TRANSACTION { \
            amount: $amount, \
            timestamp: datetime($timestamp), \
            is_fraud: $is_fraud, \
            alert_id: $alert_id \
        }]->(t)"
    )
    parameters = {
        "source_id": tx_data['nameOrig'],
        "target_id": tx_data['nameDest'],
        "amount": tx_data['amount'],
        "timestamp": tx_data['timestamp'].isoformat(),
        "is_fraud": tx_data.get('isFraud', False),
        "alert_id": tx_data.get('alert_id')
    }
    conn.query(query, parameters)

def get_latest_transactions(limit=25):
    """
    Получает последние транзакции из Neo4j.

    Args:
        limit (int): Максимальное количество транзакций для возврата.

    Returns:
        list[dict]: Список словарей, где каждый словарь представляет транзакцию.
    """
    conn = get_neo4j_connection()
    query = (
        "MATCH (sender:Account)-[tx:TRANSACTION]->(receiver:Account) "
        "RETURN "
        "    sender.id AS sender_id, "
        "    receiver.id AS receiver_id, "
        "    tx.amount AS amount, "
        "    tx.timestamp AS timestamp, "
        "    tx.is_fraud AS is_fraud, "
        "    tx.alert_id AS alert_id "
        "ORDER BY tx.timestamp DESC "
        "LIMIT $limit"
    )
    parameters = {"limit": limit}
    result = conn.query(query, parameters)

    # Преобразуем результат в список словарей
    transactions = [
        {
            "Отправитель": record["sender_id"],
            "Получатель": record["receiver_id"],
            "Сумма": record["amount"],
            "Время": record["timestamp"].to_native(), # Преобразуем в нативный datetime
            "Мошенничество": record["is_fraud"],
            "ID Предупреждения": record["alert_id"]
        }
        for record in result
    ]
    return transactions

from collections import defaultdict

def get_max_depth_for_account(account_id: str) -> int:
    """
    Рассчитывает максимальную глубину связей для заданного счета.
    Находит самую длинную из всех кратчайших путей к другим счетам.
    Если пути не найдены, возвращает 1.
    """
    conn = get_neo4j_connection()
    # Ищем самый длинный из кратчайших путей от указанного узла до любого другого.
    # Ограничиваем максимальную глубину поиска (например, 10), чтобы избежать слишком долгих запросов.
    query = """
    MATCH (start:Account {id: $account_id}), (target:Account)
    WHERE start <> target
    MATCH path = shortestPath((start)-[*..10]-(target))
    RETURN MAX(length(path)) AS max_depth
    """
    parameters = {"account_id": account_id}
    result = conn.query(query, parameters)

    # Проверяем, что результат не пустой и содержит значение max_depth
    if result and result[0] and result[0].get('max_depth') is not None:
        # Возвращаем найденную глубину, но не меньше 1
        return max(1, result[0]['max_depth'])
    
    # Если путей нет или счет не найден, возвращаем 1, чтобы слайдер в дашборде имел корректное минимальное значение.
    return 1

def get_graph_for_accounts(account_ids: list[str], depth: int = 1):
    """
    Получает подграф для указанных счетов.
    Возвращает плоский список узлов и ребер, где каждое ребро - одна транзакция.
    """
    if not account_ids:
        return {'nodes': [], 'edges': []}

    conn = get_neo4j_connection()
    # Глубина должна быть в разумных пределах
    depth = max(1, min(int(depth), 5))

    # Запрос для получения всех узлов и связей на заданной глубине.
    query = f"""
    MATCH path = (a:Account)-[:TRANSACTION*1..{depth}]-(b)
    WHERE a.name IN $account_ids
    UNWIND nodes(path) AS node
    UNWIND relationships(path) AS rel
    RETURN COLLECT(DISTINCT node) AS nodes, COLLECT(DISTINCT rel) AS edges
    """
    result = conn.query(query, {"account_ids": account_ids, "depth": depth})

    if not result or not result[0]['nodes']:
        # Если у начальных узлов нет связей, вернем хотя бы их самих
        query_start_nodes = """
        MATCH (n:Account)
        WHERE n.name IN $account_ids
        RETURN COLLECT(n) as nodes
        """
        start_nodes_result = conn.query(query_start_nodes, {"account_ids": account_ids})
        if start_nodes_result and start_nodes_result[0]['nodes']:
             raw_nodes = start_nodes_result[0]['nodes']
             processed_nodes = [{'id': node['id'], 'name': node['id']} for node in raw_nodes]
             return {'nodes': processed_nodes, 'edges': []}
        return {'nodes': [], 'edges': []}

    raw_nodes = result[0]['nodes']
    raw_edges = result[0]['edges']

    # Обрабатываем узлы: просто извлекаем их ID и имя.
    all_node_ids = {node['name'] for node in raw_nodes}
    processed_nodes = [{'id': node['name'], 'name': node['name']} for node in raw_nodes]

    # Обрабатываем ребра: создаем плоский список транзакций, обогащенный данными для подсказок.
    from neo4j.time import DateTime
    processed_edges = []
    for rel in raw_edges:
        start_node_id = rel.start_node['name']
        end_node_id = rel.end_node['name']
        
        # Убедимся, что оба конца ребра находятся в нашем наборе узлов
        if start_node_id in all_node_ids and end_node_id in all_node_ids:
            properties = dict(rel.items())
            timestamp = properties.get('timestamp')
            if isinstance(timestamp, DateTime):
                native_dt = timestamp.to_native()
                date_str = native_dt.strftime('%Y-%m-%d')
                time_str = native_dt.strftime('%H:%M:%S')
            else:
                date_str, time_str = 'N/A', 'N/A'

            processed_edges.append({
                'id': properties.get('transaction_id', rel.id),
                'source': start_node_id,
                'target': end_node_id,
                'amount': properties.get('amount', 0),
                'is_fraud': properties.get('is_fraud', False),
                'currency': properties.get('currency', 'N/A'),
                'date': date_str,
                'time': time_str
            })

    return {'nodes': processed_nodes, 'edges': processed_edges}
