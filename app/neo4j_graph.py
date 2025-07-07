import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from . import schemas

# Загружаем переменные окружения из .env файла
load_dotenv()

class Neo4jGraph:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def add_transaction(self, transaction: schemas.Transaction):
        with self._driver.session() as session:
            session.write_transaction(self._create_transaction_graph, transaction)

    @staticmethod
    def _create_transaction_graph(tx, transaction: schemas.Transaction):
        # Используем MERGE, чтобы создать узел, только если он не существует
        # Это предотвращает дублирование счетов
        query = (
            "MERGE (s:Account {name: $sender_name}) "
            "MERGE (r:Account {name: $receiver_name}) "
            "CREATE (s)-[t:TRANSACTION]->(r) "
            "SET t.amount = $amount, "
            "    t.currency = $currency, "
            "    t.datetime = datetime($datetime), "
            "    t.is_fraud = $is_fraud, "
            "    t.fraud_probability = $fraud_probability"
        )
        
        sender_name = transaction.sender.name
        receiver_name = transaction.receiver.name

        tx.run(query, 
               sender_name=sender_name,
               receiver_name=receiver_name, 
               amount=transaction.amount,
               currency=transaction.payment_currency_iso,
               datetime=transaction.datetime.isoformat(),
               is_fraud=transaction.is_fraud,
               fraud_probability=transaction.fraud_probability
        )

# Глобальный экземпляр для переиспользования
# В реальном приложении управление жизненным циклом должно быть более строгим
# (например, создаваться при старте FastAPI и закрываться при остановке)

# Данные для подключения из переменных окружения
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def get_neo4j_db():
    db = Neo4jGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        yield db
    finally:
        db.close()
