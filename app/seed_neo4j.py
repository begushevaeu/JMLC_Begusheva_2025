# Принудительно загружаем конфигурацию в первую очередь, чтобы избежать ошибок импорта
import app.config

from sqlalchemy.orm import joinedload

# Используем централизованные объекты из модуля database
from app.database import SessionLocal
from app.models import Transaction, Account
from app.neo4j_utils import get_neo4j_connection

def seed_existing_transactions_to_neo4j():
    """Переносит транзакции из PostgreSQL в Neo4j, используя эффективную пакетную операцию."""
    db = SessionLocal()
    conn = get_neo4j_connection()
    BATCH_SIZE = 50000  # Размер пакета для обработки

    try:
        # --- 1. Очистка Neo4j в пакетном режиме ---
        print("Очистка всех данных из Neo4j (в пакетном режиме)...")
        while True:
            # Удаляем порциями, чтобы не перегружать память
            result = conn.query("MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n)")
            # Neo4j-driver v5+ возвращает список записей, поэтому нужен доступ по индексу
            count = result[0]['count(n)'] if result else 0
            if count == 0:
                break
            print(f"Удалена порция из {count} узлов...")
        print("База данных Neo4j полностью очищена.")

        # --- 2. Создание индекса для ускорения MERGE ---
        print("Обеспечение наличия индекса для :Account(id)...")
        conn.query("CREATE INDEX account_id_index IF NOT EXISTS FOR (n:Account) ON (n.id)")
        print("Индекс успешно создан или уже существовал.")

        # --- 3. Извлечение и загрузка данных пакетами ---
        print("Начинается пакетная загрузка транзакций из PostgreSQL в Neo4j...")
        offset = 0
        total_processed = 0
        while True:
            print(f"Извлечение пакета транзакций (смещение: {offset}, размер: {BATCH_SIZE})...")
            transactions = db.query(Transaction).options(
                joinedload(Transaction.sender),
                joinedload(Transaction.receiver)
            ).offset(offset).limit(BATCH_SIZE).all()

            if not transactions:
                print("Больше транзакций для обработки не найдено.")
                break

            # Подготовка данных для текущего пакета
            tx_list = [
                {
                    'tx_id': tx.id,
                    'orig_id': tx.sender.name,  # Используем имя счета как ID в графе
                    'dest_id': tx.receiver.name, # Используем имя счета как ID в графе
                    'amount': float(tx.amount),
                    'timestamp': tx.datetime.isoformat(),
                    'isFraud': tx.is_fraud or False,
                    'currency': tx.payment_currency_iso
                }
                for tx in transactions
            ]

            # Выполнение запроса для текущего пакета
            query = """
            UNWIND $tx_list AS tx
            MERGE (orig:Account {id: tx.orig_id})
            MERGE (dest:Account {id: tx.dest_id})
            CREATE (orig)-[r:TRANSACTION {
                transaction_id: tx.tx_id,
                amount: tx.amount,
                timestamp: datetime(tx.timestamp),
                is_fraud: tx.isFraud,
                currency: tx.currency
            }]->(dest)
            """
            conn.query(query, parameters={'tx_list': tx_list})
            
            processed_count = len(transactions)
            total_processed += processed_count
            print(f"Загружен пакет из {processed_count} транзакций. Всего обработано: {total_processed}.")

            offset += BATCH_SIZE

        print(f"\nУспешно загружено {total_processed} транзакций в Neo4j!")

    finally:
        db.close()

if __name__ == "__main__":
    seed_existing_transactions_to_neo4j()
    