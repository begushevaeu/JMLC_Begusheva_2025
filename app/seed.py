import pandas as pd
from sqlalchemy.orm import Session
from . import models
from .models import DataLoadingState
from .database import SessionLocal
from .database import get_neo4j_driver, engine, Base
import logging
import os
import argparse

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Построение абсолютного пути к файлу данных
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE_PATH = os.path.join(BASE_DIR, 'Datasets', 'test_predictions.csv')

def seed_data(db: Session, fraction: float = None, offset: int = None, limit: int = None):
    """Заполняет PostgreSQL и Neo4j, используя данные из CSV файла."""
    logger.info(f"Загрузка данных из {DATA_FILE_PATH}...")
    try:
        df_full = pd.read_csv(DATA_FILE_PATH)
        df_full['datetime'] = pd.to_datetime(df_full['datetime'])
    except FileNotFoundError:
        logger.error(f"Файл данных не найден: {DATA_FILE_PATH}. Посев невозможен.")
        return

    # Определяем срез данных для загрузки
    if fraction:
        num_rows = int(len(df_full) * fraction)
        df = df_full.head(num_rows).copy()
        logger.info(f"Выбрана доля данных: {fraction * 100}%, что составляет {len(df)} записей.")
    elif offset is not None and limit is not None:
        df = df_full.iloc[offset : offset + limit].copy()
        logger.info(f"Выбран срез данных: {limit} записей с отступом {offset}.")
    else:
        df = df_full
        logger.info(f"Выбраны все данные: {len(df)} записей.")

    if df.empty:
        logger.info("Нет данных для загрузки. Завершение.")
        return

    # --- Шаг 1: Загрузка в PostgreSQL ---
    logger.info("Начало создания счетов в PostgreSQL...")
    df['sender_account'] = df['sender_account'].astype(str)
    df['receiver_account'] = df['receiver_account'].astype(str)
    unique_account_names = set(df['sender_account'].unique()) | set(df['receiver_account'].unique())
    existing_accounts = db.query(models.Account.name).filter(models.Account.name.in_(unique_account_names)).all()
    existing_account_names = {name for (name,) in existing_accounts}
    new_account_names = unique_account_names - existing_account_names
    
    if new_account_names:
        logger.info(f"Создание {len(new_account_names)} новых счетов в PostgreSQL...")
        db.bulk_insert_mappings(models.Account, [{'name': name} for name in new_account_names])
        db.commit()
    
    logger.info("Начало создания транзакций в PostgreSQL...")
    all_accounts = db.query(models.Account.id, models.Account.name).all()
    account_map = {name: id for id, name in all_accounts}

    transactions_to_create = []
    for _, row in df.iterrows():
        sender_id = account_map.get(row['sender_account'])
        receiver_id = account_map.get(row['receiver_account'])
        
        if sender_id and receiver_id:
            transactions_to_create.append({
                'from_account_id': sender_id,
                'to_account_id': receiver_id,
                'amount': row['amount'],
                'datetime': row['datetime'],
                'payment_type': row['payment_type'],
                'payment_currency_iso': row['payment_currency_iso'],
                'received_currency_iso': row['received_currency_iso'],
                'sender_bank_location': row['sender_bank_location'],
                'receiver_bank_location': row['receiver_bank_location'],
                'is_fraud': bool(row['predicted_label']),
                'fraud_probability': float(row['predicted_probability'])
            })

    if transactions_to_create:
        logger.info(f"Начало пакетной вставки {len(transactions_to_create)} транзакций в PostgreSQL...")
        total_pg_tx = len(transactions_to_create)
        batch_size_pg = 50000  # Размер пакета для PostgreSQL

        for i in range(0, total_pg_tx, batch_size_pg):
            batch = transactions_to_create[i:i + batch_size_pg]
            db.bulk_insert_mappings(models.Transaction, batch)
            db.commit()
            logger.info(f"Загружено в PostgreSQL: {min(i + batch_size_pg, total_pg_tx)} / {total_pg_tx} транзакций...")

        logger.info("Посев данных в PostgreSQL успешно завершен.")

    # --- Шаг 2: Перенос данных из PostgreSQL в Neo4j ---
    logger.info("Начало переноса данных в Neo4j...")
    neo4j_driver = None
    try:
        neo4j_driver = get_neo4j_driver()
        with neo4j_driver.session() as session:
            logger.info("Очистка существующих данных в Neo4j...")
            session.run("MATCH (n) DETACH DELETE n")

            logger.info("Создание индекса для Account(name) для ускорения...")
            session.run("CREATE INDEX account_name_index IF NOT EXISTS FOR (n:Account) ON (n.name)")

            # Оптимизация: Создаем обратный словарь для быстрого поиска имени по ID
            logger.info("Создание обратного словаря для ID счетов для ускорения...")
            id_to_account_map = {v: k for k, v in account_map.items()}

            logger.info("Подготовка и загрузка данных в Neo4j порциями...")
            batch_size = 10000  # Обрабатываем по 10 000 транзакций за раз
            total_transactions = len(transactions_to_create)

            for i in range(0, total_transactions, batch_size):
                batch = transactions_to_create[i:i + batch_size]
                tx_list_for_neo4j = []
                
                for tx_data in batch:
                    sender_name = id_to_account_map.get(tx_data['from_account_id'])
                    receiver_name = id_to_account_map.get(tx_data['to_account_id'])
                    
                    if sender_name and receiver_name:
                        tx_list_for_neo4j.append({
                            "sender_name": str(sender_name),
                            "receiver_name": str(receiver_name),
                            "amount": tx_data['amount'],
                            "datetime": tx_data['datetime'].isoformat(),
                            "risk_score": tx_data['fraud_probability'],
                            "is_fraud": tx_data['is_fraud']
                        })

                if tx_list_for_neo4j:
                    # Используем MERGE для всего пути, чтобы гарантировать идемпотентность
                    # и избежать дубликатов. Мы делаем MERGE для транзакции по ее ключевым
                    # свойствам, которые делают ее уникальной (например, время, сумма, участники).
                    session.run("""
                        UNWIND $tx_list as tx
                        MERGE (sender:Account {name: tx.sender_name})
                        MERGE (receiver:Account {name: tx.receiver_name})
                        // MERGE на узле транзакции, чтобы избежать дублей при повторном запуске
                        MERGE (sender)-[:SENT]->(t:Transaction {
                            amount: tx.amount, 
                            datetime: datetime(tx.datetime)
                        })-[:TO]->(receiver)
                        // ON CREATE устанавливает свойства только при первом создании узла
                        ON CREATE SET t.risk_score = tx.risk_score, t.is_fraud = tx.is_fraud
                        """, tx_list=tx_list_for_neo4j)
                        
                logger.info(f"Загружено в Neo4j: {min(i + batch_size, total_transactions)} / {total_transactions} транзакций...")

            logger.info("Перенос данных в Neo4j успешно завершен.")

    except Exception as e:
        logger.error(f"Ошибка при заполнении Neo4j: {e}", exc_info=True)
    finally:
        if neo4j_driver:
            neo4j_driver.close()

    # --- Шаг 3: Обновление состояния загрузки ---
    state = db.query(DataLoadingState).first()
    if not state:
        logger.info("Создание начальной записи о состоянии загрузки.")
        state = DataLoadingState()
        db.add(state)

    end_index = offset + len(df) if offset is not None else len(df)
    state.last_processed_index = end_index
    logger.info(f"Обновление last_processed_index на {end_index}.")
    db.commit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скрипт для заполнения БД данными из CSV.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fraction", type=float, help="Какую долю данных загрузить (например, 0.8 для 80%%).")
    group.add_argument("--chunk", nargs=2, type=int, metavar=('OFFSET', 'LIMIT'), help="Загрузить срез данных: отступ и количество.")
    parser.add_argument("--clear-first", action="store_true", help="Полностью очистить таблицы PostgreSQL перед заполнением.")

    args = parser.parse_args()

    db = SessionLocal()
    try:
        if args.clear_first:
            logger.info("Обнаружен флаг --clear-first. Очистка таблиц PostgreSQL...")
            Base.metadata.drop_all(bind=engine)
            Base.metadata.create_all(bind=engine)
            logger.info("Таблицы PostgreSQL успешно очищены и созданы заново.")

        if args.fraction:
            seed_data(db, fraction=args.fraction)
        elif args.chunk:
            seed_data(db, offset=args.chunk[0], limit=args.chunk[1])
        else:
            seed_data(db) # Загрузка всех данных по умолчанию
    finally:
        db.close()