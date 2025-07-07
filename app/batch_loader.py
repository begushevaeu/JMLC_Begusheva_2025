import pandas as pd
import logging
import os
from sqlalchemy.orm import Session
from .database import SessionLocal
from .models import Account, Transaction, DataLoadingState
from .database import get_neo4j_driver

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Константы
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE_PATH = os.path.join(BASE_DIR, 'Datasets', 'test_predictions.csv')
BATCH_SIZE = 1000  # Размер одной порции данных для загрузки

def load_batch():
    """Загружает один батч данных в PostgreSQL и Neo4j."""
    logger.info("Попытка запуска фоновой задачи: load_batch")
    db = SessionLocal()

    # Проверяем, была ли уже начальная загрузка
    state = db.query(DataLoadingState).first()
    if not state or state.last_processed_index == 0:
        logger.info("Начальная загрузка данных (seed) еще не завершена. Пропускаем выполнение batch_loader.")
        db.close()
        return

    logger.info("Начальная загрузка завершена. Запускаем batch_loader.")
    neo4j_driver = None
    try:
        # Шаг 1: Получить текущее состояние загрузки
        state = db.query(DataLoadingState).first()
        if not state:
            logger.warning("Состояние загрузки не найдено. Процесс не может быть запущен.")
            return

        last_processed_index = state.last_processed_index
        logger.info(f"Текущий индекс последней обработанной строки: {last_processed_index}")

        # Шаг 2: Прочитать следующий батч из CSV
        df_chunk = pd.read_csv(
            DATA_FILE_PATH, 
            skiprows=range(1, last_processed_index + 1), 
            nrows=BATCH_SIZE,
            header=0
        )

        if df_chunk.empty:
            logger.info("Все данные уже загружены. Завершение работы.")
            return

        logger.info(f"Прочитано {len(df_chunk)} новых строк для обработки.")
        df_chunk['datetime'] = pd.to_datetime(df_chunk['datetime'])

        # Шаг 3: Загрузка в PostgreSQL (логика похожа на seed.py)
        # --- Загрузка в PostgreSQL ---
        logger.info("Начало загрузки в PostgreSQL...")
        df_chunk['sender_account'] = df_chunk['sender_account'].astype(str)
        df_chunk['receiver_account'] = df_chunk['receiver_account'].astype(str)
        unique_account_names = set(df_chunk['sender_account'].unique()) | set(df_chunk['receiver_account'].unique())
        
        existing_accounts = db.query(Account.name).filter(Account.name.in_(unique_account_names)).all()
        existing_account_names = {name for (name,) in existing_accounts}
        new_account_names = unique_account_names - existing_account_names

        if new_account_names:
            logger.info(f"Создание {len(new_account_names)} новых счетов в PostgreSQL...")
            db.bulk_insert_mappings(Account, [{'name': name} for name in new_account_names])
            db.commit() # Коммитим счета сразу, чтобы получить их ID

        all_accounts_in_batch = db.query(Account.id, Account.name).filter(Account.name.in_(unique_account_names)).all()
        account_map = {name: id for id, name in all_accounts_in_batch}

        transactions_to_create = []
        for _, row in df_chunk.iterrows():
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
            logger.info(f"Вставка {len(transactions_to_create)} транзакций в PostgreSQL...")
            db.bulk_insert_mappings(Transaction, transactions_to_create)
            # Коммит будет в конце, после Neo4j


        # Шаг 4: Загрузка в Neo4j
        # --- Загрузка в Neo4j ---
        logger.info("Начало загрузки в Neo4j...")
        neo4j_driver = get_neo4j_driver()
        with neo4j_driver.session() as session:
            # Создание счетов
            for account_name in unique_account_names:
                session.run("MERGE (a:Account {name: $name})", name=account_name)
            logger.info(f"Создано/обновлено {len(unique_account_names)} узлов счетов в Neo4j.")

            # Создание транзакций
            for tx_data in transactions_to_create:
                # Получаем имена счетов по ID для графа
                sender_name = next((name for name, id in account_map.items() if id == tx_data['from_account_id']), None)
                receiver_name = next((name for name, id in account_map.items() if id == tx_data['to_account_id']), None)
                
                if sender_name and receiver_name:
                    session.run("""
                        MATCH (sender:Account {name: $sender_name})
                        MATCH (receiver:Account {name: $receiver_name})
                        CREATE (sender)-[t:TRANSACTION {
                            amount: $amount, 
                            datetime: $datetime, 
                            is_fraud: $is_fraud,
                            fraud_probability: $fraud_probability
                        }]->(receiver)
                    """, 
                    sender_name=sender_name, 
                    receiver_name=receiver_name, 
                    amount=tx_data['amount'],
                    datetime=tx_data['datetime'].isoformat(),
                    is_fraud=tx_data['is_fraud'],
                    fraud_probability=tx_data['fraud_probability']
                    )
            logger.info(f"Создано {len(transactions_to_create)} транзакций и связей в Neo4j.")


        # Шаг 5: Обновить состояние
        end_index = last_processed_index + len(df_chunk)
        state.last_processed_index = end_index
        logger.info(f"Индекс обновлен до {end_index}.")
        db.commit()
        logger.info(f"Процесс успешно завершен. Новый индекс: {state.last_processed_index}")

    except Exception as e:
        logger.error(f"Ошибка во время загрузки батча: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()
        if neo4j_driver:
            neo4j_driver.close()
