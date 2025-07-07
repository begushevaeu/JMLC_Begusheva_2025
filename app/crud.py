from datetime import datetime
from sqlalchemy import or_
from sqlalchemy.orm import aliased
from sqlalchemy.orm import Session, joinedload
from typing import List
from . import models, schemas
from .neo4j_utils import add_transaction_to_graph

# --- Операции для Счетов ---

def get_account(db: Session, account_id: int):
    """Получает счет по его числовому ID."""
    return db.query(models.Account).filter(models.Account.id == account_id).first()

def get_account_by_name(db: Session, name: str):
    """Получает счет по его уникальному имени."""
    return db.query(models.Account).filter(models.Account.name == name).first()

def get_fraudulent_transactions(db: Session, skip: int = 0, limit: int = 100) -> List[models.Transaction]:
    """
    Получение списка транзакций, помеченных как мошеннические.
    """
    return db.query(models.Transaction).filter(models.Transaction.is_fraud == True).order_by(models.Transaction.datetime.desc()).offset(skip).limit(limit).all()

def get_all_accounts_by_role(db: Session, role: str):
    """
    Получает список всех счетов, которые выступали в определенной роли.
    :param role: 'sender' или 'receiver'.
    """
    if role == 'sender':
        account_ids_query = db.query(models.Transaction.from_account_id).distinct()
    elif role == 'receiver':
        account_ids_query = db.query(models.Transaction.to_account_id).distinct()
    else:
        return []
    
    # Получаем имена счетов по их ID
    accounts = db.query(models.Account.name).filter(models.Account.id.in_(account_ids_query)).all()
    return [acc[0] for acc in accounts]

def get_accounts(db: Session, skip: int = 0, limit: int = 100):
    """Получает список счетов."""
    return db.query(models.Account).offset(skip).limit(limit).all()

def get_accounts_count(db: Session):
    return db.query(models.Account).count()

def create_account(db: Session, account: schemas.AccountCreate) -> models.Account:
    """Создает новый счет в базе данных."""
    db_account = models.Account(name=account.name)
    db.add(db_account)
    db.commit()
    db.refresh(db_account)
    return db_account

# --- Операции для Транзакций ---

def get_transaction(db: Session, transaction_id: int):
    """Получает транзакцию по ее ID с предзагрузкой данных отправителя и получателя."""
    return (
        db.query(models.Transaction)
        .options(joinedload(models.Transaction.sender), joinedload(models.Transaction.receiver))
        .filter(models.Transaction.id == transaction_id)
        .first()
    )

def get_transactions(db: Session, skip: int = 0, limit: int = 100):
    """Получает список транзакций."""
    return db.query(models.Transaction).offset(skip).limit(limit).all()

def get_transactions_by_sender_receiver(db: Session, sender_name: str = None, receiver_name: str = None, limit: int = 100):
    """Получает транзакции по отправителю и/или получателю с присоединением risk_score."""
    query = db.query(
        models.Transaction,
        models.Alert.risk_score
    ).outerjoin(models.Alert, models.Transaction.id == models.Alert.transaction_id).options(
        joinedload(models.Transaction.sender),
        joinedload(models.Transaction.receiver)
    )

    if sender_name:
        SenderAccount = aliased(models.Account)
        query = query.join(SenderAccount, models.Transaction.from_account_id == SenderAccount.id)
        query = query.filter(SenderAccount.name == sender_name)

    if receiver_name:
        ReceiverAccount = aliased(models.Account)
        query = query.join(ReceiverAccount, models.Transaction.to_account_id == ReceiverAccount.id)
        query = query.filter(ReceiverAccount.name == receiver_name)

    results = query.order_by(models.Transaction.datetime.desc()).limit(limit).all()

    # Результат теперь - список кортежей (Transaction, risk_score). Преобразуем его.
    transactions = []
    for tx, risk_score in results:
        tx.risk_score = risk_score  # Динамически добавляем атрибут
        transactions.append(tx)

    return transactions

def get_transactions_by_account_name(db: Session, account_name: str, limit: int = 100):
    """Получает все транзакции для указанного счета (отправленные и полученные)."""
    # Сначала найдем сам счет, чтобы получить его ID
    account = get_account_by_name(db, name=account_name)
    if not account:
        return [] # Если счет не найден, возвращаем пустой список

    # Ищем все транзакции, где ID счета совпадает либо с отправителем, либо с получателем
    return (
        db.query(models.Transaction)
        .options(joinedload(models.Transaction.sender), joinedload(models.Transaction.receiver))
        .filter(or_(models.Transaction.from_account_id == account.id, models.Transaction.to_account_id == account.id))
        .order_by(models.Transaction.datetime.desc()) # Сортируем по дате, от новых к старым
        .limit(limit)
        .all()
    )

def get_counterparty_accounts(db: Session, account_name: str, relationship: str):
    """
    Находит все счета-контрагенты для заданного счета.
    :param account_name: Имя основного счета (например, 'C12345').
    :param relationship: 'sender' (найти всех получателей) или 'receiver' (найти всех отправителей).
    """
    account = get_account_by_name(db, name=account_name)
    if not account:
        return []

    if relationship == 'sender':
        # Ищем всех уникальных получателей для этого отправителя
        query = (
            db.query(models.Account.name)
            .join(models.Transaction, models.Transaction.to_account_id == models.Account.id)
            .filter(models.Transaction.from_account_id == account.id)
            .distinct()
        )
    elif relationship == 'receiver':
        # Ищем всех уникальных отправителей для этого получателя
        query = (
            db.query(models.Account.name)
            .join(models.Transaction, models.Transaction.from_account_id == models.Account.id)
            .filter(models.Transaction.to_account_id == account.id)
            .distinct()
        )
    else:
        return []

    results = query.all()
    # results будет списком кортежей, например [('C67890',), ('C54321',)]
    return [result[0] for result in results]

def create_transaction(db: Session, transaction: schemas.TransactionCreate) -> models.Transaction:
    """
    Создает новую транзакцию, сохраняет ее в PostgreSQL и добавляет в граф Neo4j.
    Транзакция может быть создана только между существующими счетами.
    """
    # 1. Проверяем, существуют ли оба счета в базе данных
    sender_account = get_account(db, account_id=transaction.from_account_id)
    receiver_account = get_account(db, account_id=transaction.to_account_id)

    if not sender_account or not receiver_account:
        # В реальном приложении здесь лучше выбросить HTTPException
        raise ValueError("Один или оба счета не существуют. Транзакция не может быть создана.")

    # 2. Создаем объект транзакции SQLAlchemy на основе схемы Pydantic
    db_transaction = models.Transaction(**transaction.model_dump())
    
    # 3. Сохраняем транзакцию в PostgreSQL
    db.add(db_transaction)
    db.commit()
    db.refresh(db_transaction)

    # 4. Добавляем транзакцию в граф Neo4j
    # Готовим данные в формате, который ожидает seed_neo4j
    tx_data_for_graph = {
        'orig_id': sender_account.id,
        'orig_name': sender_account.name,
        'dest_id': receiver_account.id,
        'dest_name': receiver_account.name,
        'amount': db_transaction.amount,
        'timestamp': db_transaction.datetime.isoformat(),
        'isFraud': db_transaction.is_fraud,
        'alert_id': None, # Предполагаем, что при создании новой транзакции алерта еще нет
        'payment_type': db_transaction.payment_type,
        'currency': db_transaction.payment_currency_iso
    }
    
    # Вызываем утилиту для добавления транзакции в Neo4j
    add_transaction_to_graph(tx_data_for_graph)

    return db_transaction

# --- Операции для Оповещений ---

def get_alert(db: Session, alert_id: int):
    """Получает оповещение по его ID."""
    return db.query(models.Alert).filter(models.Alert.id == alert_id).first()

def get_alerts(db: Session, skip: int = 0, limit: int = 100):
    """Получает список оповещений с предзагрузкой транзакций, отправителей и получателей."""
    return (
        db.query(models.Alert)
        .options(
            joinedload(models.Alert.transaction).joinedload(models.Transaction.sender),
            joinedload(models.Alert.transaction).joinedload(models.Transaction.receiver)
        )
        .order_by(models.Alert.timestamp.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

def create_alert(db: Session, transaction_id: int, risk_score: float) -> models.Alert | None:
    """
    Создает новое оповещение, если оно еще не существует для данной транзакции.
    Возвращает созданное оповещение или None, если оно уже существует.
    """
    existing_alert = db.query(models.Alert).filter(models.Alert.transaction_id == transaction_id).first()
    if existing_alert:
        return None

    db_alert = models.Alert(transaction_id=transaction_id, risk_score=risk_score)
    db.add(db_alert)
    db.commit()
    db.refresh(db_alert)
    return db_alert