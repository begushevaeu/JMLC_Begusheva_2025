from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from typing import List

from . import models, schemas, crud
from .database import engine, get_db
from .routers import graph
from .batch_loader import load_batch
from apscheduler.schedulers.background import BackgroundScheduler

app = FastAPI(
    title="ShAMLock: AML Service",
    description="Сервис для обнаружения подозрительных транзакций с использованием GNN.",
    version="1.0.0",
)

# from PIL import Image
# logo = Image.open("static/shamlock_logo.png")  # положи логотип в папку `static/`






# Настройка CORS
origins = [
    "http://localhost",
    "http://localhost:8501",  # Стандартный порт Streamlit
    "http://127.0.0.1",
    "http://127.0.0.1:8501",
    "http://localhost:8050",  # Добавлено для подключения дашборда
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(graph.router)

# --- Фоновые задачи с APScheduler ---

scheduler = BackgroundScheduler(daemon=True)

@app.on_event("startup")
def start_scheduler():
    """Запускает планировщик фоновых задач при старте приложения."""
    # Добавляем задачу, которая будет выполняться каждый час
    scheduler.add_job(load_batch, 'interval', hours=1)
    scheduler.start()
    print("Планировщик фоновых задач запущен.")

@app.on_event("shutdown")
def shutdown_scheduler():
    """Останавливает планировщик при завершении работы приложения."""
    scheduler.shutdown()
    print("Планировщик фоновых задач остановлен.")




@app.get("/", include_in_schema=False)
async def root():
    """Перенаправляет на страницу документации API."""
    return RedirectResponse(url="/docs")


# --- Эндпоинты для Счетов ---

@app.post("/accounts/", response_model=schemas.Account, tags=["Accounts"])
def create_new_account(account: schemas.AccountCreate, db: Session = Depends(get_db)):
    """Создание нового счета по его имени."""
    db_account = crud.get_account_by_name(db, name=account.name)
    if db_account:
        raise HTTPException(status_code=400, detail="Счет с таким именем уже существует")
    return crud.create_account(db=db, account=account)

@app.get("/accounts/", response_model=List[schemas.Account], tags=["Accounts"])
def read_accounts(skip: int = 0, limit: int = 1000, db: Session = Depends(get_db)):
    """
    Получение списка счетов. По умолчанию лимит 1000.
    """
    accounts = crud.get_accounts(db, skip=skip, limit=limit)
    return accounts


@app.get("/accounts/by_role", response_model=List[str], tags=["Accounts"])
def read_accounts_by_role(role: str, db: Session = Depends(get_db)):
    """
    Получение списка счетов по их роли в транзакциях (sender или receiver).
    """
    if role not in ['sender', 'receiver']:
        raise HTTPException(status_code=400, detail="Роль должна быть 'sender' или 'receiver'")
    accounts = crud.get_all_accounts_by_role(db, role=role)
    return accounts

@app.get("/accounts/{account_id}", response_model=schemas.Account, tags=["Accounts"])
def read_account(account_id: int, db: Session = Depends(get_db)):
    """Получение информации о счете по его ID."""
    db_account = crud.get_account(db, account_id=account_id)
    if db_account is None:
        raise HTTPException(status_code=404, detail="Счет не найден")
    return db_account

@app.get("/accounts/{account_name}/transactions", response_model=List[schemas.Transaction], tags=["Accounts"])
def read_account_transactions(account_name: str, db: Session = Depends(get_db)):
    """Получение списка транзакций для конкретного счета (входящие и исходящие)."""
    transactions = crud.get_transactions_by_account_name(db, account_name=account_name)
    return transactions

@app.get("/accounts/{account_name}/counterparties", response_model=List[str], tags=["Accounts"])
def read_counterparty_accounts(account_name: str, relationship: str, db: Session = Depends(get_db)):
    """
    Получение списка счетов-контрагентов.
    - `relationship=sender`: найти всех, кому `account_name` отправлял деньги.
    - `relationship=receiver`: найти всех, от кого `account_name` получал деньги.
    """
    counterparties = crud.get_counterparty_accounts(db, account_name=account_name, relationship=relationship)
    if not counterparties:
        # Возвращаем 200 OK со пустым списком, если контрагентов нет
        return []
    return counterparties

# --- Эндпоинты для Транзакций ---

@app.post("/transactions/", response_model=schemas.Transaction, tags=["Transactions"])
def create_transaction(
    transaction: schemas.TransactionCreate, 
    db: Session = Depends(get_db)
):
    """
    Создание новой транзакции.
    - Проверяет, существуют ли счета отправителя и получателя.
    - Сохраняет транзакцию в основной базе данных.
    - **Примечание**: Интеграция с графовой базой данных Neo4j запланирована на будущие версии.
    """
    try:
        db_transaction = crud.create_transaction(db=db, transaction=transaction)
        return db_transaction
    except ValueError as e:
        # Эта ошибка возникает, если crud.create_transaction не находит один из счетов
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/transactions/", response_model=List[schemas.Transaction], tags=["Transactions"])
def read_transactions_with_filters(
    sender_name: str = None, 
    receiver_name: str = None, 
    db: Session = Depends(get_db)
):
    """Получение списка транзакций с возможностью фильтрации по отправителю и получателю."""
    transactions = crud.get_transactions_by_sender_receiver(
        db, sender_name=sender_name, receiver_name=receiver_name
    )
    return transactions

@app.get("/transactions/fraud", response_model=List[schemas.Transaction], tags=["Transactions"])
def read_fraud_transactions(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Получение списка транзакций, помеченных как мошеннические.
    """
    transactions = crud.get_fraudulent_transactions(db, skip=skip, limit=limit)
    return transactions

@app.get("/transactions/{transaction_id}", response_model=schemas.Transaction, tags=["Transactions"])
def read_transaction(transaction_id: int, db: Session = Depends(get_db)):
    """Получение информации о транзакции по ее ID."""
    db_transaction = crud.get_transaction(db, transaction_id=transaction_id)
    if db_transaction is None:
        raise HTTPException(status_code=404, detail="Транзакция не найдена")
    return db_transaction



# --- Секция для Оповещений (Alerts) ---

@app.get("/alerts/", response_model=List[schemas.Alert], tags=["Alerts"])
def read_alerts(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Получение списка всех оповещений."""
    alerts = crud.get_alerts(db, skip=skip, limit=limit)
    return alerts


@app.get("/alerts/{alert_id}", response_model=schemas.Alert, tags=["Alerts"])
def read_alert(alert_id: int, db: Session = Depends(get_db)):
    """Получение информации об оповещении по его ID."""
    db_alert = crud.get_alert(db, alert_id=alert_id)
    if db_alert is None:
        raise HTTPException(status_code=404, detail="Оповещение не найдено")
    return db_alert


# Дальше здесь будут эндпоинты для работы со счетами и моделью.
# Например, эндпоинт для предсказания:
# @app.post("/predict/", response_model=schemas.Alert)
# def predict_transaction(transaction: schemas.TransactionCreate, db: Session = Depends(get_db)):
#     # Здесь будет логика вызова модели, сохранения транзакции и создания алерта
#     pass

# --- Отладочные эндпоинты ---

@app.post("/debug/run-batch", tags=["Debug"], summary="Принудительный запуск загрузки одного батча")
def trigger_batch_load():
    """
    Этот эндпоинт позволяет вручную запустить процесс загрузки одного батча данных.
    Используйте его для отладки, чтобы не ждать следующего запуска по расписанию.
    """
    try:
        load_batch()
        return {"status": "success", "message": "Загрузка батча успешно запущена и завершена."}
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Произошла ошибка во время выполнения загрузки: {str(e)}"
        )