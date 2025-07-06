from sqlalchemy import create_engine, Column, String, Float, Integer, Boolean, ForeignKey, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

# Подключение к PostgreSQL
engine = create_engine("postgresql+psycopg2://postgres:8495@localhost:5432/aml_project")
Base = declarative_base()

# Таблица Account
class Account(Base):
    __tablename__ = "account"
    account_id = Column(String, primary_key=True)

# Таблица Transaction
class Transaction(Base):
    __tablename__ = "transaction"
    transaction_id = Column(String, primary_key=True)
    sender_account_id = Column(String, ForeignKey("account.account_id", ondelete="CASCADE"))
    receiver_account_id = Column(String, ForeignKey("account.account_id", ondelete="CASCADE"))
    amount = Column(Float)
    payment_type = Column(String)
    payment_currency_iso = Column(String)
    received_currency_iso = Column(String)
    sender_bank_location = Column(String)
    receiver_bank_location = Column(String)
    hour = Column(Integer)
    weekday = Column(Integer)
    month = Column(Integer)
    is_laundering = Column(Boolean)
    timestamp = Column(TIMESTAMP)

    sender = relationship("Account", foreign_keys=[sender_account_id])
    receiver = relationship("Account", foreign_keys=[receiver_account_id])

# Таблица Prediction (если нужно)
class Prediction(Base):
    __tablename__ = "prediction"
    transaction_id = Column(String, ForeignKey("transaction.transaction_id"), primary_key=True)
    score = Column(Float)
    predicted_label = Column(Boolean)

# Создание всех таблиц
Base.metadata.create_all(engine)
print("✅ Все таблицы успешно созданы в базе данных.")