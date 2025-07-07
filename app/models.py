import datetime
from sqlalchemy import (
    Column, Integer, Float, DateTime, ForeignKey, Boolean, String, func
)
from sqlalchemy.orm import relationship
from .database import Base

class Account(Base):
    __tablename__ = "accounts"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)

    sent_transactions = relationship("Transaction", foreign_keys="[Transaction.from_account_id]", back_populates="sender")
    received_transactions = relationship("Transaction", foreign_keys="[Transaction.to_account_id]", back_populates="receiver")

class Transaction(Base):
    __tablename__ = 'transactions'

    id = Column(Integer, primary_key=True, index=True)
    from_account_id = Column(Integer, ForeignKey('accounts.id'), nullable=False)
    to_account_id = Column(Integer, ForeignKey('accounts.id'), nullable=False)
    amount = Column(Float, nullable=False)
    datetime = Column(DateTime, nullable=False, index=True)
    
    payment_type = Column(String)
    payment_currency_iso = Column(String(3))
    received_currency_iso = Column(String(3))
    sender_bank_location = Column(String)
    receiver_bank_location = Column(String)

    is_fraud = Column(Boolean, default=False, nullable=False)
    fraud_probability = Column(Float, default=0.0, nullable=False)

    sender = relationship("Account", foreign_keys=[from_account_id], back_populates="sent_transactions")
    receiver = relationship("Account", foreign_keys=[to_account_id], back_populates="received_transactions")

    alert = relationship("Alert", back_populates="transaction", uselist=False)

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"))
    risk_score = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    # Отношения
    transaction = relationship("Transaction", back_populates="alert")


class DataLoadingState(Base):
    __tablename__ = 'data_loading_state'

    id = Column(Integer, primary_key=True)
    last_processed_index = Column(Integer, nullable=False, default=0)