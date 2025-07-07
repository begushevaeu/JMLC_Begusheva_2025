from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Dict, Any

# --- Схемы для Account ---
class AccountBase(BaseModel):
    name: str

class AccountCreate(AccountBase):
    pass

class Account(AccountBase):
    id: int

    class Config:
        from_attributes = True

# --- Схемы для Transaction ---
class TransactionBase(BaseModel):
    from_account_id: int
    to_account_id: int
    amount: float
    datetime: datetime
    payment_type: Optional[str] = None
    payment_currency_iso: Optional[str] = None
    received_currency_iso: Optional[str] = None
    sender_bank_location: Optional[str] = None
    receiver_bank_location: Optional[str] = None

class TransactionCreate(TransactionBase):
    is_fraud: bool = False
    fraud_probability: float = 0.0

class Transaction(TransactionBase):
    id: int
    is_fraud: bool
    fraud_probability: float
    sender: Account
    receiver: Account
    risk_score: Optional[float] = None

    class Config:
        from_attributes = True

# --- Схемы для Оповещений (Alert) ---

class AlertBase(BaseModel):
    transaction_id: int
    risk_score: float
    status: str = "new"
    comment: Optional[str] = None

class AlertCreate(AlertBase):
    pass

class Alert(AlertBase):
    id: int
    timestamp: datetime
    transaction: Transaction

    class Config:
        from_attributes = True

# --- Схемы для Графа (Graph) ---

class GraphRequest(BaseModel):
    account_ids: list[str]
    depth: int = 1

class GraphResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]

class MaxDepthRequest(BaseModel):
    account_id: str

class MaxDepthResponse(BaseModel):
    max_depth: int = 1