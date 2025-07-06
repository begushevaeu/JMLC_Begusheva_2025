from sqlalchemy import create_engine, Column, String, Float, Integer, Boolean, ForeignKey, TIMESTAMP, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
import pandas as pd
import pickle
import uuid

# 1. Подключение к базе
engine = create_engine("postgresql+psycopg2://postgres:8495@localhost:5432/aml_project")
Base = declarative_base()

# 2. Определяем таблицы
class Account(Base):
    __tablename__ = "account"
    account_id = Column(String, primary_key=True)

class Transaction(Base):
    __tablename__ = "transaction"
    transaction_id = Column(String, primary_key=True)
    sender_account_id = Column(String, ForeignKey("account.account_id"))
    receiver_account_id = Column(String, ForeignKey("account.account_id"))
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

    sender_account = relationship("Account", foreign_keys=[sender_account_id])
    receiver_account = relationship("Account", foreign_keys=[receiver_account_id])

class Prediction(Base):
    __tablename__ = "prediction"
    transaction_id = Column(String, ForeignKey("transaction.transaction_id"), primary_key=True)
    predicted_label = Column(Boolean)
    predicted_proba = Column(Float)

# 3. Удаляем и создаем заново
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)
print("✅ Таблицы созданы в базе данных.")

# 4. Загружаем датасет
df = pd.read_csv("C:/Users/ostro/Documents/my_first_data_project/Datasets/df_baseline.csv")
with open(r"C:\Users\ostro\Documents\my_first_data_project\Model\test_idx.pkl", "rb") as f:
    test_idx = pickle.load(f)

df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df_test = df.loc[test_idx].reset_index(drop=True)

if 'transaction_id' not in df_test.columns:
    df_test['transaction_id'] = [str(uuid.uuid4()) for _ in range(len(df_test))]

# 5. Вставка данных
Session = sessionmaker(bind=engine)
session = Session()

# Вставка аккаунтов
accounts = set(df_test['Sender_account']).union(df_test['Receiver_account'])
session.bulk_save_objects([Account(account_id=str(a)) for a in accounts])
session.commit()

# Вставка транзакций
transactions = []
for _, row in df_test.iterrows():
    t = Transaction(
        transaction_id=row['transaction_id'],
        sender_account_id=str(row['Sender_account']),
        receiver_account_id=str(row['Receiver_account']),
        amount=float(row['Amount']),
        payment_type=row['Payment_type'],
        payment_currency_iso=row['Payment_currency_iso'],
        received_currency_iso=row['Received_currency_iso'],
        sender_bank_location=row['Sender_bank_location'],
        receiver_bank_location=row['Receiver_bank_location'],
        hour=int(row['hour']),
        weekday=int(row['weekday']),
        month=int(row['month']),
        is_laundering=bool(row['Is_laundering']),
        timestamp=row['datetime']
    )
    transactions.append(t)

session.bulk_save_objects(transactions)
session.commit()
session.close()

print("Данные успешно загружены.")