from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import torch
import joblib
from Model.GNN_model import EdgeClassifierGNN, prepare_graph_for_inference

encoder = joblib.load('./Model/encoder.pkl')
account_to_id = joblib.load('./account_to_id.pkl')

app = FastAPI()

# Загрузка модели и препроцессоров при старте
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = joblib.load('encoder.pkl')
account_to_id = joblib.load('account_to_id.pkl')

model = EdgeClassifierGNN(
    in_channels_node=5,
    in_channels_edge=encoder.transformers_input_dim if hasattr(encoder, 'transformers_input_dim') else None
)
model.load_state_dict(torch.load('edge_classifier_gnn.pt', map_location=device))
model.to(device)
model.eval()

# Pydantic модель для входящих данных (пример, подкорректируй по реальной структуре)
class Transaction(BaseModel):
    Date: str
    Time: str
    Sender_account: str
    Receiver_account: str
    Amount: float
    Payment_type: str
    Payment_currency_iso: str
    Received_currency_iso: str
    Sender_bank_location: str
    Receiver_bank_location: str
    hour: int
    weekday: int
    month: int
    Is_laundering: int  # можно убрать, если только предсказываем

class TransactionsRequest(BaseModel):
    transactions: List[Transaction]

@app.post("/predict")
async def predict(transactions_request: TransactionsRequest):
    # Конвертация в DataFrame
    df = pd.DataFrame([t.dict() for t in transactions_request.transactions])
    
    # Подготовка графа для инференса
    data = prepare_graph_for_inference(df, encoder, account_to_id)
    data = data.to(device)
    
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr, data.edge_label_index)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    
    # Возвращаем вероятности в списке
    return {"probabilities": probs.tolist()}