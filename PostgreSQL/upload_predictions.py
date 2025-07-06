import pandas as pd
import torch
import numpy as np
import pickle
import os
import re
import sys

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# --- 1. Определение архитектуры модели и функций --- 
# Это необходимо, чтобы PyTorch мог правильно загрузить веса и данные
# Добавляем путь к папке с моделью, чтобы можно было импортировать EdgeClassifierSAGE
# (Предполагается, что скрипт запускается из папки PostgreSQL)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Model', 'GNN_model_v2')))

from torch_geometric.nn import SAGEConv, BatchNorm
from torch.nn import Sequential, Linear, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.data import Data

class EdgeClassifierSAGE(torch.nn.Module):
    def __init__(self, in_channels_node, in_channels_edge, hidden_channels=256, out_channels=2):
        super().__init__()
        self.sage1 = SAGEConv(in_channels_node, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.sage3 = SAGEConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        mlp_input_dim = (hidden_channels * 2) + in_channels_edge
        self.edge_mlp = Sequential(
            Linear(mlp_input_dim, 512),
            ReLU(),
            Dropout(0.5),
            Linear(512, 256),
            ReLU(),
            Dropout(0.4),
            Linear(256, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.sage1(x, edge_index).relu()
        x = self.bn1(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.sage2(x, edge_index).relu()
        x = self.bn2(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.sage3(x, edge_index).relu()
        x = self.bn3(x)
        sender_emb = x[edge_index[0]]
        receiver_emb = x[edge_index[1]]
        edge_input = torch.cat([sender_emb, receiver_emb, edge_attr], dim=1)
        return self.edge_mlp(edge_input)

def build_graph_data_for_prediction(df_subset, node_features_df, account_to_id_map, edge_encoder, edge_scaler):
    cat_cols = ['payment_type', 'payment_currency_iso', 'received_currency_iso', 'sender_bank_location', 'receiver_bank_location', 'pattern_type']
    encoded_cat = edge_encoder.transform(df_subset[cat_cols])
    numerical_cols = ['amount', 'log_amount', 'hour', 'time_since_last_sent', 'time_since_last_received']
    scaled_num = edge_scaler.transform(df_subset[numerical_cols])
    edge_attr = torch.tensor(np.concatenate([scaled_num, encoded_cat], axis=1), dtype=torch.float)
    node_ids = list(account_to_id_map.keys())
    x_df = pd.DataFrame(index=node_ids).join(node_features_df).fillna(0)
    x = torch.tensor(x_df.values, dtype=torch.float)
    df_subset['sender_id'] = df_subset['sender_account'].map(account_to_id_map)
    df_subset['receiver_id'] = df_subset['receiver_account'].map(account_to_id_map)
    edge_index = torch.tensor(df_subset[['sender_id', 'receiver_id']].values.T, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# --- 2. Основной скрипт --- 
def main():
    print("--- Проект ShAMLock: Загрузка предсказаний ---")
    
    BASE_PATH = "c:/Users/ostro/Documents/my_first_data_project/"
    DATASET_PATH = os.path.join(BASE_PATH, "Datasets/df_baseline.csv")
    ARTIFACTS_PATH = os.path.join(BASE_PATH, "Model/GNN_model_v2/artifacts/")
    MODEL_PATH = os.path.join(BASE_PATH, "Model/GNN_model_v2/best_model.pth")
    TEST_IDX_PATH = os.path.join(BASE_PATH, "Model/test_idx.pkl")

    DB_URL = "postgresql+psycopg2://postgres:8495@localhost:5432/aml_project"
    engine = create_engine(DB_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    print("1. Загрузка данных и артефактов...")
    with open(os.path.join(ARTIFACTS_PATH, 'account_to_id.pkl'), 'rb') as f: account_to_id = pickle.load(f)
    with open(os.path.join(ARTIFACTS_PATH, 'edge_encoder.pkl'), 'rb') as f: edge_encoder = pickle.load(f)
    with open(os.path.join(ARTIFACTS_PATH, 'edge_scaler.pkl'), 'rb') as f: edge_scaler = pickle.load(f)
    with open(os.path.join(ARTIFACTS_PATH, 'best_threshold.pkl'), 'rb') as f: best_threshold = pickle.load(f)
    node_features_scaled_df = pd.read_pickle(os.path.join(ARTIFACTS_PATH, 'node_features.pkl'))

    df = pd.read_csv(DATASET_PATH)
    with open(TEST_IDX_PATH, "rb") as f: test_idx = pickle.load(f)
    df_test = df.loc[test_idx].copy()

    print("2. Предобработка тестовых данных...")
    df_test.columns = df_test.columns.str.strip().str.lower()
    df_test['datetime'] = pd.to_datetime(df_test['date'] + ' ' + df_test['time'])
    df_test = df_test.sort_values('datetime').reset_index(drop=True)
    def get_pattern_type(laundering_type):
        if not isinstance(laundering_type, str): return 'unknown'
        pattern = laundering_type.replace('Normal_', '')
        return re.sub(r'_\d$', '', pattern)
    df_test['pattern_type'] = df_test['laundering_type'].apply(get_pattern_type)
    df_test['log_amount'] = np.log1p(df_test['amount'])
    df_test['hour'] = df_test['datetime'].dt.hour
    df_test['time_since_last_sent'] = df_test.groupby('sender_account')['datetime'].diff().dt.total_seconds().fillna(0)
    df_test['time_since_last_received'] = df_test.groupby('receiver_account')['datetime'].diff().dt.total_seconds().fillna(0)
    
    test_graph = build_graph_data_for_prediction(df_test, node_features_scaled_df, account_to_id, edge_encoder, edge_scaler)

    print("3. Загрузка модели и получение предсказаний...")
    device = torch.device('cpu') # Для инференса CPU обычно достаточно
    model = EdgeClassifierSAGE(test_graph.x.size(1), test_graph.edge_attr.size(1)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    with torch.no_grad():
        test_graph = test_graph.to(device)
        out = model(test_graph.x, test_graph.edge_index, test_graph.edge_attr)
        probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
    labels = (probs > best_threshold).astype(bool)

    print("4. Получение transaction_id из базы данных...")
    # Загружаем транзакции из БД, чтобы получить правильные ID
    db_transactions = pd.read_sql("SELECT transaction_id, timestamp FROM transaction ORDER BY timestamp ASC", engine)
    
    if len(db_transactions) != len(df_test):
        print(f"\n\n❌ ОШИБКА: Несовпадение количества транзакций! В БД: {len(db_transactions)}, в тестовом наборе: {len(df_test)}")
        return

    print("5. Формирование итогового датафрейма...")
    predictions_df = pd.DataFrame({
        'transaction_id': db_transactions['transaction_id'],
        'predicted_label': labels,
        'predicted_proba': probs
    })

    print("6. Загрузка предсказаний в базу данных...")
    try:
        session.execute(text('TRUNCATE TABLE prediction RESTART IDENTITY;'))
        predictions_df.to_sql('prediction', engine, if_exists='append', index=False)
        session.commit()
        print("\n✅ Предсказания успешно загружены в базу данных!")
    except Exception as e:
        print(f"\n❌ Ошибка при загрузке в БД: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    main()
