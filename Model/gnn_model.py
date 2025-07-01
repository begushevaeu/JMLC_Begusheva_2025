import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch.nn import Linear, ReLU, Sequential, Dropout, BatchNorm1d 
import pandas as pd
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix, roc_curve


def prepare_graph_data(df):
    # Построение графа из DataFrame
    all_accounts = pd.Index(df['Sender_account'].tolist() + df['Receiver_account'].tolist()).unique()
    account_to_id = {acc: i for i, acc in enumerate(all_accounts)}
    df['sender_id'] = df['Sender_account'].map(account_to_id)
    df['receiver_id'] = df['Receiver_account'].map(account_to_id)

    edge_index = torch.tensor(df[['sender_id', 'receiver_id']].values.T, dtype=torch.long)
    from sklearn.preprocessing import OrdinalEncoder

    cat_cols = ['Payment_type', 'Payment_currency_iso', 'Received_currency_iso',
                'Sender_bank_location', 'Receiver_bank_location'] # категориальные признаки

    df_feat = df.copy()
    df_feat[cat_cols] = df_feat[cat_cols].fillna('missing')

    encoder = OrdinalEncoder()
    df_feat[cat_cols] = encoder.fit_transform(df_feat[cat_cols]) # кодируем категории

    df_feat['log_amount'] = df_feat['Amount'].apply(lambda x: np.log1p(x)) # добавляем логарифм суммы

    edge_feature_cols = ['Amount', 'log_amount', 'hour', 'weekday', 'month'] + cat_cols
    edge_features = df_feat[edge_feature_cols].astype(float).values
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    num_nodes = len(account_to_id)
    sender_id = df['sender_id'].values
    receiver_id = df['receiver_id'].values # признаки узлов

    amount_by_sender = np.zeros(num_nodes)
    amount_by_receiver = np.zeros(num_nodes)
    np.add.at(amount_by_sender, sender_id, df['Amount'].values)
    np.add.at(amount_by_receiver, receiver_id, df['Amount'].values)

    laundering_counts = np.zeros(num_nodes)
    total_sends = np.zeros(num_nodes)
    np.add.at(laundering_counts, sender_id, df['Is_laundering'].values)
    np.add.at(total_sends, sender_id, 1)
    susp_ratio = laundering_counts / (total_sends + 1e-6)

    deg_out = torch.bincount(torch.tensor(sender_id), minlength=num_nodes).float().unsqueeze(1)
    deg_in = torch.bincount(torch.tensor(receiver_id), minlength=num_nodes).float().unsqueeze(1)
    amount_out = torch.tensor(amount_by_sender).float().unsqueeze(1)
    amount_in = torch.tensor(amount_by_receiver).float().unsqueeze(1)
    susp_score = torch.tensor(susp_ratio).float().unsqueeze(1)

    x = torch.cat([deg_in, deg_out, amount_in, amount_out, susp_score], dim=1)

    y = torch.tensor(df['Is_laundering'].astype(int).values, dtype=torch.long)

    num_edges = len(df)
    perm = torch.randperm(num_edges)
    train_mask = torch.zeros(num_edges, dtype=torch.bool)
    train_mask[perm[:int(0.8 * num_edges)]] = True
    test_mask = ~train_mask

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.train_mask = train_mask
    data.test_mask = test_mask
    return data

class EdgeClassifierGNN(torch.nn.Module):
    def __init__(self, in_channels_node, in_channels_edge, hidden_channels=64):
        super().__init__()
        self.sage1 = SAGEConv(in_channels_node, hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.dropout1 = Dropout(p=0.3)

        self.sage2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.dropout2 = Dropout(p=0.3)

        self.edge_mlp = Sequential(
            Linear(hidden_channels * 2 + in_channels_edge, 128),
            ReLU(),
            Dropout(0.3),
            Linear(128, 64),
            ReLU(),
            Dropout(0.2),
            Linear(64, 2)
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.sage1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.sage2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x_sender = x[edge_index[0]]
        x_receiver = x[edge_index[1]]
        edge_input = torch.cat([x_sender, edge_attr, x_receiver], dim=1)

        return self.edge_mlp(edge_input)

def train_and_eval_model(data, epochs=100, plot=True, patience=5):
    model = EdgeClassifierGNN(data.x.size(1), data.edge_attr.size(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    f1_history = []
    best_f1 = 0
    best_state = None
    no_improve = 0

    for epoch in trange(epochs, desc="Training", leave=False):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # валидация
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index, data.edge_attr)
            preds = logits.argmax(dim=1).cpu()
            f1_epoch = f1_score(data.y[data.test_mask].cpu(), preds[data.test_mask].cpu())
            f1_history.append(f1_epoch)

            if f1_epoch > best_f1:
                best_f1 = f1_epoch
                best_state = model.state_dict()
                no_improve = 0
            else:
                no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping on epoch {epoch+1}: no F1 improvement in {patience} epochs.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("⚠️ Warning: модель не улучшалась — используется последний state.")
    model.load_state_dict(best_state)  # откат к лучшей модели

    # финальная оценка — тот же блок как был
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr)
        probs = F.softmax(logits[data.test_mask], dim=1)[:, 1].cpu().numpy()
        true = data.y[data.test_mask].cpu().numpy()

        thresholds = np.linspace(0.1, 0.9, 81)
        best_f1_final, best_thresh = 0, 0.5
        for t in thresholds:
            preds = (probs >= t).astype(int)
            score = f1_score(true, preds)
            if score > best_f1_final:
                best_f1_final = score
                best_thresh = t

        preds = (probs >= best_thresh).astype(int)
        auc = roc_auc_score(true, probs)
        pr_auc = average_precision_score(true, probs)
        precision = precision_score(true, preds)
        recall = recall_score(true, preds)
        cm = confusion_matrix(true, preds)

        if plot:
            print(f"\n📊 Финальные метрики (на test):")
            print(f"Optimal threshold: {best_thresh:.2f}")
            print(f"F1-score       = {best_f1_final:.4f}")
            print(f"Precision      = {precision:.4f}")
            print(f"Recall         = {recall:.4f}")
            print(f"ROC-AUC        = {auc:.4f}")
            print(f"PR-AUC         = {pr_auc:.4f}")
            print("Confusion Matrix:")
            print(cm)

            if auc >= 0.75:
                print("✅ AUC > 0.75 — целевой уровень достигнут!")
            elif auc >= 0.65:
                print("🟡 AUC > 0.65 — минимальный уровень пройден.")
            else:
                print("❌ AUC ниже 0.65 — модель требует улучшения.")

            if best_f1_final >= 0.6:
                print("✅ F1 > 0.6 — модель устойчива.")
            else:
                print("❌ F1 < 0.6 — неустойчивое качество.")

            # F1-график
            plt.figure(figsize=(5, 3))
            plt.plot(f1_history, label="Val F1")
            plt.xlabel("Epoch")
            plt.ylabel("F1-score")
            plt.title("F1 over Epochs")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()

            # ROC-кривая
            fpr, tpr, _ = roc_curve(true, probs)
            plt.figure(figsize=(5, 4))
            plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

    return best_f1_final, auc