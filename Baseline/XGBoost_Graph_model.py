import pandas as pd
import networkx as nx
from XGBoost_model import XGBoostBaseline

class XGBoostWithGFP:
    def __init__(self, df_path):
        self.df_path = df_path
        self.model = XGBoostBaseline()

    def load_and_prepare(self):
        df = pd.read_csv(self.df_path)
        G = nx.from_pandas_edgelist(
            df,
            source="Sender_account",
            target="Receiver_account",
            create_using=nx.DiGraph()
        )

        degree_dict = dict(G.degree())
        in_degree_dict = dict(G.in_degree())
        out_degree_dict = dict(G.out_degree())

        graph_features = pd.DataFrame({
            "Sender_account": list(degree_dict.keys()),
            "node_degree": list(degree_dict.values()),
            "node_in_degree": list(in_degree_dict.values()),
            "node_out_degree": list(out_degree_dict.values())
        })

        df = df.merge(graph_features, on="Sender_account", how="left")
        self.data = self.model.prepare_data(df)

    def train_and_evaluate(self):
        self.model.train(self.data["X_train"], self.data["y_train"])

        metrics_bal = self.model.evaluate(self.data["X_test_bal"], self.data["y_test_bal"])
        metrics_real = self.model.evaluate(self.data["X_test_real"], self.data["y_test_real"])

        return metrics_bal, metrics_real

    def save_model(self, path="xgboost_model.pkl"):
        self.model.save_model(path)
