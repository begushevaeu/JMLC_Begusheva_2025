import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, precision_recall_curve, roc_curve, auc, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

class LightGBMBaseline:
    def __init__(self):
        self.model = lgb.LGBMClassifier(random_state=42)
        self.cat_cols = [
            "Sender_bank_location", "Receiver_bank_location",
            "Payment_type", "Payment_currency_iso", "Received_currency_iso"
        ]

    def preprocess(self, df):
        df = df.drop(columns=["Time", "Date", "Laundering_type"])
        for col in self.cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        return df

    def undersample(self, df, ratio=3):
        df_majority = df[df.Is_laundering == 0]
        df_minority = df[df.Is_laundering == 1]
        df_majority_down = df_majority.sample(n=len(df_minority) * ratio, random_state=42)
        return pd.concat([df_majority_down, df_minority])

    def prepare_data(self, df, test_size=0.2, ratio=3, random_state=42):
        df = self.preprocess(df)

        # Делим до undersample
        df_train_full, df_test_real = train_test_split(
            df, test_size=test_size, stratify=df["Is_laundering"], random_state=random_state
        )

        # Андерсэмплируем только train
        df_train_bal = self.undersample(df_train_full, ratio=ratio)

        X_train = df_train_bal.drop(columns=["Is_laundering"])
        y_train = df_train_bal["Is_laundering"]

        # Мини-тест из сбалансированных (внутренний контроль)
        X_test_bal_small, _, y_test_bal_small, _ = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state
        )

        # Реальный несбалансированный тест
        X_test_real = df_test_real.drop(columns=["Is_laundering"])
        y_test_real = df_test_real["Is_laundering"]

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test_bal": X_test_bal_small,
            "y_test_bal": y_test_bal_small,
            "X_test_real": X_test_real,
            "y_test_real": y_test_real
        }

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, y_pred)

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "roc_curve": roc_curve(y_test, y_proba),
            "pr_curve": precision_recall_curve(y_test, y_proba),
            "confusion_matrix": cm
        }

    def save_model(self, path="lightgbm_model.pkl"):
        joblib.dump(self.model, path)

    def load_model(self, path="lightgbm_model.pkl"):
        self.model = joblib.load(path)
