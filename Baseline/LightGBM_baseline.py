import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import lightgbm as lgb

def preprocess_and_train(df, target_col="Is_laundering", test_size=0.2, random_state=42):
    # Разделение на мажоритарный и миноритарный классы
    df_major = df[df[target_col] == 0]
    df_minor = df[df[target_col] == 1]

    # Уменьшение размера мажоритарного класса до размера миноритарного
    df_major_downsampled = resample(
        df_major,
        replace=False,
        n_samples=len(df_minor),
        random_state=random_state
    )

    # Объединение уменьшенного мажоритарного класса с миноритарным классом и перемешивание
    df_balanced = pd.concat([df_major_downsampled, df_minor])
    df_balanced = df_balanced.sample(frac=1, random_state=random_state)

    # Разделение на признаки и целевую переменную
    X = df_balanced.drop(columns=[target_col, "Sender_account", "Receiver_account"])
    y = df_balanced[target_col]

    # Кодирование категориальных признаков
    X = pd.get_dummies(X)

    # Деление на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Обучение модели LightGBM
    model = lgb.LGBMClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    # Оценка модели
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, report