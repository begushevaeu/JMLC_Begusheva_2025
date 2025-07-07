import os
import pandas as pd

def verify():
    """
    Загружает датасет с предсказаниями и выводит статистику
    по колонкам is_laundering, model_prediction и alert_probability.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'Datasets', 'test_sample.csv')

    print(f"Загрузка данных из {dataset_path}...")
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Ошибка: Файл {dataset_path} не найден.")
        print("Пожалуйста, сначала запустите generate_predictions.py")
        return

    # Проверяем наличие необходимых колонок
    required_cols = ['is_laundering', 'model_prediction', 'alert_probability']
    if not all(col in df.columns for col in required_cols):
        print(f"Ошибка: В файле отсутствуют необходимые колонки. {required_cols}")
        print("Пожалуйста, сначала запустите generate_predictions.py для их создания.")
        return

    print("\n--- Статистика по предсказаниям ---")

    # Подсчет '1' в is_laundering
    laundering_count = df['is_laundering'].sum()
    print(f"Количество реальных фрод-транзакций ('is_laundering' = 1): {laundering_count}")

    # Подсчет '1' в model_prediction
    prediction_count = df['model_prediction'].sum()
    print(f"Количество предсказанных фрод-транзакций ('model_prediction' = 1): {prediction_count}")

    # Подсчет уникальных значений в alert_probability
    unique_probs = df['alert_probability'].nunique()
    print(f"Количество уникальных значений вероятности ('alert_probability'): {unique_probs}")
    
    print("\n--- Анализ завершен. ---")

if __name__ == "__main__":
    verify()
