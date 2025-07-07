import pandas as pd
import pickle
import os

# Определяем пути к файлам
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'Datasets', 'df_baseline.csv')
TEST_IDX_PATH = os.path.join(BASE_DIR, 'shamlock_model', 'indices', 'test_idx.pkl')
OUTPUT_PATH = os.path.join(BASE_DIR, 'Datasets', 'test_sample.csv')

def create_test_sample_dataset():
    """
    Загружает основной датасет и индексы тестовой выборки,
    фильтрует данные и сохраняет их в новый CSV-файл.
    """
    print("--- Начало создания датасета для тестовой выборки ---")

    # 1. Загрузка основного датасета
    try:
        print(f"Загрузка основного датасета из: {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH)
        print(f"Основной датасет успешно загружен. Количество строк: {len(df)}")
    except FileNotFoundError:
        print(f"ОШИБКА: Файл не найден: {DATASET_PATH}")
        return

    # 2. Загрузка индексов тестовой выборки
    try:
        print(f"Загрузка индексов из: {TEST_IDX_PATH}")
        with open(TEST_IDX_PATH, 'rb') as f:
            test_indices = pickle.load(f)
        print(f"Индексы тестовой выборки успешно загружены. Количество индексов: {len(test_indices)}")
    except FileNotFoundError:
        print(f"ОШИБКА: Файл с индексами не найден: {TEST_IDX_PATH}")
        return

    # 3. Фильтрация датафрейма
    print("Фильтрация данных по тестовым индексам...")
    test_df = df.iloc[test_indices]
    print(f"Фильтрация завершена. Количество строк в тестовой выборке: {len(test_df)}")

    # 4. Сохранение нового датасета
    try:
        print(f"Сохранение тестовой выборки в: {OUTPUT_PATH}")
        test_df.to_csv(OUTPUT_PATH, index=False)
        print("Новый датасет test_sample.csv успешно создан!")
    except Exception as e:
        print(f"ОШИБКА: Не удалось сохранить файл. {e}")

    print("--- Процесс завершен ---")

if __name__ == "__main__":
    create_test_sample_dataset()
