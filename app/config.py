import os
from pathlib import Path
import json

# Определяем корневую директорию проекта (на один уровень выше папки 'app')
BASE_DIR = Path(__file__).resolve().parent.parent

# Составляем полный путь к файлу config.json
config_path = BASE_DIR / 'config.json'

# --- Загрузка конфигурации из JSON-файла ---
config_data = {}
if config_path.exists():
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
else:
    # В реальном приложении здесь лучше выбрасывать исключение
    print(f"КРИТИЧЕСКАЯ ОШИБКА: Файл конфигурации не найден по пути {config_path}")

# --- Настройки Neo4j ---
NEO4J_URI = config_data.get("NEO4J_URI")
NEO4J_USER = config_data.get("NEO4J_USER")
NEO4J_PASSWORD = config_data.get("NEO4J_PASSWORD")
