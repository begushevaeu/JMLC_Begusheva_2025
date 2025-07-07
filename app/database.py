import os
import json
from neo4j import GraphDatabase
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# --- Ручная загрузка .env для надежности ---
BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / '.env'

if env_path.exists():
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"\''))

# --- Настройка подключения к БД ---
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL не найден. Проверьте, что он есть в файле .env")

engine = create_engine(DATABASE_URL)

# --- Neo4j Connection ---
NEO4J_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')

def get_neo4j_driver():
    """Создает и возвращает драйвер для подключения к Neo4j."""
    try:
        with open(NEO4J_CONFIG_PATH) as f:
            config = json.load(f)
        
        uri = config.get("NEO4J_URI")
        user = config.get("NEO4J_USER")
        password = config.get("NEO4J_PASSWORD")
        
        # Исправление для Docker: заменяем neo4j:// на bolt:// для прямого подключения
        if uri and uri.startswith("neo4j://"):
            uri = uri.replace("neo4j://", "bolt://", 1)
            print(f"INFO: Auto-corrected Neo4j URI to: {uri}")
        
        if not all([uri, user, password]):
            raise ValueError("NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set in config.json")

        return GraphDatabase.driver(uri, auth=(user, password))
    except FileNotFoundError:
        print(f"Error: Neo4j config file not found at {NEO4J_CONFIG_PATH}")
        return None
    except Exception as e:
        print(f"Error creating Neo4j driver: {e}")
        return None
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()