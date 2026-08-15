from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
NOTES_DIR = DATA_DIR / "notes"
DB_PATH = DATA_DIR / "database.db"

VECTOR_DIR = BASE_DIR / "vector_store"
MODEL_DIR = BASE_DIR / "models"

LLM_MODEL = "qwen3:8b"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5

DATA_DIR.mkdir(exist_ok=True)
NOTES_DIR.mkdir(exist_ok=True)
VECTOR_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

SUBJECTS = [
    "dbms",
    "operating_systems",
    "computer_networks",
    "machine_learning"
]