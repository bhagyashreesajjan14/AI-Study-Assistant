from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
NOTES_DIR = DATA_DIR / "notes"
DB_PATH = DATA_DIR / "database.db"

VECTOR_DIR = BASE_DIR / "vector_store"
MODEL_DIR = BASE_DIR / "models"

# Model Configurations
LLM_MODEL = "llama3:8b"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5

DATA_DIR.mkdir(exist_ok=True)
NOTES_DIR.mkdir(exist_ok=True)
VECTOR_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


def get_user_notes_dir(user_id: int, subject: str = None) -> Path:
    """Returns the notes directory path isolated for a specific user and optional subject."""
    path = NOTES_DIR / f"user_{user_id}"
    if subject:
        path = path / subject
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_user_vector_dir(user_id: int) -> Path:
    """Returns the vector store directory path isolated for a specific user."""
    path = VECTOR_DIR / f"user_{user_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


DEFAULT_SUBJECTS = [
    "dbms",
    "operating_systems",
    "computer_networks",
    "machine_learning"
]

SUBJECTS = DEFAULT_SUBJECTS