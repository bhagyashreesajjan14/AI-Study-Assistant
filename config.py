from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
NOTES_DIR = DATA_DIR / "notes"
DB_PATH = DATA_DIR / "database.db"

VECTOR_DIR = BASE_DIR / "vector_store"
MODEL_DIR = BASE_DIR / "models"

import os

# Model Configurations
LLM_MODEL = "llama3-8b-8192"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_groq_api_key() -> str:
    """Retrieves Groq API key from Streamlit secrets, environment variables, or secrets.toml."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    secrets_file = BASE_DIR / ".streamlit" / "secrets.toml"
    if secrets_file.exists():
        try:
            with open(secrets_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("GROQ_API_KEY"):
                        parts = line.split("=", 1)
                        if len(parts) == 2:
                            return parts[1].strip().strip('"\'')
        except Exception:
            pass

    return os.environ.get("GROQ_API_KEY", "")


GROQ_API_KEY = get_groq_api_key()


CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5

# Upload & File Processing Configurations
SUPPORTED_FILE_EXTENSIONS = ["pdf", "doc", "docx", "txt", "md", "png", "jpg", "jpeg", "webp"]
MAX_UPLOAD_SIZE_MB = 25

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