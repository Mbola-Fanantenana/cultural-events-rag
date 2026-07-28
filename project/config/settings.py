"""
config centralisée du projet RAG
"""

import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
PROJECT_DIR = BASE_DIR / 'project'

# ===== API KEYS =====
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OPENAGENDA_API_KEY = os.getenv("OPENAGENDA_API_KEY")

# ===== PROXY =====
PROXIES = {
    'http://': 'http://webproxy.oma:8080',
    'https://': 'http://webproxy.oma:8080',
}

PROXIES_REQUESTS = {
    'http': 'http://webproxy.oma:8080',
    'https': 'http://webproxy.oma:8080',
}

# ===== OPEN AGENDA =====
OPENAGENDA_BASE_URL = "https://api.openagenda.com/v2"
OPENAGENDA_NB_AGENDAS = 20
OPENAGENDA_EVENTS_PAR_AGENDA = 30
OPENAGENDA_OFFSET_FILE = "project/data/raw/collecte_offset.json"

# ===== MISTRAL =====
MISTRAL_MODEL = "mistral-small-latest"
MISTRAL_EMBEDDING_MODEL = "mistral-embed"
MISTRAL_MAX_TOKENS = 1000
MISTRAL_TEMPERATURE = 0.7

EMBEDDING_DIMENSION = 1024;
EMBEDDING_BATCH_SIZE = 10;
EMBEDDING_DELAY = 0.5;


# ===== FAISS =====
FAISS_INDEX_TYPE = "IndexFlatL2"  # Type d'index FAISS
FAISS_TOP_K = 5

DATA_RAW_PATH = "project/data/raw"
DATA_PROCESSED_PATH = "project/data/processed"
DATA_EMBEDDINGS_PATH = "project/data/embeddings" 
DATA_VECTORSTORE_PATH = "project/data/vectorstore"
DATA_EXAMPLES_PATH = "project/data/examples"

EMBEDDINGS_PATH = DATA_EMBEDDINGS_PATH