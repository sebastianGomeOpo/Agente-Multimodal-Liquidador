"""
config.py - Configuración centralizada del proyecto MultiDoc-Agent
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============= RUTAS BASE =============
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"

# ============= RUTAS DE DATOS =============
INPUT_DIR = DATA_DIR / "input"
INPUT_EXCEL_DIR = INPUT_DIR / "excel"
INPUT_PDF_DIR = INPUT_DIR / "pdf"

IMAGES_DIR = DATA_DIR / "images"
EXCEL_IMAGES_DIR = IMAGES_DIR / "excel_images"

PROCESSED_DIR = DATA_DIR / "processed"
EXTRACTED_TEXT_DIR = PROCESSED_DIR / "extracted_text"
EXTRACTED_TABLES_DIR = PROCESSED_DIR / "extracted_tables"

EMBEDDINGS_DIR = DATA_DIR / "embeddings"
OUTPUT_DIR = BASE_DIR / "outputs"
RESPONSES_DIR = OUTPUT_DIR / "responses"
REPORTS_DIR = OUTPUT_DIR / "reports"

CHROMA_PERSIST_DIR = BASE_DIR / "chromadb_storage"
LOG_DIR = BASE_DIR / "logs"

# ============= CREAR DIRECTORIOS =============
for directory in [
    INPUT_EXCEL_DIR, INPUT_PDF_DIR, EXCEL_IMAGES_DIR,
    EXTRACTED_TEXT_DIR, EXTRACTED_TABLES_DIR, EMBEDDINGS_DIR,
    RESPONSES_DIR, REPORTS_DIR, CHROMA_PERSIST_DIR, LOG_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# ============= CONFIGURACIÓN DE MODELOS =============
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai, anthropic, deepseek

# ============= CONFIGURACIÓN DE API KEYS =============
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LANDING_AI_API_KEY = os.getenv("LANDING_AI_API_KEY")

# Validación de API keys críticas
if not OPENAI_API_KEY:
    print("⚠️  WARNING: OPENAI_API_KEY no configurada en .env")
if not LANDING_AI_API_KEY:
    print("⚠️  WARNING: LANDING_AI_API_KEY no configurada en .env")

# ============= CONFIGURACIÓN DE CHROMADB =============
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "multidoc_collection")
CHROMA_DISTANCE_METRIC = os.getenv("CHROMA_DISTANCE_METRIC", "cosine")

# ============= CONFIGURACIÓN DE EMBEDDINGS =============
EMBEDDING_DIMENSION = 512  # CLIP base-32
EMBEDDING_BATCH_SIZE = 32

# ============= CONFIGURACIÓN DE OCR =============
OCR_PROVIDER = os.getenv("OCR_PROVIDER", "landing_ai")
ADE_API_URL = "https://api.va.landing.ai/v1/ade/parse"
ADE_MODEL = "dpt-2-latest"
ADE_TIMEOUT_PDF = 120  # segundos
ADE_TIMEOUT_IMAGE = 60  # segundos

# ============= CONFIGURACIÓN DE EXCEL =============
EXCEL_CHUNK_ROWS = 50
EXCEL_CHUNK_COLS = 50
EXCEL_DPI = 200

# ============= CONFIGURACIÓN DE LOGGING =============
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOG_DIR / "multidoc_agent.log"

# ============= CONFIGURACIÓN DE LLM =============
LLM_TEMPERATURE = 0.2  # Baja para respuestas determinísticas
LLM_MAX_TOKENS = 2048
LLM_TOP_P = 0.9

# ============= CONFIGURACIÓN DE AGENTE =============
AGENT_TIMEOUT = 60  # segundos
RETRIEVE_TOP_K = 5  # Documentos a recuperar
SIMILARITY_THRESHOLD = 0.5  # Umbral mínimo de similitud coseno

# ============= CONFIGURACIÓN DE ReAct =============
REACT_MAX_ITERATIONS = 3  # Máximo de iteraciones pensamiento-acción
REACT_ENABLE_COT = True  # Activar Chain of Thought