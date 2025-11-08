"""
test_config.py - Tests de configuración
"""

import pytest
from pathlib import Path
import os

from src.utils import config


class TestConfigPaths:
    """Tests de rutas de configuración"""
    
    def test_base_directories_exist(self):
        """Test: Directorios base se crean correctamente"""
        assert config.BASE_DIR.exists()
        assert config.DATA_DIR.exists()
        assert config.SRC_DIR.exists()
    
    def test_input_directories_exist(self):
        """Test: Directorios de entrada se crean"""
        assert config.INPUT_EXCEL_DIR.exists()
        assert config.INPUT_PDF_DIR.exists()
    
    def test_output_directories_exist(self):
        """Test: Directorios de salida se crean"""
        assert config.EMBEDDINGS_DIR.exists()
        assert config.EXTRACTED_TEXT_DIR.exists()
        assert config.EXTRACTED_TABLES_DIR.exists()
    
    def test_chroma_directory_exists(self):
        """Test: Directorio de ChromaDB se crea"""
        assert config.CHROMA_PERSIST_DIR.exists()


class TestConfigValues:
    """Tests de valores de configuración"""
    
    def test_clip_model_name(self):
        """Test: Nombre del modelo CLIP"""
        assert config.CLIP_MODEL_NAME == "openai/clip-vit-base-patch32"
    
    def test_embedding_dimension(self):
        """Test: Dimensión de embeddings"""
        assert config.EMBEDDING_DIMENSION == 512
    
    def test_retrieve_top_k(self):
        """Test: Número de documentos a recuperar"""
        assert config.RETRIEVE_TOP_K == 5
    
    def test_excel_chunk_size(self):
        """Test: Tamaño de chunks de Excel"""
        assert config.EXCEL_CHUNK_ROWS == 50
        assert config.EXCEL_CHUNK_COLS == 50
    
    def test_llm_temperature(self):
        """Test: Temperatura del LLM"""
        assert config.LLM_TEMPERATURE == 0.2
        assert 0 <= config.LLM_TEMPERATURE <= 1


class TestAPIKeys:
    """Tests de validación de API keys"""
    
    def test_api_keys_loaded(self):
        """Test: API keys se cargan desde .env"""
        # Nota: Este test asume que el .env existe
        # En un entorno de CI/CD, se usarían secrets
        
        # No comparamos valores específicos por seguridad
        assert config.OPENAI_API_KEY is not None or config.OPENAI_API_KEY == ""
        assert config.LANDING_AI_API_KEY is not None or config.LANDING_AI_API_KEY == ""
    
    def test_ade_api_url(self):
        """Test: URL del API ADE"""
        assert config.ADE_API_URL == "https://api.va.landing.ai/v1/ade/parse"