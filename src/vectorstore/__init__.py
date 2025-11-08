"""
vectorstore/__init__.py

Exporta las clases y funciones principales del módulo de vectorstore.
"""

# Exporta la clase principal del gestor de ChromaDB y su factory
from .chroma_manager import ChromaManager, get_chroma_manager

# Exporta la clase principal del indexador multimodal
from .multimodal_indexer import (
    MultimodalIndexer, 
    index_all_embeddings  # <- La función principal del pipeline (Paso 5)
)