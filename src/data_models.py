"""
data_models.py - Modelos de datos compartidos para todo el pipeline
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class EmbeddingRecord(BaseModel):
    """
    Modelo unificado para un registro de embedding (texto o imagen).
    Este es el formato que consume MultimodalIndexer.index_batch().
    """
    embedding: List[float] = Field(description="Vector de embedding (512 dimensiones para CLIP)")
    document: str = Field(description="Ruta de la imagen O texto descriptivo")
    metadata: Dict[str, Any] = Field(description="Metadatos estructurados")
    
    class Config:
        json_schema_extra = {
            "example": {
                "embedding": [0.1, 0.2, 0.3],
                "document": "data/images/excel_images/file_chunk_r0_c0.png",
                "metadata": {
                    "type": "excel_image",
                    "source_file": "file.xlsx",
                    "chunk": "r0_c0"
                }
            }
        }


class DocumentType(str):
    """Tipos de documentos válidos en el sistema"""
    EXCEL_IMAGE = "excel_image"
    PDF = "pdf"
    TEXT = "text"


class IndexingResult(BaseModel):
    """Resultado de un proceso de indexación"""
    status: Literal["success", "error", "warning"]
    indexed: int = 0
    errors: List[str] = Field(default_factory=list)
    ids: List[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    """Resultado de una búsqueda semántica"""
    id: str
    distance: float
    similarity: float  # 1 - distance
    metadata: Dict[str, Any]
    document: str


class AgentResponse(BaseModel):
    """Respuesta final del agente"""
    answer: str = Field(description="Respuesta en lenguaje natural")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Fuentes citadas")
    quality_score: float = Field(ge=0.0, le=1.0, description="Confianza de la respuesta")
    reasoning_steps: List[str] = Field(default_factory=list, description="Pasos de razonamiento (CoT)")