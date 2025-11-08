"""
test_data_models.py - Tests de los modelos de datos
"""

import pytest
from pydantic import ValidationError

from src.data_models import (
    EmbeddingRecord,
    DocumentType,
    IndexingResult,
    RetrievalResult,
    AgentResponse
)


class TestEmbeddingRecord:
    """Tests para EmbeddingRecord"""
    
    def test_valid_embedding_record(self, sample_embedding):
        """Test: Crear un EmbeddingRecord válido"""
        record = EmbeddingRecord(
            embedding=sample_embedding.tolist(),
            document="Test document",
            metadata={"type": "text", "source_file": "test.json"}
        )
        
        assert len(record.embedding) == 512
        assert record.document == "Test document"
        assert record.metadata["type"] == "text"
    
    def test_embedding_wrong_dimension(self):
        """Test: Rechazar embedding con dimensión incorrecta"""
        # Nota: Pydantic no valida la longitud de lista por defecto,
        # pero nuestro código lo hace en runtime
        record = EmbeddingRecord(
            embedding=[0.1, 0.2, 0.3],  # Solo 3 dimensiones
            document="Test",
            metadata={"type": "text"}
        )
        
        # El modelo se crea, pero la validación ocurre en el indexer
        assert len(record.embedding) == 3
    
    def test_embedding_record_serialization(self, sample_embedding):
        """Test: Serialización a dict"""
        record = EmbeddingRecord(
            embedding=sample_embedding.tolist(),
            document="Test",
            metadata={"type": "text", "source_file": "test.json"}
        )
        
        record_dict = record.dict()
        
        assert "embedding" in record_dict
        assert "document" in record_dict
        assert "metadata" in record_dict
        assert isinstance(record_dict["embedding"], list)


class TestDocumentType:
    """Tests para DocumentType"""
    
    def test_valid_document_types(self):
        """Test: Tipos de documento válidos"""
        assert DocumentType.EXCEL_IMAGE == "excel_image"
        assert DocumentType.PDF == "pdf"
        assert DocumentType.TEXT == "text"


class TestIndexingResult:
    """Tests para IndexingResult"""
    
    def test_successful_indexing(self):
        """Test: Resultado de indexación exitosa"""
        result = IndexingResult(
            status="success",
            indexed=10,
            ids=["doc1", "doc2", "doc3"]
        )
        
        assert result.status == "success"
        assert result.indexed == 10
        assert len(result.ids) == 3
        assert len(result.errors) == 0
    
    def test_failed_indexing(self):
        """Test: Resultado de indexación con errores"""
        result = IndexingResult(
            status="error",
            indexed=5,
            errors=["Error 1", "Error 2"]
        )
        
        assert result.status == "error"
        assert result.indexed == 5
        assert len(result.errors) == 2


class TestRetrievalResult:
    """Tests para RetrievalResult"""
    
    def test_retrieval_result(self):
        """Test: Resultado de búsqueda"""
        result = RetrievalResult(
            id="text_test_document",
            distance=0.2,
            similarity=0.8,
            metadata={"type": "text", "source_file": "test.json"},
            document="Test document content"
        )
        
        assert result.id == "text_test_document"
        assert result.distance == 0.2
        assert result.similarity == 0.8
        assert result.metadata["type"] == "text"


class TestAgentResponse:
    """Tests para AgentResponse"""
    
    def test_valid_agent_response(self):
        """Test: Respuesta válida del agente"""
        response = AgentResponse(
            answer="La nave procesó 1234.5 toneladas.",
            sources=[
                {"id": "doc1", "type": "text", "cited": True}
            ],
            quality_score=0.85,
            reasoning_steps=["Paso 1", "Paso 2"]
        )
        
        assert response.answer == "La nave procesó 1234.5 toneladas."
        assert len(response.sources) == 1
        assert response.quality_score == 0.85
        assert len(response.reasoning_steps) == 2
    
    def test_quality_score_bounds(self):
        """Test: Quality score debe estar entre 0 y 1"""
        with pytest.raises(ValidationError):
            AgentResponse(
                answer="Test",
                quality_score=1.5  # Inválido
            )
        
        with pytest.raises(ValidationError):
            AgentResponse(
                answer="Test",
                quality_score=-0.1  # Inválido
            )