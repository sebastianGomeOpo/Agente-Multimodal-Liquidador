"""
test_multimodal_indexer.py - Tests del indexador multimodal
"""

import pytest
from unittest.mock import patch, MagicMock

from src.vectorstore.multimodal_indexer import MultimodalIndexer
from src.data_models import EmbeddingRecord, IndexingResult


class TestMultimodalIndexer:
    """Tests para MultimodalIndexer"""
    
    @patch('src.vectorstore.multimodal_indexer.get_chroma_manager')
    @patch('src.vectorstore.multimodal_indexer.CLIPEncoder')
    def test_generate_stable_id_excel_image(self, mock_encoder, mock_chroma):
        """Test: Generar ID estable para imagen de Excel"""
        indexer = MultimodalIndexer()
        
        metadata = {
            "type": "excel_image",
            "source_file": "test_data.xlsx",
            "chunk": "r0_c0"
        }
        
        doc_id = indexer._generate_stable_id(metadata)
        
        assert doc_id == "excel_test_data_r0_c0"
    
    @patch('src.vectorstore.multimodal_indexer.get_chroma_manager')
    @patch('src.vectorstore.multimodal_indexer.CLIPEncoder')
    def test_generate_stable_id_pdf(self, mock_encoder, mock_chroma):
        """Test: Generar ID estable para PDF"""
        indexer = MultimodalIndexer()
        
        metadata = {
            "type": "pdf",
            "source_file": "document.pdf",
            "page": 3
        }
        
        doc_id = indexer._generate_stable_id(metadata)
        
        assert doc_id == "pdf_document_p3"
    
    @patch('src.vectorstore.multimodal_indexer.get_chroma_manager')
    @patch('src.vectorstore.multimodal_indexer.CLIPEncoder')
    def test_generate_stable_id_text(self, mock_encoder, mock_chroma):
        """Test: Generar ID estable para texto"""
        indexer = MultimodalIndexer()
        
        metadata = {
            "type": "text",
            "source_file": "test_document"
        }
        
        doc_id = indexer._generate_stable_id(metadata)
        
        assert doc_id == "text_test_document"
    
    @patch('src.vectorstore.multimodal_indexer.get_chroma_manager')
    @patch('src.vectorstore.multimodal_indexer.CLIPEncoder')
    def test_index_batch_validates_dimensions(self, mock_encoder, mock_chroma, sample_embedding_record):
        """Test: index_batch valida dimensiones de embeddings"""
        mock_manager = MagicMock()
        mock_collection = MagicMock()
        mock_manager.get_collection.return_value = mock_collection
        mock_chroma.return_value = mock_manager
        
        indexer = MultimodalIndexer()
        
        # Record válido (512 dimensiones)
        valid_record = sample_embedding_record
        
        # Record inválido (3 dimensiones)
        invalid_record = EmbeddingRecord(
            embedding=[0.1, 0.2, 0.3],
            document="Test",
            metadata={"type": "text", "source_file": "test"}
        )
        
        result = indexer.index_batch([valid_record, invalid_record])
        
        # Solo el válido debe indexarse
        assert result.indexed == 1
        assert len(result.errors) == 1
    
    @patch('src.vectorstore.multimodal_indexer.get_chroma_manager')
    @patch('src.vectorstore.multimodal_indexer.CLIPEncoder')
    def test_index_batch_success(self, mock_encoder, mock_chroma, sample_embedding_record):
        """Test: index_batch indexa correctamente registros válidos"""
        mock_manager = MagicMock()
        mock_collection = MagicMock()
        mock_manager.get_collection.return_value = mock_collection
        mock_chroma.return_value = mock_manager
        
        indexer = MultimodalIndexer()
        
        records = [sample_embedding_record, sample_embedding_record]
        result = indexer.index_batch(records)
        
        assert result.status == "success"
        assert result.indexed == 2
        assert len(result.ids) == 2
        
        # Verificar que se llamó a collection.add
        mock_collection.add.assert_called_once()
    
    @patch('src.vectorstore.multimodal_indexer.get_chroma_manager')
    @patch('src.vectorstore.multimodal_indexer.CLIPEncoder')
    def test_semantic_search(self, mock_encoder, mock_chroma, mock_chroma_collection, sample_embedding):
        """Test: semantic_search realiza búsqueda correctamente"""
        mock_manager = MagicMock()
        mock_manager.get_collection.return_value = mock_chroma_collection
        mock_chroma.return_value = mock_manager
        
        mock_encoder_instance = MagicMock()
        mock_encoder_instance.encode_text.return_value = sample_embedding
        mock_encoder.return_value = mock_encoder_instance
        
        indexer = MultimodalIndexer()
        results = indexer.semantic_search("test query", n_results=5)
        
        assert len(results) == 2
        assert results[0].id == "text_test_document"
        assert results[0].similarity > 0
    
    @patch('src.vectorstore.multimodal_indexer.get_chroma_manager')
    @patch('src.vectorstore.multimodal_indexer.CLIPEncoder')
    def test_search_filters_by_similarity(self, mock_encoder, mock_chroma, mock_chroma_collection, sample_embedding):
        """Test: Filtrar resultados por similitud mínima"""
        mock_manager = MagicMock()
        mock_manager.get_collection.return_value = mock_chroma_collection
        mock_chroma.return_value = mock_manager
        
        mock_encoder_instance = MagicMock()
        mock_encoder_instance.encode_text.return_value = sample_embedding
        mock_encoder.return_value = mock_encoder_instance
        
        indexer = MultimodalIndexer()
        
        # Buscar con umbral alto (0.9)
        results = indexer.semantic_search("test query", min_similarity=0.9)
        
        # Los resultados mock tienen similitudes de 0.8 y 0.7, ambos bajo 0.9
        assert len(results) == 0