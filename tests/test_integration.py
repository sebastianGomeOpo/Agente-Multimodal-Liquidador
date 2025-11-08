"""
test_integration.py - Tests de integración end-to-end
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.embeddings.embedding_pipeline import process_all_multimodal
from src.vectorstore.multimodal_indexer import index_all_embeddings
from src.agent import run_agent_query


class TestIntegrationPipeline:
    """Tests de integración del pipeline completo"""
    
    @pytest.mark.integration
    @patch('src.embeddings.clip_encoder.CLIPModel')
    @patch('src.embeddings.clip_encoder.CLIPProcessor')
    def test_embedding_to_indexing_integration(
        self,
        mock_processor,
        mock_model,
        temp_test_dir,
        sample_structured_json,
        sample_embedding
    ):
        """Test: Integración del Paso 5 → Paso 6"""
        # Setup: Crear archivos de entrada
        json_file = temp_test_dir / "test_structure.json"
        with open(json_file, 'w') as f:
            json.dump(sample_structured_json, f)
        
        # Mock del encoder
        mock_model_instance = MagicMock()
        mock_proc_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_processor.from_pretrained.return_value = mock_proc_instance
        
        mock_features = MagicMock()
        mock_features.cpu.return_value.numpy.return_value = sample_embedding.reshape(1, -1)
        mock_model_instance.get_text_features.return_value = mock_features
        
        # Ejecutar Paso 5
        with patch('src.embeddings.embedding_pipeline.EXTRACTED_TABLES_DIR', temp_test_dir):
            with patch('src.embeddings.embedding_pipeline.EXCEL_IMAGES_DIR', temp_test_dir):
                with patch('src.embeddings.embedding_pipeline.EMBEDDINGS_DIR', temp_test_dir):
                    embedding_result = process_all_multimodal()
        
        assert embedding_result["text_count"] > 0
        
        # Verificar que se crearon los archivos de embeddings
        text_embeddings_file = temp_test_dir / "text_embeddings.json"
        assert text_embeddings_file.exists()
        
        # Verificar formato de los embeddings
        with open(text_embeddings_file, 'r') as f:
            embeddings_data = json.load(f)
        
        assert len(embeddings_data) > 0
        assert all("embedding" in record for record in embeddings_data)
        assert all("document" in record for record in embeddings_data)
        assert all("metadata" in record for record in embeddings_data)
    
    @pytest.mark.integration
    @patch('src.vectorstore.multimodal_indexer.get_chroma_manager')
    @patch('src.vectorstore.multimodal_indexer.CLIPEncoder')
    @patch('src.agent.nodes.ChatOpenAI')
    def test_full_agent_query_flow(
        self,
        mock_llm_class,
        mock_encoder_class,
        mock_chroma,
        mock_chroma_collection,
        mock_llm_response,
        sample_embedding
    ):
        """Test: Flujo completo de consulta del agente"""
        # Setup mocks
        mock_manager = MagicMock()
        mock_manager.get_collection.return_value = mock_chroma_collection
        mock_chroma.return_value = mock_manager
        
        mock_encoder = MagicMock()
        mock_encoder.encode_text.return_value = sample_embedding
        mock_encoder_class.return_value = mock_encoder
        
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_llm_response
        mock_llm_class.return_value = mock_llm
        
        # Ejecutar query
        result = run_agent_query("¿Cuánto tonelaje tiene la bodega B1?")
        
        # Verificaciones
        assert result["status"] == "success"
        assert "response" in result
        
        response = result["response"]
        assert "answer" in response
        assert "sources" in response
        assert "quality_score" in response
        assert "reasoning_steps" in response
        
        # Verificar que el LLM fue llamado
        mock_llm.invoke.assert_called_once()
        
        # Verificar que se hizo búsqueda semántica
        mock_chroma_collection.query.assert_called()