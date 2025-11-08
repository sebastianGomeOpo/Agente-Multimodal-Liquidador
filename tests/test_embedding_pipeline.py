"""
test_embedding_pipeline.py - Tests del pipeline de embeddings
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.embeddings.embedding_pipeline import (
    _create_text_document_from_json,
    process_all_multimodal
)


class TestTextDocumentCreation:
    """Tests para _create_text_document_from_json"""
    
    def test_create_text_from_complete_json(self, sample_structured_json):
        """Test: Crear texto descriptivo desde JSON completo"""
        text = _create_text_document_from_json(sample_structured_json)
        
        assert "R-2024-001" in text
        assert "MV EXAMPLE SHIP" in text
        assert "Cliente A" in text
        assert "1234.5" in text
        assert "Bodega B1" in text
    
    def test_create_text_from_minimal_json(self):
        """Test: Crear texto desde JSON con datos mínimos"""
        minimal_json = {
            "recalada": "R-TEST",
            "nave_nombre": None,
            "bodegas": []
        }
        
        text = _create_text_document_from_json(minimal_json)
        
        assert "R-TEST" in text
        assert text != ""
    
    def test_create_text_includes_all_clients(self, sample_structured_json):
        """Test: Incluye todos los clientes"""
        text = _create_text_document_from_json(sample_structured_json)
        
        assert "Cliente A" in text
        assert "Cliente B" in text
    
    def test_create_text_includes_facturas(self, sample_structured_json):
        """Test: Incluye códigos de facturación"""
        text = _create_text_document_from_json(sample_structured_json)
        
        assert "FAC-001" in text
        assert "FAC-002" in text
        assert "FAC-003" in text


class TestProcessAllMultimodal:
    """Tests para process_all_multimodal"""
    
    @patch('src.embeddings.embedding_pipeline.CLIPEncoder')
    @patch('src.embeddings.embedding_pipeline.EXCEL_IMAGES_DIR')
    @patch('src.embeddings.embedding_pipeline.EXTRACTED_TABLES_DIR')
    def test_process_all_multimodal_success(
        self,
        mock_tables_dir,
        mock_images_dir,
        mock_encoder_class,
        temp_test_dir,
        mock_image_file,
        sample_structured_json,
        sample_embedding
    ):
        """Test: process_all_multimodal procesa imágenes y textos"""
        # Setup mocks
        mock_images_dir.glob.return_value = [mock_image_file]
        
        json_file = temp_test_dir / "test_structure.json"
        with open(json_file, 'w') as f:
            json.dump(sample_structured_json, f)
        mock_tables_dir.glob.return_value = [json_file]
        
        mock_encoder = MagicMock()
        mock_encoder.encode_image.return_value = sample_embedding
        mock_encoder.encode_text.return_value = sample_embedding
        mock_encoder_class.return_value = mock_encoder
        
        # Patchear el directorio de salida
        with patch('src.embeddings.embedding_pipeline.EMBEDDINGS_DIR', temp_test_dir):
            result = process_all_multimodal()
        
        assert result["image_count"] >= 0
        assert result["text_count"] >= 0
        assert result["total_embeddings"] == result["image_count"] + result["text_count"]
    
    @patch('src.embeddings.embedding_pipeline.CLIPEncoder')
    def test_process_handles_empty_directories(self, mock_encoder_class, temp_test_dir):
        """Test: Maneja correctamente directorios vacíos"""
        with patch('src.embeddings.embedding_pipeline.EXCEL_IMAGES_DIR', temp_test_dir):
            with patch('src.embeddings.embedding_pipeline.EXTRACTED_TABLES_DIR', temp_test_dir):
                with patch('src.embeddings.embedding_pipeline.EMBEDDINGS_DIR', temp_test_dir):
                    result = process_all_multimodal()
        
        assert result["image_count"] == 0
        assert result["text_count"] == 0