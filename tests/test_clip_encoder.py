"""
test_clip_encoder.py - Tests del encoder CLIP
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.embeddings.clip_encoder import CLIPEncoder
from src.utils.config import EMBEDDING_DIMENSION


class TestCLIPEncoder:
    """Tests para CLIPEncoder"""
    
    @patch('src.embeddings.clip_encoder.CLIPModel')
    @patch('src.embeddings.clip_encoder.CLIPProcessor')
    def test_singleton_pattern(self, mock_processor, mock_model):
        """Test: CLIPEncoder implementa patrón singleton"""
        encoder1 = CLIPEncoder()
        encoder2 = CLIPEncoder()
        
        assert encoder1 is encoder2
    
    @patch('src.embeddings.clip_encoder.CLIPModel')
    @patch('src.embeddings.clip_encoder.CLIPProcessor')
    def test_encode_text_returns_correct_dimension(self, mock_processor, mock_model, mock_clip_model):
        """Test: encode_text devuelve embedding de 512 dimensiones"""
        mock_model_instance, mock_proc_instance = mock_clip_model
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_processor.from_pretrained.return_value = mock_proc_instance
        
        encoder = CLIPEncoder()
        embedding = encoder.encode_text("Test text")
        
        assert embedding is not None
        assert embedding.shape == (EMBEDDING_DIMENSION,)
    
    @patch('src.embeddings.clip_encoder.CLIPModel')
    @patch('src.embeddings.clip_encoder.CLIPProcessor')
    def test_encode_text_normalizes_embedding(self, mock_processor, mock_model, mock_clip_model):
        """Test: encode_text normaliza el embedding"""
        mock_model_instance, mock_proc_instance = mock_clip_model
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_processor.from_pretrained.return_value = mock_proc_instance
        
        encoder = CLIPEncoder()
        embedding = encoder.encode_text("Test text")
        
        # Verificar que está normalizado (norma ≈ 1.0)
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01
    
    @patch('src.embeddings.clip_encoder.CLIPModel')
    @patch('src.embeddings.clip_encoder.CLIPProcessor')
    @patch('src.embeddings.clip_encoder.Image')
    def test_encode_image_handles_file_not_found(self, mock_image, mock_processor, mock_model):
        """Test: encode_image maneja correctamente archivo no encontrado"""
        mock_image.open.side_effect = FileNotFoundError
        
        encoder = CLIPEncoder()
        embedding = encoder.encode_image("nonexistent.png")
        
        assert embedding is None
    
    @patch('src.embeddings.clip_encoder.CLIPModel')
    @patch('src.embeddings.clip_encoder.CLIPProcessor')
    def test_batch_encode_texts(self, mock_processor, mock_model, mock_clip_model):
        """Test: batch_encode_texts procesa múltiples textos"""
        mock_model_instance, mock_proc_instance = mock_clip_model
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_processor.from_pretrained.return_value = mock_proc_instance
        
        encoder = CLIPEncoder()
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = encoder.batch_encode_texts(texts)
        
        assert len(embeddings) == 3
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)