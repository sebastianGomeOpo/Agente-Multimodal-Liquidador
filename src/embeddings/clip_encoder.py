"""
clip_encoder.py - Generador de embeddings multimodales con CLIP
"""

import numpy as np
from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import Optional

from src.utils.logger import get_logger
from src.utils.config import CLIP_MODEL_NAME, EMBEDDING_DIMENSION

logger = get_logger(__name__)


class CLIPEncoder:
    """
    Encoder singleton para generar embeddings multimodales con CLIP.
    Se asegura de cargar el modelo una sola vez.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_name: str = CLIP_MODEL_NAME):
        """Inicializa CLIP encoder (solo una vez)"""
        if hasattr(self, '_initialized'):
            return
            
        try:
            logger.info(f"Cargando modelo CLIP: {model_name}")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Usando device: {self.device}")
            
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            
            self.model.to(self.device)
            self.model.eval()
            
            self._initialized = True
            logger.info("✓ CLIP encoder cargado exitosamente")
        
        except Exception as e:
            logger.error(f"Error al cargar CLIP: {e}")
            raise
    
    def encode_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Convierte imagen a embedding CLIP normalizado
        
        Args:
            image_path: Ruta a la imagen
        
        Returns:
            np.ndarray: Embedding de 512 dimensiones (normalizado) o None si falla
        """
        try:
            logger.debug(f"Codificando imagen: {Path(image_path).name}")
            
            image = Image.open(image_path).convert("RGB")
            
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            
            # Normalizar embeddings (CRÍTICO para similitud coseno)
            embedding = image_features.cpu().numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)
            
            # Validar dimensión
            if embedding.shape[0] != EMBEDDING_DIMENSION:
                logger.error(f"Dimensión incorrecta: {embedding.shape[0]} != {EMBEDDING_DIMENSION}")
                return None
            
            logger.debug(f"✓ Imagen codificada: {embedding.shape}")
            return embedding
        
        except Exception as e:
            logger.error(f"Error al codificar imagen {image_path}: {e}")
            return None
    
    def encode_text(self, text: str) -> Optional[np.ndarray]:
        """
        Convierte texto a embedding CLIP normalizado
        
        Args:
            text: Texto a codificar
        
        Returns:
            np.ndarray: Embedding de 512 dimensiones (normalizado) o None si falla
        """
        try:
            logger.debug(f"Codificando texto: {text[:50]}...")
            
            inputs = self.processor(text=text, return_tensors="pt", truncation=True, max_length=77)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            
            # Normalizar embeddings
            embedding = text_features.cpu().numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)
            
            # Validar dimensión
            if embedding.shape[0] != EMBEDDING_DIMENSION:
                logger.error(f"Dimensión incorrecta: {embedding.shape[0]} != {EMBEDDING_DIMENSION}")
                return None
            
            logger.debug(f"✓ Texto codificado: {embedding.shape}")
            return embedding
        
        except Exception as e:
            logger.error(f"Error al codificar texto: {e}")
            return None
    
    def batch_encode_images(self, image_paths: list) -> list:
        """
        Codifica múltiples imágenes
        
        Args:
            image_paths: Lista de rutas a imágenes
        
        Returns:
            list: Lista de embeddings (solo los exitosos)
        """
        embeddings = []
        
        logger.info(f"Codificando {len(image_paths)} imágenes...")
        
        for i, image_path in enumerate(image_paths):
            embedding = self.encode_image(image_path)
            if embedding is not None:
                embeddings.append(embedding)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Procesadas {i + 1}/{len(image_paths)} imágenes")
        
        logger.info(f"✓ {len(embeddings)}/{len(image_paths)} imágenes codificadas exitosamente")
        return embeddings
    
    def batch_encode_texts(self, texts: list) -> list:
        """
        Codifica múltiples textos
        
        Args:
            texts: Lista de textos
        
        Returns:
            list: Lista de embeddings (solo los exitosos)
        """
        embeddings = []
        
        logger.info(f"Codificando {len(texts)} textos...")
        
        for i, text in enumerate(texts):
            embedding = self.encode_text(text)
            if embedding is not None:
                embeddings.append(embedding)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Procesados {i + 1}/{len(texts)} textos")
        
        logger.info(f"✓ {len(embeddings)}/{len(texts)} textos codificados exitosamente")
        return embeddings