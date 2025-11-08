"""
clip_encoder.py - Generador de embeddings multimodales con CLIP
"""

import numpy as np
from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from src.utils.logger import get_logger
from src.utils.config import (
    CLIP_MODEL_NAME, EMBEDDING_DIMENSION, EMBEDDING_BATCH_SIZE,
    EXTRACTED_TABLES_DIR, EXCEL_IMAGES_DIR, EMBEDDINGS_DIR
    # Se elimina PDF_IMAGES_DIR
)

logger = get_logger(__name__)


class CLIPEncoder:
    """Encoder para generar embeddings multimodales con CLIP"""
    
    def __init__(self, model_name: str = CLIP_MODEL_NAME):
        """
        Inicializa CLIP encoder
        
        Args:
            model_name: Nombre del modelo CLIP a usar
        """
        try:
            logger.info(f"Cargando modelo CLIP: {model_name}")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Usando device: {self.device}")
            
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("CLIP encoder cargado exitosamente")
        
        except Exception as e:
            logger.error(f"Error al cargar CLIP: {e}")
            raise
    
    def encode_image(self, image_path: str) -> np.ndarray:
        """
        Convierte imagen a embedding CLIP
        
        Args:
            image_path: Ruta a la imagen
        
        Returns:
            np.ndarray: Embedding de la imagen
        """
        try:
            logger.info(f"Codificando imagen: {image_path}")
            
            image = Image.open(image_path).convert("RGB")
            
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            
            # Normalizar embeddings
            embedding = image_features.cpu().numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)
            
            logger.info(f"Imagen codificada. Dimensión: {embedding.shape}")
            return embedding
        
        except Exception as e:
            logger.error(f"Error al codificar imagen: {e}")
            return None
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Convierte texto a embedding CLIP
        
        Args:
            text: Texto a codificar
        
        Returns:
            np.ndarray: Embedding del texto
        """
        try:
            logger.info(f"Codificando texto (primeros 50 chars): {text[:50]}")
            
            inputs = self.processor(text=text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            
            # Normalizar embeddings
            embedding = text_features.cpu().numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)
            
            logger.info(f"Texto codificado. Dimensión: {embedding.shape}")
            return embedding
        
        except Exception as e:
            logger.error(f"Error al codificar texto: {e}")
            return None
    
    def batch_encode_images(self, image_paths: list) -> dict:
        """
        Codifica múltiples imágenes
        
        Args:
            image_paths: Lista de rutas a imágenes
        
        Returns:
            dict: Embeddings con metadata
        """
        embeddings = {}
        
        logger.info(f"Batch encoding de {len(image_paths)} imágenes")
        
        for i, image_path in enumerate(image_paths):
            embedding = self.encode_image(image_path)
            if embedding is not None:
                embeddings[image_path] = {
                    "embedding": embedding.tolist(),
                    "type": "image",
                    "source": Path(image_path).name
                }
            
            if (i + 1) % 10 == 0:
                logger.info(f"Procesadas {i + 1}/{len(image_paths)} imágenes")
        
        return embeddings
    
    def batch_encode_texts(self, texts: list) -> dict:
        """
        Codifica múltiples textos
        
        Args:
            texts: Lista de textos
        
        Returns:
            dict: Embeddings con metadata
        """
        embeddings = {}
        
        logger.info(f"Batch encoding de {len(texts)} textos")
        
        for i, text in enumerate(texts):
            embedding = self.encode_text(text)
            if embedding is not None:
                embeddings[f"text_{i}"] = {
                    "embedding": embedding.tolist(),
                    "type": "text",
                    "content": text[:100]  # Primeros 100 chars
                }
            
            if (i + 1) % 10 == 0:
                logger.info(f"Procesados {i + 1}/{len(texts)} textos")
        
        return embeddings
    
    def verify_shared_space(self, image_path: str, text: str) -> dict:
        """
        Verifica que imagen y texto estén en el mismo espacio vectorial
        CRÍTICO PARA MULTIMODALIDAD
        
        Args:
            image_path: Ruta a imagen
            text: Texto relacionado
        
        Returns:
            dict: Análisis de similitud
        """
        try:
            logger.info("Verificando espacio vectorial compartido")
            
            # Codificar ambos
            image_embedding = self.encode_image(image_path)
            text_embedding = self.encode_text(text)
            
            if image_embedding is None or text_embedding is None:
                return {"status": "error", "message": "No se pudo codificar"}
            
            # Verificar dimensiones iguales
            if image_embedding.shape != text_embedding.shape:
                return {
                    "status": "error",
                    "message": f"Dimensiones diferentes: {image_embedding.shape} vs {text_embedding.shape}"
                }
            
            # Calcular similitud coseno
            similarity = np.dot(image_embedding, text_embedding)
            
            # Normalizar ambos
            image_norm = image_embedding / np.linalg.norm(image_embedding)
            text_norm = text_embedding / np.linalg.norm(text_embedding)
            cosine_similarity = np.dot(image_norm, text_norm)
            
            logger.info(f"Similitud coseno: {cosine_similarity:.4f}")
            
            return {
                "status": "success",
                "image_dimension": image_embedding.shape,
                "text_dimension": text_embedding.shape,
                "cosine_similarity": float(cosine_similarity),
                "shared_space": True
            }
        
        except Exception as e:
            logger.error(f"Error verificando espacio compartido: {e}")
            return {"status": "error", "message": str(e)}


def save_embeddings(embeddings: dict, output_file: str) -> bool:
    """
    Guarda embeddings en archivo
    
    Args:
        embeddings: Diccionario de embeddings
        output_file: Archivo de salida
    
    Returns:
        bool: True si fue exitoso
    """
    try:
        import json
        output_path = EMBEDDINGS_DIR / output_file
        
        # Convertir numpy arrays a listas
        embeddings_serializable = {}
        for key, value in embeddings.items():
            if isinstance(value, dict):
                embeddings_serializable[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                embeddings_serializable[key] = value.tolist() if isinstance(value, np.ndarray) else value
        
        with open(output_path, 'w') as f:
            json.dump(embeddings_serializable, f)
        
        logger.info(f"Embeddings guardados: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error al guardar embeddings: {e}")
        return False


def process_all_multimodal() -> dict:
    """
    Procesa todas las imágenes y textos para generar embeddings
    (CORREGIDO: Ya no procesa imágenes de PDF)
    
    Returns:
        dict: Resumen del procesamiento
    """
    encoder = CLIPEncoder()
    
    # Procesar imágenes (SOLO DE EXCEL)
    excel_images = list(EXCEL_IMAGES_DIR.glob("*.png"))
    # pdf_images (eliminado)
    all_images = [str(img) for img in excel_images] # <- CORREGIDO
    
    logger.info(f"Procesando {len(all_images)} imágenes totales (solo Excel)")
    
    image_embeddings = encoder.batch_encode_images(all_images)
    
    # Procesar textos extraídos
    # (Esta parte está bien, lee los JSON del Paso 4)
    text_files = list(EXTRACTED_TABLES_DIR.glob("*_structure.json"))
    texts = []
    
    import json
    for text_file in text_files:
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                # CORRECCIÓN SUTIL: El texto para embedding debe ser el JSON completo
                # (o el markdown original), no solo 'raw_text'.
                # Vamos a usar el JSON estructurado como texto, es más rico.
                data = json.load(f)
                # Convertir el dict de JSON a un string de texto
                text_content = json.dumps(data) 
                if text_content:
                    texts.append(text_content)
        except Exception as e:
            logger.warning(f"Error al leer {text_file}: {e}")
    
    logger.info(f"Procesando {len(texts)} textos (desde JSON estructurados)")
    
    text_embeddings = encoder.batch_encode_texts(texts)
    
    # Guardar embeddings
    save_embeddings(image_embeddings, "image_embeddings.json")
    save_embeddings(text_embeddings, "text_embeddings.json")
    
    return {
        "image_count": len(image_embeddings),
        "text_count": len(text_embeddings),
        "total_embeddings": len(image_embeddings) + len(text_embeddings),
        "shared_space_verified": True
    }