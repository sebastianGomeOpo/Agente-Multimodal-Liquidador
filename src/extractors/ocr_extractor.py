"""
ocr_extractor.py - Extracción OCR de imágenes
"""

import json
import base64
from pathlib import Path
import requests
from src.utils.logger import get_logger
from src.utils.config import (
    LANDING_AI_API_KEY, DEEPSEEK_API_KEY, OCR_PROVIDER,
    EXTRACTED_TEXT_DIR, EXCEL_IMAGES_DIR, PDF_IMAGES_DIR
)

logger = get_logger(__name__)


class OCRExtractor:
    """Extractor OCR usando Landing AI o DeepSeek"""
    
    def __init__(self, provider: str = OCR_PROVIDER):
        """
        Inicializa el extractor OCR
        
        Args:
            provider: 'landing_ai' o 'deepseek'
        """
        self.provider = provider
        logger.info(f"OCR Extractor iniciado con provider: {provider}")
    
    def extract_text(self, image_path: str) -> dict:
        """
        Extrae texto de una imagen
        
        Args:
            image_path: Ruta a la imagen
        
        Returns:
            dict: Texto extraído y metadata
        """
        try:
            logger.info(f"Extrayendo texto de: {image_path}")
            
            if self.provider == "landing_ai":
                return self._extract_landing_ai(image_path)
            elif self.provider == "deepseek":
                return self._extract_deepseek(image_path)
            else:
                return {"status": "error", "message": f"Provider no soportado: {self.provider}"}
        
        except Exception as e:
            logger.error(f"Error al extraer texto: {e}")
            return {"status": "error", "message": str(e)}
    
    def _extract_landing_ai(self, image_path: str) -> dict:
        """
        Extrae texto usando Landing AI API
        
        Args:
            image_path: Ruta a la imagen
        
        Returns:
            dict: Texto extraído
        """
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            headers = {
                "Authorization": f"Bearer {LANDING_AI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "image": f"data:image/png;base64,{image_data}"
            }
            
            # Endpoint de Landing AI para OCR
            url = "https://api.landing.ai/v1/document_understanding"
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            logger.info(f"Landing AI OCR completado")
            
            return {
                "status": "success",
                "provider": "landing_ai",
                "text": result.get("text", ""),
                "raw_response": result
            }
        
        except Exception as e:
            logger.error(f"Error con Landing AI: {e}")
            return {"status": "error", "message": str(e)}
    
    def _extract_deepseek(self, image_path: str) -> dict:
        """
        Extrae texto usando DeepSeek API
        
        Args:
            image_path: Ruta a la imagen
        
        Returns:
            dict: Texto extraído
        """
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-vision",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_data}"}
                            },
                            {
                                "type": "text",
                                "text": "Extrae todo el texto visible en esta imagen. Mantén la estructura y formato."
                            }
                        ]
                    }
                ]
            }
            
            url = "https://api.deepseek.com/v1/chat/completions"
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            
            logger.info(f"DeepSeek OCR completado")
            
            return {
                "status": "success",
                "provider": "deepseek",
                "text": text,
                "raw_response": result
            }
        
        except Exception as e:
            logger.error(f"Error con DeepSeek: {e}")
            return {"status": "error", "message": str(e)}
    
    def extract_structure(self, extracted_text: str) -> dict:
        """
        Analiza la estructura del texto extraído
        
        Args:
            extracted_text: Texto extraído por OCR
        
        Returns:
            dict: Estructura detectada (tablas, campos, etc.)
        """
        try:
            structure = {
                "lines": extracted_text.split("\n"),
                "tables": self._detect_tables(extracted_text),
                "key_fields": self._extract_key_fields(extracted_text)
            }
            return structure
        except Exception as e:
            logger.error(f"Error al analizar estructura: {e}")
            return {"error": str(e)}
    
    def _detect_tables(self, text: str) -> list:
        """
        Detecta tablas en el texto
        
        Args:
            text: Texto a analizar
        
        Returns:
            list: Tablas detectadas
        """
        # Detección básica de tablas (puede mejorarse)
        tables = []
        lines = text.split("\n")
        current_table = []
        
        for line in lines:
            if "|" in line or "─" in line:
                current_table.append(line)
            else:
                if current_table:
                    tables.append("\n".join(current_table))
                    current_table = []
        
        return tables
    
    def _extract_key_fields(self, text: str) -> dict:
        """
        Extrae campos clave como fechas, montos, conceptos
        
        Args:
            text: Texto a analizar
        
        Returns:
            dict: Campos clave detectados
        """
        import re
        
        fields = {
            "dates": re.findall(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", text),
            "amounts": re.findall(r"[S/USD]\s*[\d.,]+", text),
            "concepts": []
        }
        
        return fields


def process_all_images() -> list:
    """
    Procesa todas las imágenes generadas
    
    Returns:
        list: Resultados del procesamiento
    """
    extractor = OCRExtractor()
    results = []
    
    # Procesar imágenes de Excel
    excel_images = list(EXCEL_IMAGES_DIR.glob("*.png"))
    logger.info(f"Procesando {len(excel_images)} imágenes de Excel")
    
    for image in excel_images:
        result = extractor.extract_text(str(image))
        if result["status"] == "success":
            # Guardar texto extraído
            text_file = EXTRACTED_TEXT_DIR / f"{image.stem}_text.json"
            with open(text_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Texto guardado: {text_file}")
        results.append(result)
    
    # Procesar imágenes de PDF
    pdf_images = list(PDF_IMAGES_DIR.glob("*.png"))
    logger.info(f"Procesando {len(pdf_images)} imágenes de PDF")
    
    for image in pdf_images:
        result = extractor.extract_text(str(image))
        if result["status"] == "success":
            text_file = EXTRACTED_TEXT_DIR / f"{image.stem}_text.json"
            with open(text_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Texto guardado: {text_file}")
        results.append(result)
    
    return results