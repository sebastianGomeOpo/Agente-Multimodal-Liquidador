"""
ocr_extractor.py - PASO 3 de la Pipeline de Indexación

Este módulo se encarga de tomar las imágenes generadas en el PASO 2
(tanto de Excel como de PDF) y usar un servicio de OCR (Reconocimiento
Óptico de Caracteres) para extraer todo el texto en bruto.

El resultado de este paso es un archivo JSON (`_text.json`) por cada
imagen, que contiene el texto crudo. Este archivo será el "input"
para el `structure_parser` (PASO 4).
"""

# --- Importaciones ---
import json     # Para crear el archivo JSON de salida
import base64   # Para codificar imágenes y enviarlas en una solicitud API
import re       # Para expresiones regulares (usado en el parser)
import requests # Para hacer llamadas a las APIs de OCR (Landing AI, DeepSeek)
from pathlib import Path # Para manejar rutas de archivos
from typing import List, Dict, Any # Para type hints

# Importaciones de nuestro propio proyecto
from src.utils.logger import get_logger
from src.utils.config import (
    LANDING_AI_API_KEY,  # Clave API para Landing AI
    DEEPSEEK_API_KEY,    # Clave API para DeepSeek
    OCR_PROVIDER,        # Cuál de los dos proveedores usar (definido en .env)
    EXTRACTED_TEXT_DIR,  # Dónde guardar los archivos _text.json
    EXCEL_IMAGES_DIR,    # De dónde leer las imágenes de Excel
    PDF_IMAGES_DIR       # De dónde leer las imágenes de PDF
)

# Configuración del logger para este archivo
logger = get_logger(__name__)


# --- 1. La Clase Principal del Extractor OCR ---

class OCRExtractor:
    """
    Esta clase actúa como un "cliente" para los servicios de OCR.
    Su única responsabilidad es tomar la ruta de una imagen y devolver el texto.
    """
    
    def __init__(self, provider: str = OCR_PROVIDER):
        """
        Inicializa el extractor de OCR.
        
        Args:
            provider (str): El nombre del servicio a usar ('landing_ai' o 'deepseek').
                            Esto se carga desde el archivo .env a través de config.py.
        """
        self.provider = provider
        logger.info(f"Extractor OCR iniciado con el proveedor: {provider}")
        
        # Validar que las API keys necesarias estén presentes
        if provider == 'landing_ai' and not LANDING_AI_API_KEY:
            logger.error("Falta LANDING_AI_API_KEY en el archivo .env")
        elif provider == 'deepseek' and not DEEPSEEK_API_KEY:
            logger.error("Falta DEEPSEEK_API_KEY en el archivo .env")

    def extract_text(self, image_path: str) -> dict:
        """
        Extrae texto de una imagen usando el proveedor seleccionado.
        Esta es la función pública principal de la clase.
        
        Args:
            image_path (str): La ruta completa al archivo de imagen (ej. ".../imagen.png")
        
        Returns:
            dict: Un diccionario con el estado ("success" o "error") y el texto extraído.
        """
        try:
            logger.info(f"Extrayendo texto de: {Path(image_path).name}")
            
            # Selecciona el método privado correcto basado en el proveedor
            if self.provider == "landing_ai":
                return self._extract_landing_ai(image_path)
            elif self.provider == "deepseek":
                return self._extract_deepseek(image_path)
            else:
                logger.error(f"Proveedor de OCR no soportado: {self.provider}")
                return {"status": "error", "message": f"Proveedor no soportado: {self.provider}"}
        
        except Exception as e:
            logger.error(f"Error al extraer texto de {image_path}: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
    
    def _extract_landing_ai(self, image_path: str) -> dict:
        """
        Lógica específica para llamar a la API de Landing AI.
        """
        try:
            # 1. Leer la imagen y codificarla en base64
            # Las APIs no aceptan archivos; aceptan el "texto" de la imagen (base64)
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            # 2. Configurar los Headers de la solicitud (Autenticación)
            headers = {
                "Authorization": f"Bearer {LANDING_AI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # 3. Configurar el Payload (el cuerpo de la solicitud)
            # Le decimos a la API que le estamos enviando una imagen PNG en base64
            payload = {
                "image": f"data:image/png;base64,{image_data}"
            }
            
            # 4. Definir el Endpoint de la API
            url = "https://api.landing.ai/v1/document_understanding"
            
            # 5. Hacer la llamada a la API
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status() # Lanza un error si la respuesta no es 2xx
            
            result = response.json()
            
            logger.info(f"Landing AI OCR completado para: {Path(image_path).name}")
            
            # 6. Devolver la respuesta en un formato estándar
            return {
                "status": "success",
                "provider": "landing_ai",
                "text": result.get("text", ""), # El texto extraído
                "raw_response": result # La respuesta completa de la API
            }
        
        except Exception as e:
            logger.error(f"Error con la API de Landing AI: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
    
    def _extract_deepseek(self, image_path: str) -> dict:
        """
        Lógica específica para llamar a la API de DeepSeek (que es multimodal).
        """
        try:
            # 1. Leer y codificar la imagen en base64
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            # 2. Configurar Headers (Autenticación)
            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # 3. Configurar el Payload
            # DeepSeek usa un formato de "chat" (mensajes)
            payload = {
                "model": "deepseek-vision", # El modelo multimodal
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                # Parte 1: La imagen
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_data}"}
                            },
                            {
                                # Parte 2: El prompt (la instrucción)
                                "type": "text",
                                "text": "Extrae todo el texto visible en esta imagen. Mantén la estructura y formato."
                            }
                        ]
                    }
                ]
            }
            
            # 4. Definir el Endpoint
            url = "https://api.deepseek.com/v1/chat/completions"
            
            # 5. Hacer la llamada a la API
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            # 6. Extraer el texto de la respuesta del chat
            text = result["choices"][0]["message"]["content"]
            
            logger.info(f"DeepSeek OCR completado para: {Path(image_path).name}")
            
            # 7. Devolver en formato estándar
            return {
                "status": "success",
                "provider": "deepseek",
                "text": text,
                "raw_response": result
            }
        
        except Exception as e:
            logger.error(f"Error con la API de DeepSeek: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    # NOTA: Los métodos 'extract_structure', '_detect_tables' y '_extract_key_fields'
    # han sido eliminados.
    # Esa lógica ahora es responsabilidad exclusiva de 'structure_parser.py',
    # que es mucho más potente por usar un LLM.


# --- 2. Función de Orquestación (La que llama main.py) ---

def process_all_images() -> List[Dict[str, Any]]:
    """
    Esta es la función "orquestadora" para el PASO 3.
    
    1. Crea una instancia del `OCRExtractor`.
    2. Busca todas las imágenes PNG en las carpetas de salida del PASO 2.
    3. Itera sobre cada imagen y llama a `extractor.extract_text()`.
    4. Guarda el resultado (el texto crudo) en un nuevo archivo JSON
       en la carpeta `EXTRACTED_TEXT_DIR`.
       
    Estos archivos JSON son el "producto" de este módulo y el "insumo"
    para el `structure_parser`.
    """
    logger.info("Iniciando PASO 3: Extracción OCR de todas las imágenes...")
    
    extractor = OCRExtractor() # Usa el provider definido en .env
    results = []
    
    # 1. Unir las listas de imágenes de Excel y PDF
    excel_images = list(EXCEL_IMAGES_DIR.glob("*.png"))
    pdf_images = list(PDF_IMAGES_DIR.glob("*.png"))
    all_images = excel_images + pdf_images
    
    logger.info(f"Se encontraron {len(all_images)} imágenes totales para procesar.")
    
    # 2. Iterar y procesar cada imagen
    for image_path in all_images:
        # 3. Llamar al extractor (Landing AI o DeepSeek)
        result = extractor.extract_text(str(image_path))
        
        if result["status"] == "success":
            # 4. Guardar el texto extraído en EXTRACTED_TEXT_DIR
            # El nombre del archivo será, ej: "mi_imagen_page_1_text.json"
            text_file_name = f"{image_path.stem}_text.json"
            text_file_path = EXTRACTED_TEXT_DIR / text_file_name
            
            try:
                # Guardamos el diccionario completo (status, provider, text, etc.)
                with open(text_file_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Texto guardado en: {text_file_name}")
                result["output_file"] = text_file_name
                
            except Exception as e:
                logger.error(f"No se pudo guardar el archivo JSON {text_file_name}: {e}")
                result["status"] = "error"
                result["message"] = f"Error al guardar JSON: {e}"
        
        results.append(result)
    
    logger.info(f"Extracción OCR completada. {len(results)} archivos procesados.")
    return results