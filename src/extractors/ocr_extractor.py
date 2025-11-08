"""
ocr_extractor.py - PASO 3 de la Pipeline de Indexación (CORREGIDO)

Este módulo ahora usa el endpoint 'Agentic Document Extraction' (ADE)
de Landing AI (v1/ade/parse), que es más moderno y potente.

1. Acepta PDFs originales (mucho mejor) y también imágenes PNG.
2. Devuelve Markdown estructurado, no solo texto crudo.
"""

# --- Importaciones ---
import json
import requests  # Se añade 'requests' para llamadas API directas
from pathlib import Path
from typing import List, Dict, Any

# Importaciones de nuestro propio proyecto
from src.utils.logger import get_logger
from src.utils.config import (
    LANDING_AI_API_KEY,
    OCR_PROVIDER,
    EXTRACTED_TEXT_DIR,
    EXCEL_IMAGES_DIR,   # De dónde leer las imágenes de Excel
    INPUT_PDF_DIR       # ¡NUEVO! Leemos los PDF originales
)

# Configuración del logger para este archivo
logger = get_logger(__name__)

# Definir el endpoint moderno de Landing AI
ADE_API_URL = "https://api.va.landing.ai/v1/ade/parse"


# --- 1. La Clase Principal del Extractor OCR (Modificada) ---

class OCRExtractor:
    """
    Cliente para el servicio de extracción de documentos (ADE).
    Maneja tanto PDF como imágenes.
    """

    def __init__(self, provider: str = OCR_PROVIDER):
        """
        Inicializa el extractor de OCR.
        
        Args:
            provider (str): El nombre del servicio a usar ('landing_ai' o 'deepseek').
        """
        self.provider = provider
        logger.info(f"Extractor OCR iniciado con el proveedor: {provider}")

        if provider == 'landing_ai' and not LANDING_AI_API_KEY:
            logger.error("Falta LANDING_AI_API_KEY en el archivo .env")
        # (Se podría añadir la lógica de DeepSeek aquí si se desea)

    def extract_text(self, document_path: str) -> dict:
        """
        Extrae texto/markdown de un documento (PDF o PNG).
        Actúa como un enrutador (router) que llama al método correcto.
        
        Args:
            document_path (str): Ruta completa al archivo (PDF o PNG).
        
        Returns:
            dict: Diccionario con estado y texto (markdown).
        """
        try:
            path_obj = Path(document_path)
            file_name = path_obj.name
            logger.info(f"Extrayendo contenido de: {file_name}")

            if self.provider == "landing_ai":
                # Enrutar basado en la extensión del archivo
                if path_obj.suffix.lower() == ".pdf":
                    return self._extract_ade_pdf(document_path)
                elif path_obj.suffix.lower() == ".png":
                    return self._extract_ade_image(document_path)
                else:
                    logger.warning(f"Tipo de archivo no soportado: {file_name}")
                    return {"status": "error", "message": "Tipo de archivo no soportado"}
            
            # (Aquí iría la lógica para 'deepseek' si se implementa)
            else:
                logger.error(f"Proveedor de OCR no soportado: {self.provider}")
                return {"status": "error", "message": f"Proveedor no soportado: {self.provider}"}
        
        except Exception as e:
            logger.error(f"Error al extraer texto de {document_path}: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _extract_ade_pdf(self, pdf_path: str) -> dict:
        """
        Lógica para llamar al endpoint ADE enviando un archivo PDF.
        """
        headers = {"Authorization": f"Bearer {LANDING_AI_API_KEY}"}
        path_obj = Path(pdf_path)

        try:
            with open(pdf_path, 'rb') as f_pdf:
                files_payload = {
                    'document': (path_obj.name, f_pdf, 'application/pdf'),
                    'model': (None, 'dpt-2-latest'),
                }
                
                response = requests.post(ADE_API_URL, headers=headers, files=files_payload, timeout=120) # Timeout más largo para PDFs
                response.raise_for_status() # Lanza error si no es 2xx

                logger.info(f"ADE (PDF) completado para: {path_obj.name}")
                return {
                    "status": "success",
                    "provider": "landing_ai_ade_pdf",
                    "source_file": path_obj.name,
                    "text": response.json().get('markdown', ''), # ¡Obtenemos Markdown!
                    "raw_response": response.json()
                }

        except requests.exceptions.RequestException as e:
            logger.error(f"Error de API (PDF) para {path_obj.name}: {e}", exc_info=True)
            return {"status": "error", "message": str(e), "source_file": path_obj.name}

    def _extract_ade_image(self, image_path: str) -> dict:
        """
        Lógica para llamar al endpoint ADE enviando un archivo de IMAGEN (PNG).
        """
        headers = {"Authorization": f"Bearer {LANDING_AI_API_KEY}"}
        path_obj = Path(image_path)

        try:
            with open(image_path, 'rb') as f_img:
                files_payload = {
                    'document': (path_obj.name, f_img, 'image/png'), # MIME type es image/png
                    'model': (None, 'dpt-2-latest'),
                }
                
                response = requests.post(ADE_API_URL, headers=headers, files=files_payload, timeout=60)
                response.raise_for_status()

                logger.info(f"ADE (Imagen) completado para: {path_obj.name}")
                return {
                    "status": "success",
                    "provider": "landing_ai_ade_image",
                    "source_file": path_obj.name,
                    "text": response.json().get('markdown', ''), # ¡También Markdown!
                    "raw_response": response.json()
                }

        except requests.exceptions.RequestException as e:
            logger.error(f"Error de API (Imagen) para {path_obj.name}: {e}", exc_info=True)
            return {"status": "error", "message": str(e), "source_file": path_obj.name}


# --- 2. Función de Orquestación (Modificada) ---

def process_all_documents() -> List[Dict[str, Any]]: # RENOMBRADA
    """
    Esta es la función "orquestadora" para el PASO 3 (CORREGIDA).
    
    1. Crea una instancia del `OCRExtractor`.
    2. Busca los PDFs originales en `INPUT_PDF_DIR`.
    3. Busca las imágenes PNG de Excel en `EXCEL_IMAGES_DIR`.
    4. Itera sobre todos y llama a `extractor.extract_text()`.
    5. Guarda el resultado (el Markdown) en `EXTRACTED_TEXT_DIR`.
    """
    logger.info("Iniciando PASO 3: Extracción de Markdown (ADE) de todos los documentos...")
    
    extractor = OCRExtractor()
    results = []
    
    # 1. Obtener los PDFs originales
    pdf_files = list(INPUT_PDF_DIR.glob("*.pdf"))
    logger.info(f"Se encontraron {len(pdf_files)} PDFs originales para procesar.")

    # 2. Obtener las imágenes de Excel (generadas en el Paso 1)
    excel_png_files = list(EXCEL_IMAGES_DIR.glob("*.png"))
    logger.info(f"Se encontraron {len(excel_png_files)} imágenes de Excel para procesar.")

    # 3. Unir ambas listas
    all_document_paths = [Path(p) for p in pdf_files + excel_png_files]
    
    logger.info(f"Se procesarán {len(all_document_paths)} documentos en total.")
    
    # 4. Iterar y procesar cada documento
    for doc_path in all_document_paths:
        
        # 5. Llamar al extractor (que enrutará a PDF o Imagen)
        result = extractor.extract_text(str(doc_path))
        
        if result["status"] == "success":
            # 6. Guardar el markdown extraído en EXTRACTED_TEXT_DIR
            text_file_name = f"{doc_path.stem}_text.json"
            text_file_path = EXTRACTED_TEXT_DIR / text_file_name
            
            try:
                # Guardamos el diccionario completo (status, provider, text, etc.)
                with open(text_file_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Markdown guardado en: {text_file_name}")
                result["output_file"] = text_file_name
                
            except Exception as e:
                logger.error(f"No se pudo guardar el archivo JSON {text_file_name}: {e}")
                result["status"] = "error"
                result["message"] = f"Error al guardar JSON: {e}"
        
        results.append(result)
    
    logger.info(f"Extracción de Markdown completada. {len(results)} archivos procesados.")
    return results