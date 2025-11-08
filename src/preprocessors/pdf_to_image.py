"""
pdf_to_image.py - Conversión de PDF a imágenes PNG
"""

import json
from pathlib import Path
from datetime import datetime
from pdf2image import convert_from_path
from src.utils.logger import get_logger
from src.utils.config import INPUT_PDF_DIR, PDF_IMAGES_DIR

logger = get_logger(__name__)


def pdf_to_images(pdf_path: str, dpi: int = 300, page_range: list = None) -> dict:
    """
    Convierte páginas del PDF a imágenes PNG
    
    Args:
        pdf_path: Ruta al archivo PDF
        dpi: Resolución (default: 300)
        page_range: Lista de páginas específicas (default: todas)
    
    Returns:
        dict: Información sobre las imágenes generadas
    """
    try:
        logger.info(f"Iniciando conversión de PDF: {pdf_path}")
        
        # Convertir PDF a imágenes
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=page_range[0] if page_range else None,
            last_page=page_range[1] if page_range else None
        )
        
        logger.info(f"PDF convertido: {len(images)} páginas")
        
        # Guardar imágenes
        file_name = Path(pdf_path).stem
        image_paths = []
        
        for page_num, image in enumerate(images, 1):
            image_name = f"{file_name}_page_{page_num}.png"
            image_path = PDF_IMAGES_DIR / image_name
            image.save(str(image_path), "PNG")
            image_paths.append(image_name)
            logger.info(f"Página {page_num} guardada: {image_name}")
        
        # Metadata
        metadata = {
            "original_file": Path(pdf_path).name,
            "total_pages": len(images),
            "dpi": dpi,
            "timestamp": datetime.now().isoformat(),
            "image_files": image_paths,
            "page_range": page_range
        }
        
        save_pdf_metadata(file_name, metadata)
        
        return {
            "status": "success",
            "metadata": metadata,
            "image_count": len(images)
        }
    
    except Exception as e:
        logger.error(f"Error al convertir PDF a imágenes: {e}")
        return {"status": "error", "message": str(e), "image_count": 0}


def save_pdf_metadata(file_name: str, metadata: dict) -> bool:
    """
    Guarda metadata del PDF procesado
    
    Args:
        file_name: Nombre del archivo
        metadata: Información de metadata
    
    Returns:
        bool: True si fue exitoso
    """
    try:
        metadata_path = PDF_IMAGES_DIR / f"{file_name}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Metadata guardada: {metadata_path}")
        return True
    except Exception as e:
        logger.error(f"Error al guardar metadata: {e}")
        return False


def get_pdf_info(pdf_path: str) -> dict:
    """
    Obtiene información sobre el PDF (cantidad de páginas)
    
    Args:
        pdf_path: Ruta al archivo PDF
    
    Returns:
        dict: Información del PDF
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        return {
            "total_pages": len(reader.pages),
            "file_name": Path(pdf_path).name
        }
    except Exception as e:
        logger.error(f"Error al obtener info del PDF: {e}")
        return {"error": str(e)}


def process_all_pdfs(dpi: int = 300) -> list:
    """
    Procesa todos los PDF en el directorio de entrada
    
    Args:
        dpi: Resolución para conversión
    
    Returns:
        list: Lista de resultados
    """
    results = []
    pdf_files = list(INPUT_PDF_DIR.glob("*.pdf"))
    
    logger.info(f"Encontrados {len(pdf_files)} archivos PDF")
    
    for pdf_file in pdf_files:
        result = pdf_to_images(str(pdf_file), dpi=dpi)
        results.append(result)
    
    return results