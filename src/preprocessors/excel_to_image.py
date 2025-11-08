"""
excel_to_image.py - Conversión de Excel a imagen PNG
"""

import json
from pathlib import Path
from datetime import datetime
import openpyxl
from PIL import Image
import io
from src.utils.logger import get_logger
from src.utils.config import INPUT_EXCEL_DIR, EXCEL_IMAGES_DIR

logger = get_logger(__name__)


def excel_to_image(excel_path: str, sheet_name: str = None, range_to_capture: str = None) -> dict:
    """
    Convierte un rango específico de Excel a imagen PNG
    
    Args:
        excel_path: Ruta al archivo Excel
        sheet_name: Nombre de la hoja (default: primera hoja)
        range_to_capture: Rango a capturar, ej: "A1:D10" (default: A1:Z100)
    
    Returns:
        dict: Información sobre la imagen generada
    """
    try:
        logger.info(f"Iniciando conversión de Excel: {excel_path}")
        
        # Abrir archivo Excel
        wb = openpyxl.load_workbook(excel_path)
        ws = wb.active if sheet_name is None else wb[sheet_name]
        
        # Rango por defecto si no se especifica
        if range_to_capture is None:
            range_to_capture = "A1:Z100"
        
        logger.info(f"Usando rango: {range_to_capture}")
        
        # Obtener el rango de celdas
        cells = ws[range_to_capture]
        
        # Obtener dimensiones
        if isinstance(cells, tuple):
            rows = [cell for cell in cells]
        else:
            rows = [[cells]]
        
        # Crear HTML para visualización
        html_content = _create_html_from_cells(ws, range_to_capture)
        
        # Guardar información de metadata
        file_name = Path(excel_path).stem
        metadata = {
            "original_file": Path(excel_path).name,
            "sheet_name": ws.title,
            "range": range_to_capture,
            "timestamp": datetime.now().isoformat(),
            "file_name": f"{file_name}.png",
            "html_content": html_content
        }
        
        logger.info(f"Excel procesado: {file_name}")
        
        return {
            "status": "success",
            "metadata": metadata,
            "html": html_content
        }
    
    except Exception as e:
        logger.error(f"Error al convertir Excel a imagen: {e}")
        return {"status": "error", "message": str(e)}


def _create_html_from_cells(ws, range_to_capture: str) -> str:
    """
    Crea HTML con la tabla del rango especificado
    
    Args:
        ws: Worksheet de openpyxl
        range_to_capture: Rango a capturar
    
    Returns:
        str: HTML de la tabla
    """
    html = "<table border='1' cellpadding='5' cellspacing='0' style='font-family: Arial;'>"
    
    for row in ws[range_to_capture]:
        html += "<tr>"
        for cell in row:
            value = cell.value if cell.value is not None else ""
            html += f"<td>{value}</td>"
        html += "</tr>"
    
    html += "</table>"
    return html


def html_to_png(html_content: str, output_path: str, width: int = 1200, height: int = 800) -> bool:
    """
    Convierte HTML a PNG usando una librería (requiere instalación adicional)
    Para producción, usar imgkit o similar
    
    Args:
        html_content: Contenido HTML
        output_path: Ruta de salida
        width: Ancho de la imagen
        height: Alto de la imagen
    
    Returns:
        bool: True si fue exitoso
    """
    try:
        # NOTA: Requiere wkhtmltoimage o similar instalado
        # Para desarrollo, usar solución alternativa
        logger.warning("html_to_png requiere configuración adicional. Usar alternativa.")
        return False
    except Exception as e:
        logger.error(f"Error al convertir HTML a PNG: {e}")
        return False


def save_excel_metadata(file_name: str, metadata: dict) -> bool:
    """
    Guarda metadata del Excel procesado
    
    Args:
        file_name: Nombre del archivo
        metadata: Información de metadata
    
    Returns:
        bool: True si fue exitoso
    """
    try:
        metadata_path = EXCEL_IMAGES_DIR / f"{Path(file_name).stem}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Metadata guardada: {metadata_path}")
        return True
    except Exception as e:
        logger.error(f"Error al guardar metadata: {e}")
        return False


def process_all_excels() -> list:
    """
    Procesa todos los Excel en el directorio de entrada
    
    Returns:
        list: Lista de resultados
    """
    results = []
    excel_files = list(INPUT_EXCEL_DIR.glob("*.xlsx"))
    
    logger.info(f"Encontrados {len(excel_files)} archivos Excel")
    
    for excel_file in excel_files:
        result = excel_to_image(str(excel_file))
        if result["status"] == "success":
            save_excel_metadata(excel_file.name, result["metadata"])
        results.append(result)
    
    return results