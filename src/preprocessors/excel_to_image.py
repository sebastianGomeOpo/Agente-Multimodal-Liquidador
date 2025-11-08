"""
excel_to_image.py - Conversión de Excel a imagen PNG con chunking
"""

import json
from pathlib import Path
from datetime import datetime
import openpyxl
import matplotlib.pyplot as plt
import logging

from src.utils.config import INPUT_EXCEL_DIR, EXCEL_IMAGES_DIR, EXCEL_CHUNK_ROWS, EXCEL_CHUNK_COLS, EXCEL_DPI

logger = logging.getLogger(__name__)


def _is_temp_file(file_path: Path) -> bool:
    """Detecta archivos temporales de Excel"""
    return file_path.name.startswith('~$')


def excel_to_image(excel_path: str, sheet_name: str = None, range_to_capture: str = None) -> dict:
    """
    Convierte un rango de Excel en múltiples imágenes PNG (chunks de 50x50).
    
    Args:
        excel_path: Ruta al archivo Excel
        sheet_name: Nombre de la hoja (default: primera hoja)
        range_to_capture: Rango a capturar (default: auto-detectar)
    
    Returns:
        dict: Información sobre las imágenes generadas
    """
    try:
        # Validar que el archivo existe
        if not Path(excel_path).exists():
            logger.error(f"Archivo no encontrado: {excel_path}")
            return {"status": "error", "message": "Archivo no encontrado"}
        
        logger.info(f"Convirtiendo Excel: {Path(excel_path).name}")
        
        wb = openpyxl.load_workbook(excel_path, data_only=True)
        ws = wb.active if sheet_name is None else wb[sheet_name]
        
        if range_to_capture is None:
            range_to_capture = ws.dimensions
        
        # Extraer datos
        data = []
        try:
            for row in ws[range_to_capture]:
                data.append([cell.value if cell.value is not None else "" for cell in row])
        except Exception as e:
            logger.error(f"Error leyendo rango '{range_to_capture}': {e}")
            return {"status": "error", "message": f"Rango inválido: {range_to_capture}"}
        
        # Filtrar filas vacías
        data = [row for row in data if any(str(cell).strip() != "" for cell in row)]
        
        if not data:
            logger.warning(f"Rango vacío en {excel_path}")
            return {"status": "error", "message": "Rango vacío"}
        
        num_rows = len(data)
        num_cols = len(data[0]) if num_rows > 0 else 0
        
        # Chunking
        generated_image_paths = []
        file_name_stem = Path(excel_path).stem
        
        for r_start in range(0, num_rows, EXCEL_CHUNK_ROWS):
            for c_start in range(0, num_cols, EXCEL_CHUNK_COLS):
                r_end = min(r_start + EXCEL_CHUNK_ROWS, num_rows)
                c_end = min(c_start + EXCEL_CHUNK_COLS, num_cols)
                
                chunk_data = [row[c_start:c_end] for row in data[r_start:r_end]]
                
                # Validar que el chunk tiene contenido
                if not chunk_data or not any(any(str(cell).strip() for cell in row) for row in chunk_data):
                    continue
                
                # Crear imagen del chunk
                num_chunk_rows = len(chunk_data)
                num_chunk_cols = len(chunk_data[0])
                
                fig_width = max(15, num_chunk_cols * 1.5)
                fig_height = max(5, num_chunk_rows * 0.35)
                
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                ax.axis('off')
                
                table = ax.table(cellText=chunk_data, loc='center', cellLoc='left')
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 1.2)
                
                chunk_image_name = f"{file_name_stem}_chunk_r{r_start}_c{c_start}.png"
                image_path = EXCEL_IMAGES_DIR / chunk_image_name
                
                plt.savefig(image_path, bbox_inches='tight', dpi=EXCEL_DPI)
                plt.close(fig)
                
                generated_image_paths.append(str(image_path))
        
        if not generated_image_paths:
            logger.warning(f"No se generaron chunks para {excel_path}")
            return {"status": "error", "message": "No se generaron chunks"}
        
        logger.info(f"✓ Generados {len(generated_image_paths)} chunks PNG")
        
        # Guardar metadata
        metadata = {
            "original_file": Path(excel_path).name,
            "image_files": generated_image_paths,
            "sheet_name": ws.title,
            "range": range_to_capture,
            "chunk_size": f"{EXCEL_CHUNK_ROWS}x{EXCEL_CHUNK_COLS}",
            "timestamp": datetime.now().isoformat()
        }
        
        metadata_path = EXCEL_IMAGES_DIR / f"{file_name_stem}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return {"status": "success", "metadata": metadata}
    
    except Exception as e:
        logger.error(f"Error convirtiendo Excel: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


def process_all_excels() -> list:
    """Procesa todos los Excel en el directorio de entrada"""
    results = []
    excel_files = list(INPUT_EXCEL_DIR.glob("*.xlsx"))
    
    # Filtrar archivos temporales
    excel_files = [f for f in excel_files if not _is_temp_file(f)]
    
    logger.info(f"Encontrados {len(excel_files)} archivos Excel")
    
    for excel_file in excel_files:
        result = excel_to_image(str(excel_file))
        results.append(result)
    
    return results