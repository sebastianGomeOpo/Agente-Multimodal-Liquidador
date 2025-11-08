"""
excel_to_image.py - Conversión de Excel a imagen PNG
(Versión corregida usando Matplotlib para renderizar la imagen)
"""

import json
from pathlib import Path
from datetime import datetime
import openpyxl
import matplotlib.pyplot as plt
import logging # Usar logging en lugar de get_logger para evitar error de importación circular
from src.utils.config import INPUT_EXCEL_DIR, EXCEL_IMAGES_DIR

# Configura el logger para este módulo
logger = logging.getLogger(__name__)


def excel_to_image(excel_path: str, sheet_name: str = None, range_to_capture: str = None) -> dict:
    """
    Convierte un rango específico de Excel a imagen PNG usando Matplotlib.
    
    Args:
        excel_path: Ruta al archivo Excel
        sheet_name: Nombre de la hoja (default: primera hoja)
        range_to_capture: Rango a capturar, ej: "A1:Z100" (default: A1:Z100)
    
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
        
        # --- Lógica de Matplotlib para crear la imagen ---
        
        # 1. Extraer los datos de las celdas a una lista de listas
        data = []
        for row in ws[range_to_capture]:
            data.append([cell.value if cell.value is not None else "" for cell in row])
        
        # 2. Filtrar filas/columnas vacías que openpyxl a veces lee
        data = [row for row in data if any(cell != "" for cell in row)]
        if not data:
            logger.warning(f"No se encontraron datos en el rango {range_to_capture} para {excel_path}")
            return {"status": "error", "message": "Rango vacío o sin datos"}

        num_rows = len(data)
        num_cols = len(data[0])

        # 3. Crear una figura de Matplotlib
        # Ajustar el tamaño de la figura basado en la cantidad de datos
        # Estos valores son experimentales, puedes ajustarlos
        fig_width = max(15, num_cols * 1.5) 
        fig_height = max(5, num_rows * 0.35)
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('off') # Ocultar los ejes del gráfico

        # 4. "Dibujar" la tabla en la figura
        table = ax.table(cellText=data, loc='center', cellLoc='left')
        
        # 5. Estilizar la tabla
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2) # Escalar para que se ajuste mejor

        # 6. Definir la ruta de salida de la imagen
        file_name = Path(excel_path).stem
        image_name = f"{file_name}.png"
        image_path = EXCEL_IMAGES_DIR / image_name
        
        # 7. Guardar la figura como PNG
        plt.savefig(
            image_path, 
            bbox_inches='tight', # Recortar el espacio en blanco
            dpi=200                # Buena resolución para OCR
        )
        plt.close(fig) # Cerrar la figura para liberar memoria
        
        # --------------------------------------------------
        
        logger.info(f"Excel procesado y guardado como PNG: {image_name}")

        # Guardar información de metadata
        metadata = {
            "original_file": Path(excel_path).name,
            "image_file": image_name,
            "image_path": str(image_path),
            "sheet_name": ws.title,
            "range": range_to_capture,
            "timestamp": datetime.now().isoformat()
        }
        
        # Guardar el JSON de metadata
        save_excel_metadata(file_name, metadata)
        
        return {
            "status": "success",
            "metadata": metadata
        }
    
    except Exception as e:
        logger.error(f"Error al convertir Excel a imagen: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


def save_excel_metadata(file_name: str, metadata: dict) -> bool:
    """
    Guarda metadata del Excel procesado en un archivo JSON.
    
    Args:
        file_name: Nombre del archivo (sin extensión)
        metadata: Información de metadata
    
    Returns:
        bool: True si fue exitoso
    """
    try:
        # Usamos .stem para asegurarnos de no tener doble extensión
        metadata_path = EXCEL_IMAGES_DIR / f"{Path(file_name).stem}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Metadata de Excel guardada: {metadata_path.name}")
        return True
    except Exception as e:
        logger.error(f"Error al guardar metadata de Excel: {e}")
        return False


def process_all_excels() -> list:
    """
    Procesa todos los Excel en el directorio de entrada.
    
    Returns:
        list: Lista de resultados
    """
    results = []
    excel_files = list(INPUT_EXCEL_DIR.glob("*.xlsx"))
    
    logger.info(f"Encontrados {len(excel_files)} archivos Excel")
    
    for excel_file in excel_files:
        result = excel_to_image(str(excel_file))
        results.append(result)
    
    return results

# --- Código de prueba (para ejecutar este archivo directamente) ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Iniciando prueba de conversión Excel -> PNG...")
    
    # Asegurarnos que los directorios existan
    INPUT_EXCEL_DIR.mkdir(parents=True, exist_ok=True)
    EXCEL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Crear un Excel de prueba si no hay ninguno
    test_excel_path = INPUT_EXCEL_DIR / "test_excel_file.xlsx"
    if not any(INPUT_EXCEL_DIR.glob("*.xlsx")):
        try:
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Hoja1"
            ws['A1'] = "Concepto"
            ws['B1'] = "Monto"
            ws['A2'] = "Servicio A"
            ws['B2'] = 150.75
            ws['A3'] = "Servicio B"
            ws['B3'] = 200.00
            ws['A4'] = "Total"
            ws['B4'] = 350.75
            wb.save(test_excel_path)
            logger.info(f"Creado archivo Excel de prueba: {test_excel_path.name}")
        except Exception as e:
            logger.error(f"No se pudo crear el archivo Excel de prueba: {e}")

    # Ejecutar el procesamiento
    process_results = process_all_excels()
    
    print("\n--- Resultados del Procesamiento ---")
    print(json.dumps(process_results, indent=2))
    
    if process_results and process_results[0].get('status') == 'success':
        image_path = process_results[0].get('metadata', {}).get('image_path')
        if image_path:
            logger.info(f"¡Éxito! Imagen generada en: {image_path}")