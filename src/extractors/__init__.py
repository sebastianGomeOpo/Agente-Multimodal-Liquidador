"""
extractors/__init__.py

Exporta las clases y funciones principales del módulo de extracción (Pasos 3 y 4).
"""

# Importa las clases y funciones del PASO 3 (OCR)
from .ocr_extractor import OCRExtractor, process_all_images

# Importa las clases y funciones del PASO 4 (Parser)
# Ahora 'process_all_extracted_text' existe y puede ser importado.
from .structure_parser import StructureParser, process_all_extracted_text