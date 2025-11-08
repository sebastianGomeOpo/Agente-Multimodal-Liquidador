"""
extractors/__init__.py

Exporta las clases y funciones principales del módulo de extracción (Pasos 3 y 4).
"""

# Importa las clases y funciones del PASO 3 (OCR)
# (CORREGIDO: exporta la nueva función 'process_all_documents')
from .ocr_extractor import OCRExtractor, process_all_documents

# Importa las clases y funciones del PASO 4 (Parser)
from .structure_parser import StructureParser, process_all_extracted_text