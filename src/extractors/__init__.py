"""
extractors/__init__.py

Exporta las clases y funciones principales del módulo de extracción.
"""

from src.extractors.ocr_extractor import OCRExtractor, process_all_images
from src.extractors.structure_parser import StructureParser, process_all_extracted_text