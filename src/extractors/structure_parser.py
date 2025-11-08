"""
structure_parser.py - Parser para estructurar texto OCR
"""

import json
import re
from pathlib import Path
from datetime import datetime
from src.utils.logger import get_logger
from src.utils.config import EXTRACTED_TEXT_DIR, EXTRACTED_TABLES_DIR

logger = get_logger(__name__)


class StructureParser:
    """Parser para estructurar y normalizar texto OCR"""
    
    def __init__(self):
        """Inicializa el parser"""
        logger.info("StructureParser inicializado")
    
    def parse(self, text: str) -> dict:
        """
        Parsea y estructura el texto extraído
        
        Args:
            text: Texto extraído por OCR
        
        Returns:
            dict: Texto estructurado
        """
        try:
            logger.info("Iniciando parse de texto")
            
            structured = {
                "raw_text": text,
                "tables": self.extract_tables(text),
                "fields": self.extract_fields(text),
                "metadata": self.extract_metadata(text),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("Parse completado exitosamente")
            return structured
        
        except Exception as e:
            logger.error(f"Error al parsear: {e}")
            return {"status": "error", "message": str(e)}
    
    def extract_tables(self, text: str) -> list:
        """
        Extrae tablas del texto
        
        Args:
            text: Texto a analizar
        
        Returns:
            list: Tablas estructuradas
        """
        tables = []
        lines = text.split("\n")
        current_table = []
        in_table = False
        
        for line in lines:
            # Detectar líneas de tabla (contienen | o están alineadas)
            if "|" in line or "─" in line or "┌" in line:
                in_table = True
                current_table.append(line)
            else:
                if in_table and current_table:
                    # Procesar tabla completada
                    table_data = self._parse_table_lines(current_table)
                    if table_data:
                        tables.append(table_data)
                    current_table = []
                    in_table = False
        
        # Procesar última tabla si existe
        if current_table:
            table_data = self._parse_table_lines(current_table)
            if table_data:
                tables.append(table_data)
        
        return tables
    
    def _parse_table_lines(self, lines: list) -> dict:
        """
        Convierte líneas de tabla a estructura JSON
        
        Args:
            lines: Líneas que componen la tabla
        
        Returns:
            dict: Tabla estructurada
        """
        rows = []
        headers = []
        
        for i, line in enumerate(lines):
            # Saltar líneas de separación
            if all(c in "─┌┬┐│├┼┤└┴┘" for c in line if c.strip()):
                continue
            
            # Extraer celdas
            cells = [cell.strip() for cell in line.split("|") if cell.strip()]
            
            if i == 0 or not headers:
                headers = cells
            else:
                row_data = dict(zip(headers, cells))
                rows.append(row_data)
        
        return {
            "headers": headers,
            "rows": rows,
            "row_count": len(rows)
        }
    
    def extract_fields(self, text: str) -> dict:
        """
        Extrae campos clave del texto
        
        Args:
            text: Texto a analizar
        
        Returns:
            dict: Campos extraídos
        """
        fields = {
            "dates": self._extract_dates(text),
            "amounts": self._extract_amounts(text),
            "concepts": self._extract_concepts(text),
            "accounts": self._extract_accounts(text)
        }
        return fields
    
    def _extract_dates(self, text: str) -> list:
        """Extrae fechas del texto"""
        patterns = [
            r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",  # DD/MM/YYYY
            r"\d{4}[/-]\d{1,2}[/-]\d{1,2}",    # YYYY/MM/DD
        ]
        
        dates = []
        for pattern in patterns:
            dates.extend(re.findall(pattern, text))
        
        return list(set(dates))
    
    def _extract_amounts(self, text: str) -> list:
        """Extrae montos del texto"""
        patterns = [
            r"S/\s*[\d.,]+",      # Soles
            r"USD\s*[\d.,]+",     # Dólares
            r"\$\s*[\d.,]+",      # Símbolo $
            r"[\d.,]+\s*soles",   # Número + soles
        ]
        
        amounts = []
        for pattern in patterns:
            amounts.extend(re.findall(pattern, text, re.IGNORECASE))
        
        return amounts
    
    def _extract_concepts(self, text: str) -> list:
        """Extrae conceptos/descripciones principales"""
        keywords = [
            "liquidación", "pago", "salario", "gratificación",
            "bonificación", "descuento", "impuesto", "total"
        ]
        
        concepts = []
        text_lower = text.lower()
        
        for keyword in keywords:
            if keyword in text_lower:
                # Encontrar contexto (línea con la palabra)
                for line in text.split("\n"):
                    if keyword in line.lower():
                        concepts.append(line.strip())
        
        return concepts
    
    def _extract_accounts(self, text: str) -> list:
        """Extrae números de cuenta"""
        # Patrón común para cuentas (varía según país)
        pattern = r"\d{10,20}"
        accounts = re.findall(pattern, text)
        return list(set(accounts))
    
    def extract_metadata(self, text: str) -> dict:
        """
        Extrae metadata del documento
        
        Args:
            text: Texto a analizar
        
        Returns:
            dict: Metadata extraída
        """
        metadata = {
            "total_lines": len(text.split("\n")),
            "total_chars": len(text),
            "has_tables": "|" in text or "─" in text,
            "languages_detected": ["es"]  # Asumir español
        }
        return metadata
    
    def save_structure(self, file_name: str, structured_data: dict) -> bool:
        """
        Guarda la estructura en JSON
        
        Args:
            file_name: Nombre del archivo
            structured_data: Datos estructurados
        
        Returns:
            bool: True si fue exitoso
        """
        try:
            output_path = EXTRACTED_TABLES_DIR / f"{Path(file_name).stem}_structure.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Estructura guardada: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error al guardar estructura: {e}")
            return False


def process_all_extracted_text() -> list:
    """
    Procesa todo el texto extraído
    
    Returns:
        list: Resultados del procesamiento
    """
    parser = StructureParser()
    results = []
    
    text_files = list(EXTRACTED_TEXT_DIR.glob("*_text.json"))
    logger.info(f"Procesando {len(text_files)} archivos de texto extraído")
    
    for text_file in text_files:
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data.get("status") == "success":
                text = data.get("text", "")
                structured = parser.parse(text)
                parser.save_structure(text_file.stem, structured)
                results.append({"status": "success", "file": text_file.name})
            else:
                results.append({"status": "skipped", "file": text_file.name})
        
        except Exception as e:
            logger.error(f"Error procesando {text_file}: {e}")
            results.append({"status": "error", "file": text_file.name, "error": str(e)})
    
    return results