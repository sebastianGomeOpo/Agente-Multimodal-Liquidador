"""
tools.py - Herramientas para el agente
"""

from typing import List, Any, Dict
import re
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_totals(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    TOOL 1: Calcula sumas, promedios y estadísticas
    
    Args:
        data: Lista de datos con valores numéricos
    
    Returns:
        dict: Resultados de cálculos
    """
    try:
        logger.info(f"Calculando totales de {len(data)} elementos")
        
        if not data:
            return {"status": "error", "message": "Sin datos"}
        
        # Extraer números de los diccionarios
        values = []
        for item in data:
            if isinstance(item, dict):
                for v in item.values():
                    if isinstance(v, (int, float)):
                        values.append(float(v))
            elif isinstance(item, (int, float)):
                values.append(float(item))
        
        if not values:
            return {"status": "error", "message": "Sin valores numéricos"}
        
        result = {
            "status": "success",
            "total": sum(values),
            "promedio": sum(values) / len(values),
            "minimo": min(values),
            "maximo": max(values),
            "cantidad": len(values)
        }
        
        logger.info(f"Totales calculados: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Error al calcular totales: {e}")
        return {"status": "error", "message": str(e)}


def validate_dates(date_string: str) -> Dict[str, Any]:
    """
    TOOL 2: Valida y normaliza fechas
    
    Args:
        date_string: String de fecha
    
    Returns:
        dict: Fecha validada y normalizada
    """
    try:
        logger.info(f"Validando fecha: {date_string}")
        
        # Patrones comunes
        patterns = [
            (r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", "%d/%m/%Y"),
            (r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", "%Y/%m/%d"),
        ]
        
        for pattern, date_format in patterns:
            match = re.search(pattern, date_string)
            if match:
                try:
                    # Intentar parsear
                    date_obj = datetime.strptime(match.group(0), date_format)
                    
                    result = {
                        "status": "success",
                        "original": date_string,
                        "parsed": date_obj.isoformat(),
                        "formatted": date_obj.strftime("%d/%m/%Y"),
                        "year": date_obj.year,
                        "month": date_obj.month,
                        "day": date_obj.day
                    }
                    
                    logger.info(f"Fecha validada: {result}")
                    return result
                except:
                    continue
        
        return {"status": "error", "message": f"Formato de fecha no reconocido: {date_string}"}
    
    except Exception as e:
        logger.error(f"Error al validar fecha: {e}")
        return {"status": "error", "message": str(e)}


def compare_values(value1: Any, value2: Any) -> Dict[str, Any]:
    """
    TOOL 3: Compara valores numéricos
    
    Args:
        value1: Primer valor
        value2: Segundo valor
    
    Returns:
        dict: Resultado de comparación
    """
    try:
        logger.info(f"Comparando {value1} vs {value2}")
        
        # Extraer números si son strings
        if isinstance(value1, str):
            value1 = float(re.findall(r"\d+\.?\d*", value1.replace(",", "."))[0])
        if isinstance(value2, str):
            value2 = float(re.findall(r"\d+\.?\d*", value2.replace(",", "."))[0])
        
        value1 = float(value1)
        value2 = float(value2)
        
        difference = value1 - value2
        percentage_diff = (difference / value2 * 100) if value2 != 0 else 0
        
        result = {
            "status": "success",
            "value1": value1,
            "value2": value2,
            "difference": difference,
            "percentage_difference": percentage_diff,
            "greater": value1 > value2,
            "equal": value1 == value2
        }
        
        logger.info(f"Comparación: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Error al comparar valores: {e}")
        return {"status": "error", "message": str(e)}


def extract_key_info(document: Dict[str, Any]) -> Dict[str, Any]:
    """
    TOOL 4: Extrae información clave de un documento
    
    Args:
        document: Documento recuperado
    
    Returns:
        dict: Información clave extraída
    """
    try:
        logger.info("Extrayendo información clave del documento")
        
        metadata = document.get("metadata", {})
        content = document.get("document", "")
        
        key_info = {
            "status": "success",
            "document_id": document.get("id"),
            "document_type": metadata.get("type", "unknown"),
            "source": metadata.get("fuente", ""),
            "relevance": 1.0 - document.get("distance", 0),
            "content_preview": content[:200] if content else "",
            "metadata": metadata
        }
        
        logger.info(f"Información clave extraída: {key_info}")
        return key_info
    
    except Exception as e:
        logger.error(f"Error al extraer información clave: {e}")
        return {"status": "error", "message": str(e)}


def search_by_metadata(filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    TOOL 5: Búsqueda adicional por metadata en ChromaDB
    
    Args:
        filters: Filtros de metadata (type, fuente, etc)
    
    Returns:
        dict: Resultados de búsqueda
    """
    try:
        logger.info(f"Buscando por metadata: {filters}")
        
        from src.vectorstore.multimodal_indexer import MultimodalIndexer
        
        indexer = MultimodalIndexer()
        results = indexer.search_by_metadata(filters)
        
        logger.info(f"Búsqueda por metadata completada")
        return results
    
    except Exception as e:
        logger.error(f"Error en búsqueda por metadata: {e}")
        return {"status": "error", "message": str(e)}


# Diccionario de herramientas disponibles
AVAILABLE_TOOLS = {
    "calculate_totals": calculate_totals,
    "validate_dates": validate_dates,
    "compare_values": compare_values,
    "extract_key_info": extract_key_info,
    "search_by_metadata": search_by_metadata
}


def get_tool(tool_name: str):
    """
    Obtiene una herramienta por nombre
    
    Args:
        tool_name: Nombre de la herramienta
    
    Returns:
        Función de herramienta o None
    """
    return AVAILABLE_TOOLS.get(tool_name)


def list_tools() -> List[str]:
    """
    Lista todas las herramientas disponibles
    
    Returns:
        list: Nombres de herramientas
    """
    return list(AVAILABLE_TOOLS.keys())