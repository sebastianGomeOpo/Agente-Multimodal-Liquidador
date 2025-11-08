"""
PASO 4: Extracción estructurada usando Landing AI ADE Extract API
Convierte Markdown OCR → JSON estructurado (OperacionNave)

CAMBIO CLAVE: Ahora usa ADE Extract API en lugar de un LLM genérico
"""

import json
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from src.utils.logger import get_logger
from src.utils.config import (
    EXTRACTED_TEXT_DIR,
    EXTRACTED_TABLES_DIR,
    LANDING_AI_API_KEY
)

logger = get_logger(__name__)


# ============================================================
# ESQUEMA JSON PARA ADE EXTRACT
# ============================================================

ADE_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "recalada": {
            "type": ["string", "null"],
            "description": "Código único de la recalada (ej: 10841, 25105)"
        },
        "nave_nombre": {
            "type": ["string", "null"],
            "description": "Nombre completo de la nave (ej: SKY KNIGHT, MV OCEAN SPIRIT)"
        },
        "imo": {
            "type": ["string", "null"],
            "description": "Número IMO de la nave (ej: 9561942)"
        },
        "bandera": {
            "type": ["string", "null"],
            "description": "País de bandera de la nave"
        },
        "puerto_origen": {
            "type": ["string", "null"],
            "description": "Puerto de origen de la nave"
        },
        "puerto_destino": {
            "type": ["string", "null"],
            "description": "Puerto de destino de la nave"
        },
        "muelle": {
            "type": ["string", "null"],
            "description": "Muelle donde atracó la nave (ej: F, A, B)"
        },
        "fecha_inicio_operacion": {
            "type": ["string", "null"],
            "description": "Fecha y hora de inicio de operación (formato ISO o cualquier formato encontrado)"
        },
        "fecha_fin_operacion": {
            "type": ["string", "null"],
            "description": "Fecha y hora de término de operación"
        },
        "fecha_arribo": {
            "type": ["string", "null"],
            "description": "Fecha y hora de arribo al puerto"
        },
        "fecha_atraque": {
            "type": ["string", "null"],
            "description": "Fecha y hora de atraque"
        },
        "fecha_desatraque": {
            "type": ["string", "null"],
            "description": "Fecha y hora de desatraque"
        },
        "producto": {
            "type": ["string", "null"],
            "description": "Tipo de carga o producto (ej: CONCENTRADO DE COBRE, CLINKER)"
        },
        "tonelaje_total": {
            "type": ["number", "null"],
            "description": "Tonelaje total de la operación (número sin comas)"
        },
        "regimen": {
            "type": ["string", "null"],
            "description": "Régimen de operación (ej: EXPORTACION, IMPORTACION)"
        },
        "tipo_operacion": {
            "type": ["string", "null"],
            "description": "Tipo de operación (ej: EMBARQUE, DESCARGA, EMBARQUE INDIRECTO)"
        },
        "agente_maritimo": {
            "type": ["string", "null"],
            "description": "Nombre del agente marítimo"
        },
        "agente_estiba": {
            "type": ["string", "null"],
            "description": "Nombre del agente de estiba"
        },
        "horas_operacion": {
            "type": ["number", "null"],
            "description": "Horas totales de operación (número decimal)"
        },
        "horas_amarradero": {
            "type": ["number", "null"],
            "description": "Horas totales en amarradero (número decimal)"
        },
        "rendimiento_tn_hr": {
            "type": ["number", "null"],
            "description": "Rendimiento en toneladas por hora"
        },
        "clientes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Lista de clientes involucrados en la operación"
        },
        "bodegas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "bodega": {
                        "type": "string",
                        "description": "Identificador de bodega (ej: N°1, N°2, HOLD 1)"
                    },
                    "operacion": {
                        "type": ["string", "null"],
                        "description": "Tipo de operación en esta bodega (ej: EMBARQUE, DESCARGA)"
                    },
                    "tonelaje": {
                        "type": ["number", "null"],
                        "description": "Tonelaje movido en esta bodega (número sin comas)"
                    },
                    "lotes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Códigos de lote asociados a esta bodega"
                    }
                }
            },
            "description": "Lista de bodegas con sus operaciones"
        },
        "lotes_facturacion": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "lote": {
                        "type": "string",
                        "description": "Código del lote"
                    },
                    "cliente": {
                        "type": ["string", "null"],
                        "description": "Cliente asociado al lote"
                    },
                    "codigos_facturacion": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Códigos de facturación asociados"
                    }
                }
            },
            "description": "Lotes con información de facturación"
        }
    },
    "required": []  # Ningún campo es obligatorio, extraer lo que se pueda
}


# ============================================================
# CLIENTE ADE EXTRACT
# ============================================================

class ADEExtractClient:
    """
    Cliente para interactuar con la API de Landing AI ADE Extract.
    """
    
    def __init__(self, api_key: str):
        """
        Inicializa el cliente con la API key.
        
        Args:
            api_key: API key de Landing AI
        """
        self.api_key = api_key
        self.base_url = "https://api.va.landing.ai/v1/ade"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def extract_from_markdown(
        self,
        markdown: str,
        schema: Dict[str, Any],
        model: str = "extract-latest"
    ) -> Dict[str, Any]:
        """
        Extrae información estructurada de un Markdown usando ADE Extract.
        
        Args:
            markdown: Contenido Markdown a procesar
            schema: Esquema JSON para la extracción
            model: Modelo a usar (default: extract-latest)
        
        Returns:
            Dict con los campos extraídos
        """
        url = f"{self.base_url}/extract"
        
        # Preparar los datos del request
        files = {
            'markdown': ('document.md', markdown, 'text/markdown'),
            'schema': (None, json.dumps(schema), 'application/json'),
            'model': (None, model)
        }
        
        try:
            response = requests.post(
                url,
                headers=self.headers,
                files=files,
                timeout=60  # 60 segundos timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('extraction', {})
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error en ADE Extract API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            return {}


# ============================================================
# PROCESADOR DE ESTRUCTURA
# ============================================================

@dataclass
class ParserResult:
    """Resultado del procesamiento de un archivo."""
    source_file: str
    success: bool
    output_file: Optional[str] = None
    error: Optional[str] = None


class StructureParser:
    """
    Parser que convierte Markdown (del ADE Parse) en JSON estructurado
    usando ADE Extract API.
    """
    
    def __init__(self, api_key: str = LANDING_AI_API_KEY):
        """
        Inicializa el parser con el cliente ADE Extract.
        
        Args:
            api_key: API key de Landing AI
        """
        self.client = ADEExtractClient(api_key)
        self.schema = ADE_EXTRACT_SCHEMA
        logger.info(f"StructureParser inicializado con ADE Extract API")
    
    def parse_markdown_file(self, markdown_path: Path) -> Optional[Dict[str, Any]]:
        """
        Parsea un archivo Markdown y extrae la estructura.
        
        Args:
            markdown_path: Ruta al archivo Markdown (_text.json)
        
        Returns:
            Dict con los datos estructurados o None si falla
        """
        try:
            # Leer el archivo JSON que contiene el Markdown
            with open(markdown_path, 'r', encoding='utf-8') as f:
                markdown_data = json.load(f)
            
            markdown_text = markdown_data.get('text', '')
            
            if not markdown_text or len(markdown_text.strip()) < 10:
                logger.warning(f"Markdown vacío o muy corto en {markdown_path.name}")
                return None
            
            # Llamar a ADE Extract
            logger.info(f"Iniciando parseo estructurado con ADE Extract...")
            extracted_data = self.client.extract_from_markdown(
                markdown=markdown_text,
                schema=self.schema,
                model="extract-latest"
            )
            
            if not extracted_data:
                logger.warning(f"ADE Extract no retornó datos para {markdown_path.name}")
                return None
            
            logger.info(f"Parseo y extracción completados.")
            return extracted_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Error al leer JSON de {markdown_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error inesperado al parsear {markdown_path}: {e}")
            return None
    
    def save_structured_json(
        self,
        data: Dict[str, Any],
        output_path: Path
    ) -> bool:
        """
        Guarda los datos estructurados en un archivo JSON.
        
        Args:
            data: Datos estructurados
            output_path: Ruta de salida
        
        Returns:
            True si se guardó exitosamente
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"JSON (OperacionNave) guardado en: {output_path.name}")
            return True
        except Exception as e:
            logger.error(f"Error al guardar JSON en {output_path}: {e}")
            return False


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def process_all_extracted_text() -> Dict[str, Any]:
    """
    Procesa todos los archivos Markdown extraídos y genera JSONs estructurados.
    
    Returns:
        Dict con estadísticas del procesamiento
    """
    logger.info("Iniciando PASO 4: Extracción enfocada (Markdown → OperacionNave JSON)...")
    
    # Crear directorio de salida si no existe
    EXTRACTED_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Buscar todos los archivos _text.json
    text_files = list(EXTRACTED_TEXT_DIR.glob("*_text.json"))
    
    if not text_files:
        logger.warning(f"No se encontraron archivos de texto en {EXTRACTED_TEXT_DIR}")
        return {
            "total_files": 0,
            "success": 0,
            "failed": 0,
            "results": []
        }
    
    logger.info(f"Se encontraron {len(text_files)} archivos de Markdown para parsear.")
    
    # Inicializar parser
    parser = StructureParser()
    
    results = []
    success_count = 0
    failed_count = 0
    
    # Procesar cada archivo
    for text_file in text_files:
        # Nombre de salida: reemplazar _text.json por _structure.json
        output_name = text_file.stem.replace('_text', '_structure') + '.json'
        output_path = EXTRACTED_TABLES_DIR / output_name
        
        # Parsear con ADE Extract
        structured_data = parser.parse_markdown_file(text_file)
        
        if structured_data is None:
            failed_count += 1
            results.append(ParserResult(
                source_file=text_file.name,
                success=False,
                error="No se pudo extraer información"
            ))
            continue
        
        # Guardar JSON estructurado
        if parser.save_structured_json(structured_data, output_path):
            success_count += 1
            results.append(ParserResult(
                source_file=text_file.name,
                success=True,
                output_file=output_name
            ))
        else:
            failed_count += 1
            results.append(ParserResult(
                source_file=text_file.name,
                success=False,
                error="Error al guardar JSON"
            ))
    
    logger.info(f"Parseo completado: {len(text_files)} archivos procesados.")
    
    return {
        "total_files": len(text_files),
        "success": success_count,
        "failed": failed_count,
        "results": results
    }


# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    result = process_all_extracted_text()
    print(f"\nProcesamiento completado:")
    print(f"  Total: {result['total_files']}")
    print(f"  Exitosos: {result['success']}")
    print(f"  Fallidos: {result['failed']}")