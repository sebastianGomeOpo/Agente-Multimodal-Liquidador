import logging
import re
import json
from datetime import datetime
from typing import List, Dict, Any, Union, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Importaciones de LangChain para el LLM y Pydantic
from langchain_core.language_models.chat_models import BaseChatModel

# Importación clave del proyecto
# Asumimos que el MultimodalIndexer está disponible en esta ruta
try:
    from ..vectorstore.multimodal_indexer import MultimodalIndexer
except ImportError:
    # Permitir importación en caso de ejecución de script (aunque menos probable)
    MultimodalIndexer = Any  

# Configuración del logger
logger = logging.getLogger(__name__)

# --- Clases de Pydantic para entradas de herramientas ---
# Usar Pydantic asegura que la data que el LLM pasa a la herramienta
# sea válida y esté bien estructurada.

class CalculateTotalsInput(BaseModel):
    """Esquema de entrada para la herramienta calculate_totals."""
    values: List[Union[str, float, int]] = Field(description="Lista de valores numéricos o strings de moneda para procesar.")

class CompareValuesInput(BaseModel):
    """Esquema de entrada para la herramienta compare_values."""
    value1: Union[str, float, int] = Field(description="Primer valor a comparar (numérico o string de moneda).")
    value2: Union[str, float, int] = Field(description="Segundo valor a comparar (numérico o string de moneda).")

class ExtractKeyInfoInput(BaseModel):
    """Esquema de entrada para la herramienta extract_key_info."""
    document_content: str = Field(description="El texto completo del cual extraer información.")
    keys_to_extract: List[str] = Field(description="Lista de claves o preguntas específicas a extraer del texto. Por ejemplo: ['Total de la factura', 'Fecha de vencimiento'].")

class SearchByMetadataInput(BaseModel):
    """
    Esquema de entrada para la herramienta search_by_metadata.
    Define los filtros a aplicar en ChromaDB.
    Ver: https://docs.trychroma.com/usage-guide#using-where-filters
    """
    filters: Dict[str, Any] = Field(description="Diccionario de filtros 'where' para ChromaDB. Ejemplo: {'type': 'excel_image', 'source': 'reporte.xlsx'}")


# --- Clase Contenedora de Herramientas ---

class AgentTools:
    """
    Un "ToolKit" que agrupa todas las herramientas disponibles para el agente.
    Recibe dependencias (como el indexador y el LLM) en su inicialización
    para que las herramientas puedan usarlas.
    """

    def __init__(self, indexer: 'MultimodalIndexer', llm: BaseChatModel):
        """
        Inicializa el kit de herramientas.
        
        Args:
            indexer (MultimodalIndexer): La instancia del indexador multimodal
                                         que conecta con ChromaDB.
            llm (BaseChatModel): El modelo de lenguaje (ej. ChatOpenAI) que se
                                 usará para herramientas que requieren IA,
                                 como la extracción.
        """
        self.indexer = indexer
        self.llm = llm
        # Acceso directo a la colección de Chroma para la herramienta de metadata
        self.collection = self.indexer.manager.get_collection()
        logger.info("AgentTools inicializado con indexer y LLM.")

    def get_all_tools(self) -> List[Any]:
        """
        Devuelve una lista de todas las herramientas (métodos decorados)
        para que el agente las pueda utilizar.
        """
        return [
            self.calculate_totals,
            self.validate_and_normalize_date,
            self.compare_values,
            self.extract_key_info,
            self.search_by_metadata,
        ]

    # --- Funciones de Ayuda (Helpers) ---

    def _parse_value(self, value: Union[str, float, int]) -> Optional[float]:
        """
        Función de ayuda robusta para convertir un string (posiblemente con
        símbolos de moneda, comas, etc.) en un float limpio.
        """
        if isinstance(value, (float, int)):
            return float(value)
        
        if not isinstance(value, str):
            return None

        try:
            # 1. Quitar símbolos de moneda ($, €, £, etc.) y espacios
            cleaned_value = re.sub(r'[$\s€£]', '', value)
            
            # 2. Manejar separadores de miles (comas o puntos)
            # Si hay comas y puntos, asumir que el último es decimal
            if ',' in cleaned_value and '.' in cleaned_value:
                # Asumir formato europeo (1.234,56)
                if cleaned_value.rfind('.') < cleaned_value.rfind(','):
                    cleaned_value = cleaned_value.replace('.', '').replace(',', '.')
                # Asumir formato americano (1,234.56)
                else:
                    cleaned_value = cleaned_value.replace(',', '')
            # Solo comas (asumir decimal europeo)
            elif ',' in cleaned_value:
                cleaned_value = cleaned_value.replace(',', '.')
            
            # 3. Convertir a float
            return float(cleaned_value)
        
        except (ValueError, TypeError):
            logger.warning(f"No se pudo parsear el valor: '{value}'")
            return None

    # --- Herramienta 1: Calculadora ---

    @tool("calculate_totals", args_schema=CalculateTotalsInput)
    def calculate_totals(self, values: List[Union[str, float, int]]) -> Dict[str, float]:
        """
        Calcula la suma total, el promedio, el mínimo y el máximo de una lista
        de valores numéricos. Los valores pueden ser números o strings
        (ej. "$1,234.50" o "50.00 €").
        """
        logger.debug(f"Tool 'calculate_totals' llamada con {len(values)} valores.")
        parsed_values = [self._parse_value(v) for v in values]
        # Filtrar solo los valores que se pudieron parsear (no son None)
        numeric_values = [v for v in parsed_values if v is not None]

        if not numeric_values:
            logger.warning("No se proporcionaron valores numéricos válidos.")
            return {"error": "No se proporcionaron valores numéricos válidos."}

        try:
            total = sum(numeric_values)
            count = len(numeric_values)
            average = total / count if count > 0 else 0
            minimum = min(numeric_values)
            maximum = max(numeric_values)

            result = {
                "total_sum": total,
                "count": count,
                "average": average,
                "min_value": minimum,
                "max_value": maximum
            }
            logger.info(f"Cálculo completado: {result}")
            return result
        
        except Exception as e:
            logger.error(f"Error en calculate_totals: {e}", exc_info=True)
            return {"error": str(e)}

    # --- Herramienta 2: Validador de Fechas ---

    @tool("validate_and_normalize_date")
    def validate_and_normalize_date(self, date_string: str, output_format: str = "%Y-%m-%d") -> Dict[str, str]:
        """
        Intenta validar un string de fecha contra múltiples formatos comunes
        (ej. 'dd/mm/yyyy', 'mm-dd-yyyy', 'yyyy-mm-dd').
        Si es válida, la devuelve normalizada al formato 'YYYY-MM-DD'.
        """
        logger.debug(f"Tool 'validate_and_normalize_date' llamada con: '{date_string}'")
        common_formats = [
            "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y",
            "%m-%d-%Y", "%Y/%m/%d", "%d %b %Y", "%d %B %Y"
        ]

        for fmt in common_formats:
            try:
                # Intenta parsear la fecha
                parsed_date = datetime.strptime(date_string, fmt)
                # Si tiene éxito, la formatea y la devuelve
                normalized_date = parsed_date.strftime(output_format)
                logger.info(f"Fecha '{date_string}' validada como '{normalized_date}'.")
                return {
                    "original": date_string,
                    "normalized": normalized_date,
                    "status": "valid"
                }
            except ValueError:
                # Si falla, prueba el siguiente formato
                continue
        
        # Si todos los formatos fallan
        logger.warning(f"No se pudo validar la fecha: '{date_string}'")
        return {
            "original": date_string,
            "status": "invalid",
            "error": "No se pudo parsear la fecha con formatos conocidos."
        }

    # --- Herramienta 3: Comparador de Valores ---

    @tool("compare_values", args_schema=CompareValuesInput)
    def compare_values(self, value1: Union[str, float, int], value2: Union[str, float, int]) -> Dict[str, Any]:
        """
        Compara dos valores numéricos (incluso si están como texto con formato
        de moneda) y devuelve su diferencia y relación (mayor, menor, igual).
        """
        logger.debug(f"Tool 'compare_values' llamada con: '{value1}' y '{value2}'")
        v1 = self._parse_value(value1)
        v2 = self._parse_value(value2)

        if v1 is None or v2 is None:
            logger.warning("No se pudo comparar. Uno o ambos valores son inválidos.")
            return {"error": "Uno o ambos valores no pudieron ser parseados como números."}

        difference = v1 - v2
        
        if v1 > v2:
            relation = "value1_is_greater"
        elif v1 < v2:
            relation = "value1_is_lesser"
        else:
            relation = "values_are_equal"

        result = {
            "value1_numeric": v1,
            "value2_numeric": v2,
            "difference": difference,
            "relation": relation
        }
        logger.info(f"Comparación completada: {result}")
        return result

    # --- Herramienta 4: Extracción de Información (con LLM) ---

    @tool("extract_key_info", args_schema=ExtractKeyInfoInput)
    def extract_key_info(self, document_content: str, keys_to_extract: List[str]) -> Dict[str, Any]:
        """
        Extrae información clave y estructurada de un bloque de texto usando un LLM.
        El agente debe especificar qué claves (preguntas) quiere extraer.
        Devuelve un JSON con las claves y los valores encontrados.
        """
        logger.debug(f"Tool 'extract_key_info' llamada para {len(keys_to_extract)} claves.")
        
        # Crear el prompt para el LLM
        # Usamos formato JSON para una salida más fiable
        prompt_keys = "\n".join([f'  "{key}": "extraer valor para {key}"' for key in keys_to_extract])
        
        prompt = f"""
        Dada la siguiente lista de claves y un documento de texto, extrae
        los valores correspondientes del texto.
        
        Responde *únicamente* con un objeto JSON válido que siga este formato:
        {{
        {prompt_keys}
        }}
        
        Si no puedes encontrar un valor para una clave, usa 'null' como valor.
        
        DOCUMENTO DE TEXTO:
        ---
        {document_content[:4000]} 
        ---
        
        JSON DE SALIDA:
        """
        
        try:
            # Usar el LLM proporcionado en la inicialización
            response = self.llm.invoke(prompt)
            content = response.content
            
            # Limpiar la respuesta del LLM (a veces añaden ```json)
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                logger.error("El LLM no devolvió un JSON válido.")
                return {"error": "El LLM no devolvió un JSON válido."}

            json_string = json_match.group(0)
            result = json.loads(json_string)
            logger.info("Extracción con LLM completada.")
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Error al decodificar JSON de LLM: {e}. Respuesta: {content}")
            return {"error": "Error al decodificar la respuesta del LLM."}
        except Exception as e:
            logger.error(f"Error en extract_key_info: {e}", exc_info=True)
            return {"error": str(e)}

    # --- Herramienta 5: Búsqueda por Metadata (en ChromaDB) ---

    @tool("search_by_metadata", args_schema=SearchByMetadataInput)
    def search_by_metadata(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Busca documentos en la base de datos vectorial (ChromaDB) usando
        *únicamente* filtros de metadatos (búsqueda 'where'), sin
        búsqueda semántica. Es útil para encontrar documentos
        específicos por su nombre, tipo o fecha.
        """
        logger.debug(f"Tool 'search_by_metadata' llamada con filtros: {filters}")
        if not self.collection:
            logger.error("ChromaDB collection no está disponible.")
            return {"error": "ChromaDB collection no está disponible."}

        try:
            # Usamos el método 'get' de ChromaDB para filtrar por metadata
            # 'include' especifica qué campos queremos de vuelta
            results = self.collection.get(
                where=filters,
                include=["metadatas", "documents"] # No necesitamos 'embeddings'
            )
            
            count = len(results.get('ids', []))
            logger.info(f"Búsqueda por metadata encontró {count} resultados.")
            
            if count == 0:
                return {"message": "No se encontraron documentos con esos filtros.", "results": []}

            # Formatear la salida para que sea más limpia
            formatted_results = [
                {
                    "id": id,
                    "metadata": meta,
                    "document": doc
                }
                for id, meta, doc in zip(results['ids'], results['metadatas'], results['documents'])
            ]
            return {"count": count, "results": formatted_results}
        
        except Exception as e:
            logger.error(f"Error en search_by_metadata: {e}", exc_info=True)
            # Esto puede pasar si la sintaxis del filtro es incorrecta
            return {"error": f"Error al consultar ChromaDB: {e}"}