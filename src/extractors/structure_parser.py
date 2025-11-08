"""
structure_parser.py - PASO 4 de la Pipeline de Indexación (CORREGIDO)

Este módulo toma el MARKDOWN estructurado del PASO 3 (ocr_extractor)
y usa un LLM para convertir ese Markdown en un JSON validado por Pydantic.
"""

# --- Importaciones ---
import logging
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Type

from pydantic import BaseModel, Field, ValidationError

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from src.utils.logger import get_logger
from src.utils.config import (
    EXTRACTED_TEXT_DIR,
    EXTRACTED_TABLES_DIR,
    LLM_MODEL,
    LLM_TEMPERATURE
)

# Configuración del logger
logger = get_logger(__name__)


# --- 1. Definición del Esquema de Salida (Sin cambios) ---
# (Estos esquemas Pydantic están bien, no necesitan cambios)

class LiquidacionItem(BaseModel):
    """
    Representa un único ítem o línea de detalle dentro de la liquidación.
    """
    concepto: str = Field(description="Descripción del ítem o servicio.")
    cantidad: Optional[float] = Field(None, description="Cantidad del ítem. Puede ser nulo (Opcional).")
    precio_unitario: Optional[float] = Field(None, description="Precio por unidad del ítem. Puede ser nulo (Opcional).")
    total_item: float = Field(description="Monto total para este ítem (cantidad * precio_unitario).")

class LiquidacionSummary(BaseModel):
    """
    Representa el resumen financiero (los totales) al final del documento.
    """
    subtotal: Optional[float] = Field(None, description="Suma de todos los ítems antes de impuestos.")
    impuestos: Optional[float] = Field(None, description="Monto total de impuestos (ej. IGV, IVA).")
    total_general: float = Field(description="Monto final a pagar (subtotal + impuestos).")

class LiquidacionData(BaseModel):
    """
    El esquema raíz (principal) de nuestro documento de liquidación.
    """
    numero_factura: Optional[str] = Field(None, description="Número de factura o identificador del documento.")
    fecha_emision: Optional[str] = Field(None, description="Fecha en que se emitió el documento (formato YYYY-MM-DD).")
    cliente_nombre: Optional[str] = Field(None, description="Nombre del cliente o empresa.")
    items_detalle: List[LiquidacionItem] = Field(description="Una lista de todos los ítems detallados en el documento.")
    resumen_financiero: LiquidacionSummary = Field(description="Resumen de totales al final del documento.")


# --- 2. El Parser Estructurado (Prompt Corregido) ---

class StructureParser:
    """
    Toma texto en formato MARKDOWN (proveniente de ADE) y usa un LLM
    para "forzarlo" a que se ajuste a un esquema Pydantic.
    """
    
    def __init__(self, llm: BaseChatModel, schema: Type[BaseModel] = LiquidacionData):
        self.llm = llm
        self.target_schema = schema
        self.target_schema_json = json.dumps(schema.model_json_schema(), indent=2)
        logger.info(f"StructureParser inicializado con el esquema: {schema.__name__}")

    def _get_parsing_prompt(self) -> ChatPromptTemplate:
        """
        Crea el prompt del sistema que instruye al LLM.
        *** ESTA ES LA SECCIÓN CORREGIDA ***
        """
        # El prompt ahora le dice al LLM que espera Markdown, no texto crudo.
        system_prompt = f"""
        Eres un asistente experto en extracción de datos. Tu única tarea es
        convertir el siguiente documento, que está en formato **Markdown**,
        en un objeto JSON estructurado y válido.
        
        El Markdown puede contener tablas, listas y texto.
        
        Debes seguir *estrictamente* el siguiente esquema JSON. No inventes
        campos que no estén en el esquema. No omitas campos requeridos.
        
        Si no puedes encontrar un valor para un campo opcional, usa 'null'.
        Si no puedes encontrar un valor para un campo requerido (ej. 'total_general'),
        haz tu mejor esfuerzo para calcularlo o inferirlo del contexto.
        
        Limpia los datos: convierte "$1,234.56" a 1234.56.
        Normaliza las fechas a formato YYYY-MM-DD si es posible.
        
        ESQUEMA JSON OBJETIVO:
        {self.target_schema_json}
        
        El usuario te proporcionará el texto en **Markdown**.
        Responde *ÚNICAMENTE* con el objeto JSON válido.
        No incluyas explicaciones, saludos, o texto introductorio como '```json'.
        Tu respuesta debe ser un JSON que pueda ser parseado directamente.
        """
        
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessage(content="{ocr_text}") # La variable se sigue llamando 'ocr_text'
        ])

    def _extract_json_block(self, text: str) -> str:
        """
        Función de utilidad para limpiar la salida del LLM.
        (Sin cambios, esta función sigue siendo necesaria)
        """
        match = re.search(r'\{.*\}', text, re.DOTALL)
        
        if match:
            return match.group(0)
        else:
            logger.warning("No se encontró un bloque JSON en la respuesta del LLM. Devolviendo texto plano.")
            return text

    def parse_document(self, ocr_text: str) -> Dict[str, Any]:
        """
        Función principal de la clase. Parsea el texto (Markdown) y lo valida.
        (Sin cambios en la lógica, solo en el 'ocr_text' que ahora es Markdown)
        
        Args:
            ocr_text (str): El volcado de texto completo (en Markdown) 
                            proveniente de ocr_extractor.py.
            
        Returns:
            Dict[str, Any]: Un diccionario que se ajusta al esquema LiquidacionData.
        """
        logger.info("Iniciando parseo estructurado de texto (Markdown)...")
        
        prompt = self._get_parsing_prompt()
        chain = prompt | self.llm
        
        try:
            logger.debug("Invocando LLM para conversión Markdown -> JSON...")
            response = chain.invoke({"ocr_text": ocr_text})
            raw_content = response.content
            
            json_string = self._extract_json_block(raw_content)
            
            try:
                parsed_json = json.loads(json_string)
            except json.JSONDecodeError as e:
                logger.error(f"Fallo al decodificar JSON de la respuesta del LLM. Error: {e}")
                logger.debug(f"Respuesta LLM (raw): {raw_content}")
                raise ValueError(f"La respuesta del LLM no fue un JSON válido: {e}")

            try:
                validated_data = self.target_schema.model_validate(parsed_json)
                logger.info("Parseo y validación de esquema completados exitosamente.")
                return validated_data.model_dump() 
                
            except ValidationError as e:
                logger.error(f"La salida del LLM no validó contra el esquema Pydantic. Error: {e}")
                logger.debug(f"JSON del LLM (parseado): {parsed_json}")
                raise ValueError(f"Datos del LLM no válidos: {e}")

        except Exception as e:
            logger.error(f"Ocurrió un error inesperado durante el parseo: {e}", exc_info=True)
            raise


# --- 3. Función de Orquestación (Sin cambios) ---

def process_all_extracted_text() -> List[Dict[str, Any]]:
    """
    Esta es la función "orquestadora" para el PASO 4.
    (Esta función está bien, no necesita cambios).
    
    1. Inicializa el LLM y el Parser.
    2. Busca todos los archivos _text.json (que ahora contienen Markdown).
    3. Itera, lee el markdown de la clave "text" y usa el Parser.
    4. Guarda el JSON estructurado en `extracted_tables`.
    """
    logger.info("Iniciando PASO 4: Parseo y Estructuración (Markdown a JSON)...")
    
    try:
        llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
        parser = StructureParser(llm=llm, schema=LiquidacionData)
    except Exception as e:
        logger.error(f"No se pudo inicializar el LLM para el parser: {e}")
        logger.error("Asegúrate de que OPENAI_API_KEY esté en tu archivo .env")
        return [{"status": "error", "message": "Fallo al iniciar LLM"}]

    text_files = list(EXTRACTED_TEXT_DIR.glob("*_text.json"))
    logger.info(f"Se encontraron {len(text_files)} archivos de Markdown para parsear.")

    results = []
    
    for text_file_path in text_files:
        try:
            with open(text_file_path, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
            
            # El markdown está bajo la clave "text"
            ocr_text = ocr_data.get("text") 
            
            if not ocr_text:
                logger.warning(f"Archivo de entrada vacío (sin texto/markdown): {text_file_path.name}")
                results.append({"status": "skipped", "file": text_file_path.name, "message": "Texto vacío"})
                continue

            logger.debug(f"Parseando {text_file_path.name}...")
            structured_data = parser.parse_document(ocr_text)
            
            output_filename = f"{text_file_path.stem.replace('_text', '')}_structure.json"
            output_path = EXTRACTED_TABLES_DIR / output_filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Estructura JSON guardada en: {output_path.name}")
            results.append({
                "status": "success",
                "file": text_file_path.name,
                "output": output_path.name
            })

        except Exception as e:
            logger.error(f"Fallo al parsear el archivo {text_file_path.name}: {e}", exc_info=True)
            results.append({"status": "error", "file": text_file_path.name, "message": str(e)})

    logger.info(f"Parseo completado: {len(results)} archivos procesados.")
    return results