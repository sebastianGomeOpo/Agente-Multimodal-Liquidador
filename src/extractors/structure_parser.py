"""
structure_parser.py - PASO 4 de la Pipeline de Indexación

Este módulo es uno de los más importantes del proyecto. Su trabajo es
tomar el texto en bruto (caótico y no estructurado) que nos dio el OCR
y convertirlo en un formato JSON limpio, validado y estructurado.

Utiliza un LLM (como GPT-4) junto con Pydantic para "forzar" al
texto a que se ajuste a un esquema que definamos.
"""

# --- Importaciones ---
import logging  # Para registrar eventos y errores
import json     # Para trabajar con archivos JSON (leer y escribir)
import re       # Para Expresiones Regulares (usado para limpiar la salida del LLM)
from pathlib import Path  # Para manejar rutas de archivos de forma moderna
from typing import List, Dict, Any, Optional, Type  # Para "type hints" (ayuda a saber qué tipo de dato es cada variable)

# Importaciones de Pydantic: la clave para la validación de datos
# BaseModel: La clase base de la que heredan nuestros esquemas
# Field: Para añadir descripciones a nuestros campos (muy útil para el LLM)
# ValidationError: La excepción que se lanza si los datos no coinciden con el esquema
from pydantic import BaseModel, Field, ValidationError

# Importaciones de LangChain
from langchain_core.language_models.chat_models import BaseChatModel # Una "interfaz" genérica para cualquier modelo de chat
from langchain_core.prompts import ChatPromptTemplate               # Para construir nuestro prompt
from langchain_core.messages import SystemMessage, HumanMessage     # Tipos de mensajes para el chat
from langchain_openai import ChatOpenAI                             # La implementación concreta de un LLM de OpenAI (¡NUEVO!)

# Importaciones de nuestro propio proyecto
from src.utils.logger import get_logger
from src.utils.config import (
    EXTRACTED_TEXT_DIR,     # Dónde leer los textos del OCR
    EXTRACTED_TABLES_DIR,   # Dónde guardar los JSON estructurados (lo usamos como destino)
    LLM_MODEL,              # Qué modelo LLM usar (ej. "gpt-4")
    LLM_TEMPERATURE         # Qué tan creativo debe ser el LLM (0 = determinista)
)

# Configuración del logger para este archivo
# Usamos get_logger de nuestros utils para mantener un formato consistente
logger = get_logger(__name__)


# --- 1. Definición del Esquema de Salida (Nuestro "Contrato de Datos") ---
# Aquí definimos la "plantilla" exacta de cómo queremos que sean nuestros datos.
# Pydantic usará esto para validar la salida del LLM.
# Usar `Field(description=...)` es crucial, ya que el LLM leerá estas
# descripciones para entender qué dato poner en cada campo.

class LiquidacionItem(BaseModel):
    """
    Representa un único ítem o línea de detalle dentro de la liquidación.
    Es un "sub-esquema" que usaremos dentro del esquema principal.
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
    Esta es la estructura final que queremos obtener.
    """
    numero_factura: Optional[str] = Field(None, description="Número de factura o identificador del documento.")
    fecha_emision: Optional[str] = Field(None, description="Fecha en que se emitió el documento (formato YYYY-MM-DD).")
    cliente_nombre: Optional[str] = Field(None, description="Nombre del cliente o empresa.")
    items_detalle: List[LiquidacionItem] = Field(description="Una lista de todos los ítems detallados en el documento.")
    resumen_financiero: LiquidacionSummary = Field(description="Resumen de totales al final del documento.")


# --- 2. El Parser Estructurado (basado en LLM) ---

class StructureParser:
    """
    Esta clase es un "micro-servicio" de parseo.
    Toma texto en bruto de un OCR y usa un LLM para "forzarlo"
    a que se ajuste a un esquema Pydantic (como LiquidacionData).
    """
    
    def __init__(self, llm: BaseChatModel, schema: Type[BaseModel] = LiquidacionData):
        """
        Inicializa el parser.
        
        Args:
            llm (BaseChatModel): La instancia del LLM (ej. ChatOpenAI)
                                 que se usará para el parseo.
            schema (Type[BaseModel]): El esquema Pydantic (nuestro "contrato") al que
                                      se deben ajustar los datos.
        """
        self.llm = llm
        self.target_schema = schema
        
        # Convertimos nuestro esquema Pydantic a un formato JSON Schema
        # Esto es lo que le pasaremos al LLM en el prompt.
        self.target_schema_json = json.dumps(schema.model_json_schema(), indent=2)
        
        logger.info(f"StructureParser inicializado con el esquema: {schema.__name__}")

    def _get_parsing_prompt(self) -> ChatPromptTemplate:
        """
        Crea el prompt del sistema que instruye al LLM.
        Esta es la parte más importante (Prompt Engineering).
        """
        # Este es un "System Prompt". Define la "personalidad" y las reglas
        # que el LLM debe seguir en todo momento.
        system_prompt = f"""
        Eres un asistente experto en extracción de datos. Tu única tarea es
        convertir el texto no estructurado de un OCR en un objeto JSON
        estructurado y válido.
        
        Debes seguir *estrictamente* el siguiente esquema JSON. No inventes
        campos que no estén en el esquema. No omitas campos requeridos.
        
        Si no puedes encontrar un valor para un campo opcional, usa 'null'.
        Si no puedes encontrar un valor para un campo requerido (ej. 'total_general'),
        haz tu mejor esfuerzo para calcularlo o inferirlo del contexto.
        
        Limpia los datos: convierte "$1,234.56" a 1234.56.
        Normaliza las fechas a formato YYYY-MM-DD si es posible.
        
        ESQUEMA JSON OBJETIVO:
        {self.target_schema_json}
        
        El usuario te proporcionará el texto del OCR.
        Responde *ÚNICAMENTE* con el objeto JSON válido.
        No incluyas explicaciones, saludos, o texto introductorio como '```json'.
        Tu respuesta debe ser un JSON que pueda ser parseado directamente.
        """
        
        # Creamos una plantilla de chat.
        # 1. El SystemMessage (las reglas de arriba)
        # 2. El HumanMessage (el texto del OCR que le pasará el usuario)
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessage(content="{ocr_text}")  # "{ocr_text}" es una variable que llenaremos después
        ])

    def _extract_json_block(self, text: str) -> str:
        """
        Función de utilidad para limpiar la salida del LLM.
        A veces, el LLM responde con: "Claro, aquí tienes el JSON: ```json\n{...}\n```"
        Esta función usa una expresión regular (regex) para extraer
        únicamente el bloque JSON (de { a }).
        """
        # Busca el primer '{' y el último '}' que lo engloba todo.
        # re.DOTALL significa que el '.' también incluye saltos de línea.
        match = re.search(r'\{.*\}', text, re.DOTALL)
        
        if match:
            # Si lo encuentra, devuelve solo el JSON
            return match.group(0)
        else:
            # Si no encuentra un JSON, devuelve el texto tal cual (probablemente falle después)
            logger.warning("No se encontró un bloque JSON en la respuesta del LLM. Devolviendo texto plano.")
            return text

    def parse_document(self, ocr_text: str) -> Dict[str, Any]:
        """
        Función principal de la clase. Parsea el texto y lo valida.
        
        Args:
            ocr_text (str): El volcado de texto completo (en bruto) 
                            proveniente del ocr_extractor.py.
            
        Returns:
            Dict[str, Any]: Un diccionario que se ajusta al esquema LiquidacionData.
            
        Raises:
            ValueError: Si la salida del LLM no puede ser parseada como JSON
                        o si no valida contra el esquema Pydantic.
        """
        logger.info("Iniciando parseo estructurado de texto OCR...")
        
        # 1. Obtener la plantilla del prompt
        prompt = self._get_parsing_prompt()
        
        # 2. Crear la "cadena" (chain) de LangChain (LCEL)
        # Esto significa: "primero pasa por el prompt, y su salida pásala al llm"
        chain = prompt | self.llm
        
        try:
            # 3. Llamar al LLM con el texto del OCR
            logger.debug("Invocando LLM para parseo...")
            response = chain.invoke({"ocr_text": ocr_text})
            raw_content = response.content
            
            # 4. Limpiar y extraer el bloque JSON de la respuesta
            json_string = self._extract_json_block(raw_content)
            
            # 5. Parsear el string JSON a un diccionario Python
            try:
                parsed_json = json.loads(json_string)
            except json.JSONDecodeError as e:
                logger.error(f"Fallo al decodificar JSON de la respuesta del LLM. Error: {e}")
                logger.debug(f"Respuesta LLM (raw): {raw_content}")
                raise ValueError(f"La respuesta del LLM no fue un JSON válido: {e}")

            # 6. ¡Validar el diccionario contra nuestro esquema Pydantic!
            # Este es el paso de garantía de calidad.
            try:
                validated_data = self.target_schema.model_validate(parsed_json)
                logger.info("Parseo y validación de esquema completados exitosamente.")
                
                # Devolvemos como un diccionario estándar (no como un objeto Pydantic)
                return validated_data.model_dump() 
                
            except ValidationError as e:
                logger.error(f"La salida del LLM no validó contra el esquema Pydantic. Error: {e}")
                logger.debug(f"JSON del LLM (parseado): {parsed_json}")
                raise ValueError(f"Datos del LLM no válidos: {e}")

        except Exception as e:
            logger.error(f"Ocurrió un error inesperado durante el parseo: {e}", exc_info=True)
            raise


# --- 3. Función de Orquestación (ESTA ES LA FUNCIÓN QUE FALTABA) ---

def process_all_extracted_text() -> List[Dict[str, Any]]:
    """
    Esta es la función "orquestadora" que faltaba.
    
    1. Inicializa el LLM y el Parser.
    2. Busca todos los archivos de texto generados por el OCR.
    3. Itera sobre cada archivo, lo lee y usa el Parser para extraer la estructura.
    4. Guarda el nuevo JSON estructurado en la carpeta `extracted_tables`.
    """
    logger.info("Iniciando PASO 4: Parseo y Estructuración de Texto...")
    
    # 1. Inicializar el LLM y el Parser
    # Usamos un bloque try/except porque esto puede fallar si la OPENAI_API_KEY
    # no está configurada en el archivo .env
    try:
        # Usamos el modelo y temperatura definidos en config.py
        llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
        parser = StructureParser(llm=llm, schema=LiquidacionData)
    except Exception as e:
        logger.error(f"No se pudo inicializar el LLM para el parser: {e}")
        logger.error("Asegúrate de que OPENAI_API_KEY esté en tu archivo .env")
        return [{"status": "error", "message": "Fallo al iniciar LLM"}]

    # 2. Encontrar todos los archivos de texto de OCR (terminados en _text.json)
    # Usamos pathlib.glob para encontrar patrones de archivos
    text_files = list(EXTRACTED_TEXT_DIR.glob("*_text.json"))
    logger.info(f"Se encontraron {len(text_files)} archivos de texto para parsear.")

    results = []
    
    # 3. Iterar sobre cada archivo de texto
    for text_file_path in text_files:
        
        # Usamos un try/except dentro del bucle para que, si un archivo
        # falla, el proceso continúe con los demás.
        try:
            # 4. Leer el texto crudo del archivo JSON del OCR
            with open(text_file_path, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
            
            ocr_text = ocr_data.get("text") # El texto está bajo la clave "text"
            
            if not ocr_text:
                logger.warning(f"Archivo OCR vacío (sin texto): {text_file_path.name}")
                results.append({"status": "skipped", "file": text_file_path.name, "message": "Texto vacío"})
                continue

            # 5. Usar el parser para convertir el texto en un JSON estructurado
            logger.debug(f"Parseando {text_file_path.name}...")
            structured_data = parser.parse_document(ocr_text)
            
            # 6. Guardar la estructura JSON resultante
            # El nombre será el mismo, pero terminado en _structure.json
            output_filename = f"{text_file_path.stem.replace('_text', '')}_structure.json"
            output_path = EXTRACTED_TABLES_DIR / output_filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                # guardamos el JSON con formato bonito (indent=2)
                json.dump(structured_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Estructura guardada en: {output_path.name}")
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


# --- 4. Bloque de Prueba (si se ejecuta este archivo directamente) ---

if __name__ == "__main__":
    """
    Esto solo se ejecuta si corres el archivo directamente (ej. `python src/extractors/structure_parser.py`)
    Es una "prueba de humo" para ver si la clase StructureParser funciona.
    """
    logging.basicConfig(level=logging.INFO) # Configuración básica de logging para la prueba
    logger.info("Probando StructureParser con un LLM 'mock' (simulado)...")
    
    # --- Mocking (Simulación) de un LLM ---
    # Creamos una clase falsa que "finge" ser un LLM de LangChain.
    # No hace una llamada real a la API, solo devuelve un JSON de prueba.
    class MockChatModel(BaseChatModel):
        """Mock LLM para pruebas. Devuelve una respuesta JSON predefinida."""
        def invoke(self, prompt_messages: Dict[str, str], **kwargs) -> Any:
            
            # La respuesta JSON que simulamos que el LLM genera
            mock_json_response = """
            {
                "numero_factura": "FAC-001",
                "fecha_emision": "2024-10-25",
                "cliente_nombre": "Empresa Ejemplo S.A.",
                "items_detalle": [
                    {
                        "concepto": "Servicio de Consultoría",
                        "cantidad": 1.0,
                        "precio_unitario": 1500.0,
                        "total_item": 1500.0
                    },
                    {
                        "concepto": "Ajuste de Proyecto",
                        "cantidad": null,
                        "precio_unitario": null,
                        "total_item": -50.25
                    }
                ],
                "resumen_financiero": {
                    "subtotal": 1449.75,
                    "impuestos": 217.46,
                    "total_general": 1667.21
                }
            }
            """
            
            # Simulamos que el LLM añade texto extra (que nuestro parser debe limpiar)
            mock_llm_output = f"Claro, aquí tienes el JSON:\n```json\n{mock_json_response}\n```"
            
            # Simulamos el objeto de respuesta de LangChain
            class MockResponse:
                content = mock_llm_output
            
            return MockResponse()
        
        # Estas funciones son requeridas por la clase BaseChatModel
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            pass
        
        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            pass
        
        @property
        def _llm_type(self) -> str:
            return "mock"

    # --- Fin del Mocking ---

    # 1. Texto de OCR de ejemplo (esto vendría de ocr_extractor.py)
    sample_ocr_text = """
    LIQUIDACIÓN DE SERVICIOS
    Factura: FAC-001         Fecha: 25/10/2024
    Cliente: Empresa Ejemplo S.A.
    
    Concepto                 Total
    Servicio de Consultoría  $ 1,500.00
    Ajuste de Proyecto       $ -50.25
    
    Subtotal: 1,449.75
    IVA (15%): 217.46
    TOTAL A PAGAR: $ 1,667.21
    """

    # 2. Inicializar el parser con el LLM mockeado
    mock_llm = MockChatModel()
    parser = StructureParser(llm=mock_llm, schema=LiquidacionData)
    
    # 3. Ejecutar el parseo
    try:
        structured_data = parser.parse_document(sample_ocr_text)
        
        print("\n--- ¡Parseo Exitoso! ---")
        # ensure_ascii=False permite que se muestren tildes y 'ñ' correctamente
        print(json.dumps(structured_data, indent=2, ensure_ascii=False))
        
        print("\n--- Acceso a datos (como un diccionario) ---")
        print(f"Cliente: {structured_data['cliente_nombre']}")
        print(f"Total General: {structured_data['resumen_financiero']['total_general']}")
        
    except (ValueError, ValidationError) as e:
        print(f"\n--- ¡Error en el Parseo! ---")
        print(e)