import logging
import json
import re
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel, Field, ValidationError

# Dependencia de LangChain para el LLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# Configuración del logger
logger = logging.getLogger(__name__)

# --- 1. Definición del Esquema de Salida (Nuestro "Contrato de Datos") ---
# Definimos la estructura de negocio exacta que queremos.
# Esto es lo que el resto de nuestra aplicación usará.

class LiquidacionItem(BaseModel):
    """Representa un único ítem dentro de la liquidación."""
    concepto: str = Field(description="Descripción del ítem o servicio.")
    cantidad: Optional[float] = Field(None, description="Cantidad del ítem.")
    precio_unitario: Optional[float] = Field(None, description="Precio por unidad del ítem.")
    total_item: float = Field(description="Monto total para este ítem (cantidad * precio_unitario).")

class LiquidacionSummary(BaseModel):
    """Representa el resumen financiero al final del documento."""
    subtotal: Optional[float] = Field(None, description="Suma de todos los ítems antes de impuestos.")
    impuestos: Optional[float] = Field(None, description="Monto total de impuestos (ej. IVA).")
    total_general: float = Field(description="Monto final a pagar (subtotal + impuestos).")

class LiquidacionData(BaseModel):
    """
    El esquema raíz de nuestro documento de liquidación.
    Esto es lo que el parser *debe* producir.
    """
    numero_factura: Optional[str] = Field(None, description="Número de factura o identificador del documento.")
    fecha_emision: Optional[str] = Field(None, description="Fecha en que se emitió el documento (formato YYYY-MM-DD).")
    cliente_nombre: Optional[str] = Field(None, description="Nombre del cliente o empresa.")
    items_detalle: List[LiquidacionItem] = Field(description="Lista de todos los ítems detallados en el documento.")
    resumen_financiero: LiquidacionSummary = Field(description="Resumen de totales al final del documento.")


# --- 2. El Parser Estructurado (basado en LLM) ---

class StructureParser:
    """
    Toma texto en bruto de un OCR y usa un LLM para forzarlo
    a un esquema Pydantic (LiquidacionData).
    
    Este es el "micro-servicio" de parseo.
    """
    
    def __init__(self, llm: BaseChatModel, schema: Type[BaseModel] = LiquidacionData):
        """
        Inicializa el parser con un modelo de lenguaje.
        
        Args:
            llm (BaseChatModel): La instancia del LLM (ej. ChatOpenAI, ChatAnthropic)
                                 que se usará para el parseo.
            schema (Type[BaseModel]): El esquema Pydantic al que se deben
                                      ajustar los datos.
        """
        self.llm = llm
        self.target_schema = schema
        self.target_schema_json = json.dumps(schema.model_json_schema(), indent=2)
        logger.info(f"StructureParser inicializado con el esquema: {schema.__name__}")

    def _get_parsing_prompt(self) -> ChatPromptTemplate:
        """
        Crea el prompt del sistema que instruye al LLM.
        Esta es la parte más importante.
        """
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
        Responde *únicamente* con el objeto JSON válido.
        No incluyas explicaciones, saludos, o texto introductorio como '```json'.
        Tu respuesta debe ser un JSON que pueda ser parseado directamente.
        """
        
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessage(content="{ocr_text}")
        ])

    def _extract_json_block(self, text: str) -> str:
        """
        Extrae robustamente un bloque JSON del texto, incluso si el LLM
        añadió '```json' o explicaciones.
        """
        # Expresión regular para encontrar un bloque JSON
        # Busca desde el primer '{' hasta el último '}'
        match = re.search(r'\{.*\}', text, re.DOTALL)
        
        if match:
            return match.group(0)
        else:
            logger.warning("No se encontró un bloque JSON en la respuesta del LLM. Devolviendo texto plano.")
            return text

    def parse_document(self, ocr_text: str) -> Dict[str, Any]:
        """
        Función principal. Parsea el texto de un OCR y lo valida contra el esquema.
        
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
        
        prompt = self._get_parsing_prompt()
        chain = prompt | self.llm
        
        try:
            # 1. Llamar al LLM
            response = chain.invoke({"ocr_text": ocr_text})
            raw_content = response.content
            
            # 2. Limpiar y extraer el bloque JSON
            json_string = self._extract_json_block(raw_content)
            
            # 3. Parsear el string a un diccionario Python
            try:
                parsed_json = json.loads(json_string)
            except json.JSONDecodeError as e:
                logger.error(f"Fallo al decodificar JSON de la respuesta del LLM. Error: {e}")
                logger.debug(f"Respuesta LLM (raw): {raw_content}")
                raise ValueError(f"La respuesta del LLM no fue un JSON válido: {e}")

            # 4. Validar el diccionario contra nuestro esquema Pydantic
            # ¡Este es el paso de garantía de calidad!
            try:
                validated_data = self.target_schema.model_validate(parsed_json)
                logger.info("Parseo y validación de esquema completados exitosamente.")
                # Devolvemos como un diccionario estándar
                return validated_data.model_dump() 
                
            except ValidationError as e:
                logger.error(f"La salida del LLM no validó contra el esquema Pydantic. Error: {e}")
                logger.debug(f"JSON del LLM (parseado): {parsed_json}")
                raise ValueError(f"Datos del LLM no válidos: {e}")

        except Exception as e:
            logger.error(f"Ocurrió un error inesperado durante el parseo: {e}", exc_info=True)
            raise


# --- 3. Ejemplo de Uso (si se ejecuta este archivo) ---

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Probando StructureParser con un LLM mockeado...")
    
    # --- Mocking de un LLM ---
    # Simulamos un LLM que devuelve una respuesta JSON (a veces un poco sucia)
    class MockChatModel:
        def invoke(self, prompt_messages: Dict[str, str]) -> Any:
            
            # El texto que simulamos recibir del OCR
            ocr_input_text = prompt_messages.get("ocr_text", "") 
            
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
            
            # Simulamos que el LLM a veces añade texto extra
            mock_llm_output = f"Claro, aquí tienes el JSON:\n```json\n{mock_json_response}\n```"
            
            # Simulamos el objeto de respuesta de LangChain
            class MockResponse:
                content = mock_llm_output
            
            return MockResponse()

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
        print(json.dumps(structured_data, indent=2, ensure_ascii=False))
        
        print("\n--- Acceso a datos ---")
        print(f"Cliente: {structured_data['cliente_nombre']}")
        print(f"Total General: {structured_data['resumen_financiero']['total_general']}")
        
    except (ValueError, ValidationError) as e:
        print(f"\n--- ¡Error en el Parseo! ---")
        print(e)