"""
structure_parser.py - PASO 4 de la Pipeline de Indexación (ENFOQUE REFINADO)

Objetivo: Extraer *únicamente* la información que importa para operaciones de naves:
- recalada
- nombre de la nave
- fecha de inicio de operación
- fecha de fin de operación
- bodegas con tonelaje (> 0)
- tonelaje por bodega
- lote(s) que ingresaron a cada bodega
- clientes atendidos en la nave
- por cada lote: códigos de facturación según el cliente

Implementa extracción con LLM usando salida estructurada (Pydantic) para robustez.
Incluye post-proceso "inteligente" para normalizar números, fechas y deducir clientes si faltan.
"""

# --- Importaciones ---
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
    LLM_TEMPERATURE,
)

# Configuración del logger
logger = get_logger(__name__)


# ============================================================
# 1) ESQUEMA DE SALIDA (solo lo que nos importa)
# ============================================================

class LoteFactura(BaseModel):
    """
    Información por lote (a nivel de facturación y cliente).
    """
    lote: str = Field(description="Identificador del lote (string exacto tal como aparece en el documento).")
    cliente: Optional[str] = Field(None, description="Cliente asociado a este lote (si se deduce).")
    codigos_facturacion: List[str] = Field(default_factory=list, description="Códigos/IDs de factura del PDF para este lote y cliente.")
    bodega: Optional[str] = Field(None, description="Bodega en la que ingresó este lote (si aplica).")
    tonelaje: Optional[float] = Field(None, description="Tonelaje asociado a este lote (si se reporta).")

class BodegaRegistro(BaseModel):
    """
    Registro por bodega con tonelaje y lotes asociados.
    """
    bodega: str = Field(description="Nombre o identificador de la bodega (e.g., 'B1', 'Bodega 3').")
    tonelaje: float = Field(description="Tonelaje total reportado en esta bodega (solo bodegas con tonelaje > 0).")
    lotes: List[str] = Field(default_factory=list, description="Lista de lotes (strings) que ingresaron a esta bodega.")

class OperacionNave(BaseModel):
    """
    Esquema raíz: solo campos operativos relevantes.
    """
    recalada: Optional[str] = Field(None, description="Identificador o nombre de la recalada/escala.")
    nave_nombre: Optional[str] = Field(None, description="Nombre de la nave/barco.")
    fecha_inicio_operacion: Optional[str] = Field(None, description="YYYY-MM-DD si es posible normalizar.")
    fecha_fin_operacion: Optional[str] = Field(None, description="YYYY-MM-DD si es posible normalizar.")
    bodegas: List[BodegaRegistro] = Field(default_factory=list, description="Solo bodegas con tonelaje > 0.")
    clientes: List[str] = Field(default_factory=list, description="Clientes atendidos en la nave.")
    lotes_facturacion: List[LoteFactura] = Field(
        default_factory=list,
        description="Detalle por lote con relación cliente ↔ códigos de facturación."
    )
    # Campo opcional que mejora la “visualización inteligente”
    vista_inteligente: Optional[str] = Field(
        None,
        description="Resumen en Markdown de la operación (generado en post-proceso)."
    )


# ============================================================
# 2) PARSER ESTRUCTURADO (con salida tipada) + utilidades
# ============================================================

class StructureParser:
    """
    Toma texto en formato MARKDOWN (proveniente de ADE) y usa un LLM
    para extraer SOLO la información operacional relevante (OperacionNave).
    """

    def __init__(self, llm: BaseChatModel, schema: Type[BaseModel] = OperacionNave):
        self.llm = llm
        self.target_schema = schema
        self.target_schema_json = json.dumps(schema.model_json_schema(), indent=2)
        logger.info(f"StructureParser inicializado con el esquema: {schema.__name__}")

    def _get_parsing_prompt(self) -> ChatPromptTemplate:
        """
        Prompt *enfocado* al dominio: documentos en Markdown con tablas/listas.
        Pide explícitamente SOLO los campos operativos.
        """
        system_prompt = f"""
Eres un asistente experto en extracción de datos portuarios/logísticos.
Tu tarea: leer un documento en **Markdown** y devolver **EXCLUSIVAMENTE** un JSON
válido que cumpla el esquema. No devuelvas texto adicional ni bloques de código.

EXTRAE SOLO ESTOS CAMPOS (ignora todo lo demás):
- recalada
- nave_nombre
- fecha_inicio_operacion (YYYY-MM-DD si puedes)
- fecha_fin_operacion (YYYY-MM-DD si puedes)
- bodegas: SOLO bodegas con tonelaje > 0.
  - bodega (string)
  - tonelaje (float)
  - lotes: lista de strings de lotes que ingresaron a esa bodega
- clientes: lista con los clientes atendidos en la nave (únicos, normalizados)
- lotes_facturacion: por cada lote:
  - lote (string)
  - cliente (string si se deduce)
  - codigos_facturacion (lista de strings)
  - bodega (string si aplica)
  - tonelaje (float si aplica)

REGLAS:
- Limpia números: "$1,234.56" → 1234.56 ; "1.234,56" → 1234.56
- Normaliza fechas a YYYY-MM-DD (si no es posible, deja el valor original o null)
- Devuelve solo literales JSON que el esquema pueda validar.
- NO inventes campos fuera del esquema, no omitas claves requeridas del esquema.
- Si careces de información, usa valores nulos o listas vacías según el tipo.

ESQUEMA OBJETIVO:
{self.target_schema_json}
        """.strip()

        return ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessage(content="{ocr_text}")  # el input real en Markdown
        ])

    # Fallback por si alguna vez se usa salida no estructurada
    def _extract_json_block(self, text: str) -> str:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return match.group(0) if match else text

    # ---------------------------
    # Normalizadores inteligentes
    # ---------------------------
    @staticmethod
    def _normalize_number(x) -> Optional[float]:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if not s:
            return None
        # Quitar símbolos comunes y unificar separadores
        s = s.replace("$", "").replace("€", "").replace("S/.", "").replace("S/", "")
        s = s.replace(" ", "")
        # Formatos 1.234,56 → 1234.56
        if "," in s and "." in s and s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
        try:
            return float(s)
        except Exception:
            return None

    @staticmethod
    def _smart_clients(op: Dict[str, Any]) -> List[str]:
        """
        Deduce 'clientes' si viene vacío: agrupa clientes de lotes_facturacion.
        """
        clientes = set(op.get("clientes") or [])
        for lf in op.get("lotes_facturacion", []) or []:
            c = (lf or {}).get("cliente")
            if c:
                clientes.add(str(c).strip())
        # Limpieza básica
        clientes = {c for c in clientes if c}
        return sorted(clientes)

    @staticmethod
    def _smart_filter_bodegas(op: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Filtra bodegas con tonelaje > 0 y normaliza tonelajes.
        """
        bodegas = []
        for b in (op.get("bodegas") or []):
            tonelaje = StructureParser._normalize_number(b.get("tonelaje"))
            if tonelaje and tonelaje > 0:
                bodegas.append({
                    "bodega": str(b.get("bodega") or "").strip(),
                    "tonelaje": float(round(tonelaje, 3)),
                    "lotes": [str(x).strip() for x in (b.get("lotes") or []) if str(x).strip()]
                })
        return bodegas

    @staticmethod
    def _smart_link_bodega_in_lotes(op: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Si en lotes_facturacion falta la 'bodega', intenta deducirla
        buscando el lote dentro de bodegas.lotes.
        """
        bodegas = op.get("bodegas") or []
        lotes_fact = []
        for lf in (op.get("lotes_facturacion") or []):
            lf = dict(lf or {})
            if not lf.get("bodega"):
                lote = lf.get("lote")
                if lote:
                    for b in bodegas:
                        if lote in (b.get("lotes") or []):
                            lf["bodega"] = b.get("bodega")
                            break
            # Normaliza tonelaje del lote si es string
            lf["tonelaje"] = StructureParser._normalize_number(lf.get("tonelaje"))
            lotes_fact.append(lf)
        return lotes_fact

    @staticmethod
    def _build_smart_view(op: Dict[str, Any]) -> str:
        """
        Genera una vista Markdown compacta e “inteligente” con lo clave.
        """
        lines = []
        lines.append(f"## Operación de Nave")
        lines.append(f"- **Recalada:** {op.get('recalada') or '-'}")
        lines.append(f"- **Nave:** {op.get('nave_nombre') or '-'}")
        lines.append(f"- **Inicio:** {op.get('fecha_inicio_operacion') or '-'}")
        lines.append(f"- **Fin:** {op.get('fecha_fin_operacion') or '-'}")
        lines.append("")
        # Bodegas
        bodegas = op.get("bodegas") or []
        if bodegas:
            lines.append("### Bodegas con tonelaje")
            for b in bodegas:
                lines.append(f"- **{b.get('bodega','?')}** → {b.get('tonelaje','?')} t | Lotes: {', '.join(b.get('lotes') or []) or '-'}")
            lines.append("")
        # Clientes
        clientes = op.get("clientes") or []
        if clientes:
            lines.append(f"### Clientes atendidos ({len(clientes)})")
            lines.append(", ".join(clientes))
            lines.append("")
        # Lotes y facturación
        lotes_fact = op.get("lotes_facturacion") or []
        if lotes_fact:
            lines.append("### Lotes ↔ Facturación")
            for lf in lotes_fact:
                lote = lf.get("lote") or "?"
                cliente = lf.get("cliente") or "-"
                bodega = lf.get("bodega") or "-"
                tonelaje = lf.get("tonelaje")
                t_str = f"{tonelaje} t" if tonelaje is not None else "-"
                cods = lf.get("codigos_facturacion") or []
                lines.append(f"- **Lote {lote}** | Cliente: {cliente} | Bodega: {bodega} | Tonelaje: {t_str} | Facturas: {', '.join(cods) or '-'}")
        return "\n".join(lines)

    # ---------------------------
    # Proceso principal
    # ---------------------------
    def parse_document(self, ocr_text: str) -> Dict[str, Any]:
        """
        Lee Markdown (ocr_text), invoca LLM con salida estructurada y aplica
        post-procesado para limpiar y completar relaciones.
        """
        logger.info("Iniciando parseo estructurado (Markdown -> OperacionNave)...")

        # 1) LLM con salida estructurada (Pydantic)
        prompt = self._get_parsing_prompt()
        structured_llm = self.llm.with_structured_output(self.target_schema)
        chain = prompt | structured_llm

        try:
            logger.debug("Invocando LLM (structured output)...")
            result_obj = chain.invoke({"ocr_text": ocr_text})

            # 2) Validación Pydantic (por si acaso) + dict
            validated = self.target_schema.model_validate(result_obj)
            op = validated.model_dump()

        except Exception as e:
            logger.error(f"Fallo en structured output; intentando fallback estándar: {e}", exc_info=True)
            # --- Fallback a modo “texto → JSON” clásico ---
            raw_chain = prompt | self.llm
            response = raw_chain.invoke({"ocr_text": ocr_text})
            raw_content = response.content
            json_string = self._extract_json_block(raw_content)

            try:
                parsed_json = json.loads(json_string)
            except json.JSONDecodeError as je:
                logger.error(f"Respuesta del LLM no fue JSON válido: {je}")
                logger.debug(f"Respuesta LLM (raw): {raw_content}")
                raise ValueError(f"La respuesta del LLM no fue un JSON válido: {je}")

            try:
                validated_data = self.target_schema.model_validate(parsed_json)
                op = validated_data.model_dump()
            except ValidationError as ve:
                logger.error(f"Datos del LLM no validan contra el esquema: {ve}")
                logger.debug(f"JSON del LLM (parseado): {parsed_json}")
                raise ValueError(f"Datos del LLM no válidos: {ve}")

        # 3) Post-proceso “inteligente”
        op["bodegas"] = self._smart_filter_bodegas(op)
        op["lotes_facturacion"] = self._smart_link_bodega_in_lotes(op)
        op["clientes"] = self._smart_clients(op)
        op["vista_inteligente"] = self._build_smart_view(op)

        logger.info("Parseo y normalización completados.")
        return op


# ============================================================
# 3) ORQUESTADOR DEL PASO 4 (procesa todos los *_text.json)
# ============================================================

def process_all_extracted_text() -> List[Dict[str, Any]]:
    """
    Orquesta el PASO 4:
      1. Inicializa LLM + Parser con esquema OperacionNave.
      2. Busca todos los *_text.json (Markdown).
      3. Parsea cada archivo y guarda *_structure.json en EXTRACTED_TABLES_DIR.
    """
    logger.info("Iniciando PASO 4: Extracción enfocada (Markdown → OperacionNave JSON)...")

    try:
        llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
        parser = StructureParser(llm=llm, schema=OperacionNave)
    except Exception as e:
        logger.error(f"No se pudo inicializar el LLM para el parser: {e}")
        logger.error("Asegúrate de que OPENAI_API_KEY esté en tu archivo .env")
        return [{"status": "error", "message": "Fallo al iniciar LLM"}]

    text_files = list(EXTRACTED_TEXT_DIR.glob("*_text.json"))
    logger.info(f"Se encontraron {len(text_files)} archivos de Markdown para parsear.")

    results = []

    for text_file_path in text_files:
        try:
            with open(text_file_path, "r", encoding="utf-8") as f:
                ocr_data = json.load(f)

            ocr_text = ocr_data.get("text")
            if not ocr_text:
                logger.warning(f"Entrada vacía (sin texto/markdown): {text_file_path.name}")
                results.append({"status": "skipped", "file": text_file_path.name, "message": "Texto vacío"})
                continue

            logger.debug(f"Parseando {text_file_path.name}...")
            structured_data = parser.parse_document(ocr_text)

            output_filename = f"{text_file_path.stem.replace('_text', '')}_structure.json"
            output_path = EXTRACTED_TABLES_DIR / output_filename

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(structured_data, f, ensure_ascii=False, indent=2)

            logger.info(f"JSON (OperacionNave) guardado en: {output_path.name}")
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
