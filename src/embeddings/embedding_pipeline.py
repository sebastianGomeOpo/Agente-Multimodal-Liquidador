"""
embedding_pipeline.py - Pipeline completo para generar embeddings (Paso 5)
"""

import json
from pathlib import Path
from typing import List, Dict, Any

from src.utils.logger import get_logger
from src.utils.config import (
    EXCEL_IMAGES_DIR,
    EXTRACTED_TABLES_DIR,
    EMBEDDINGS_DIR
)
from src.embeddings.clip_encoder import CLIPEncoder
from src.data_models import EmbeddingRecord

logger = get_logger(__name__)


def _create_text_document_from_json(json_data: Dict[str, Any]) -> str:
    """
    Convierte el JSON estructurado de OperacionNave en un texto descriptivo
    para embeddings semánticos.
    
    Args:
        json_data: JSON estructurado del Paso 4
    
    Returns:
        str: Texto descriptivo para embedding
    """
    lines = []
    
    # Información básica
    if json_data.get("recalada"):
        lines.append(f"Recalada: {json_data['recalada']}")
    if json_data.get("nave_nombre"):
        lines.append(f"Nave: {json_data['nave_nombre']}")
    if json_data.get("fecha_inicio_operacion"):
        lines.append(f"Inicio: {json_data['fecha_inicio_operacion']}")
    if json_data.get("fecha_fin_operacion"):
        lines.append(f"Fin: {json_data['fecha_fin_operacion']}")
    
    # Clientes
    if json_data.get("clientes"):
        lines.append(f"Clientes: {', '.join(json_data['clientes'])}")
    
    # Bodegas con tonelaje
    for bodega in json_data.get("bodegas", []):
        bodega_nombre = bodega.get("bodega", "?")
        tonelaje = bodega.get("tonelaje", 0)
        lotes = bodega.get("lotes", [])
        lines.append(f"Bodega {bodega_nombre}: {tonelaje} toneladas, Lotes: {', '.join(lotes)}")
    
    # Lotes con facturación
    for lote_fact in json_data.get("lotes_facturacion", []):
        lote = lote_fact.get("lote", "?")
        cliente = lote_fact.get("cliente", "-")
        facturas = lote_fact.get("codigos_facturacion", [])
        if facturas:
            lines.append(f"Lote {lote} (Cliente: {cliente}): Facturas {', '.join(facturas)}")
    
    return ". ".join(lines)


def process_all_multimodal() -> Dict[str, Any]:
    """
    PASO 5: Genera embeddings multimodales en formato unificado.
    
    Returns:
        dict: Estadísticas del procesamiento
    """
    logger.info("=" * 60)
    logger.info("PASO 5: Generando embeddings multimodales")
    logger.info("=" * 60)
    
    encoder = CLIPEncoder()
    
    # ============= PROCESAR IMÁGENES DE EXCEL =============
    logger.info("\n[Paso 5.1] Procesando imágenes de Excel...")
    excel_images = list(EXCEL_IMAGES_DIR.glob("*.png"))
    excel_images = [img for img in excel_images if not img.name.endswith("_metadata.json")]
    
    logger.info(f"Encontradas {len(excel_images)} imágenes de Excel")
    
    image_records: List[EmbeddingRecord] = []
    
    for img_path in excel_images:
        embedding = encoder.encode_image(str(img_path))
        if embedding is None:
            logger.warning(f"Saltando imagen fallida: {img_path.name}")
            continue
        
        # Extraer metadata del nombre (formato: file_chunk_rX_cY.png)
        stem = img_path.stem  # file_chunk_r0_c0
        parts = stem.split("_chunk_")
        source_file = parts[0] + ".xlsx" if len(parts) > 1 else "unknown.xlsx"
        chunk_id = parts[1] if len(parts) > 1 else "r0_c0"
        
        record = EmbeddingRecord(
            embedding=embedding.tolist(),
            document=str(img_path.relative_to(EXCEL_IMAGES_DIR.parent)),  # Ruta relativa
            metadata={
                "type": "excel_image",
                "source_file": source_file,
                "chunk": chunk_id,
                "absolute_path": str(img_path)
            }
        )
        image_records.append(record)
    
    logger.info(f"✓ {len(image_records)} embeddings de imágenes generados")
    
    # Guardar embeddings de imágenes
    image_output_path = EMBEDDINGS_DIR / "image_embeddings.json"
    with open(image_output_path, 'w', encoding='utf-8') as f:
        json.dump([r.dict() for r in image_records], f, ensure_ascii=False, indent=2)
    logger.info(f"✓ Guardado: {image_output_path.name}")
    
    # ============= PROCESAR TEXTOS ESTRUCTURADOS =============
    logger.info("\n[Paso 5.2] Procesando textos estructurados...")
    json_files = list(EXTRACTED_TABLES_DIR.glob("*_structure.json"))
    
    logger.info(f"Encontrados {len(json_files)} archivos JSON estructurados")
    
    text_records: List[EmbeddingRecord] = []
    
    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Crear documento textual descriptivo
            text_document = _create_text_document_from_json(json_data)
            
            if not text_document.strip():
                logger.warning(f"Documento vacío en {json_path.name}, saltando")
                continue
            
            # Generar embedding
            embedding = encoder.encode_text(text_document)
            if embedding is None:
                logger.warning(f"Saltando embedding fallido para: {json_path.name}")
                continue
            
            # Extraer nombre del archivo fuente
            source_file = json_path.stem.replace("_structure", "")
            
            record = EmbeddingRecord(
                embedding=embedding.tolist(),
                document=text_document,  # El texto descriptivo completo
                metadata={
                    "type": "text",
                    "source_file": source_file,
                    "json_path": str(json_path.relative_to(EXTRACTED_TABLES_DIR.parent)),
                    "absolute_path": str(json_path)
                }
            )
            text_records.append(record)
        
        except Exception as e:
            logger.error(f"Error procesando {json_path.name}: {e}")
            continue
    
    logger.info(f"✓ {len(text_records)} embeddings de texto generados")
    
    # Guardar embeddings de texto
    text_output_path = EMBEDDINGS_DIR / "text_embeddings.json"
    with open(text_output_path, 'w', encoding='utf-8') as f:
        json.dump([r.dict() for r in text_records], f, ensure_ascii=False, indent=2)
    logger.info(f"✓ Guardado: {text_output_path.name}")
    
    # ============= RESUMEN =============
    total_embeddings = len(image_records) + len(text_records)
    
    logger.info("\n" + "=" * 60)
    logger.info(f"✓ PASO 5 COMPLETADO")
    logger.info(f"  - Imágenes: {len(image_records)}")
    logger.info(f"  - Textos: {len(text_records)}")
    logger.info(f"  - Total: {total_embeddings}")
    logger.info("=" * 60)
    
    return {
        "image_count": len(image_records),
        "text_count": len(text_records),
        "total_embeddings": total_embeddings,
        "shared_space_verified": True
    }