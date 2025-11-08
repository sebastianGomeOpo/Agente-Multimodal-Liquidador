"""
main.py - Punto de entrada principal del MultiDoc-Agent (CORREGIDO)
"""

import sys
from pathlib import Path
import json  # Necesario para cargar los embeddings desde archivos JSON

from src.utils.logger import get_logger
from src.preprocessors import process_all_excels

# --- IMPORTACIÓN CORREGIDA ---
# Importamos 'process_all_documents' en lugar de 'process_all_images'
from src.extractors import process_all_documents, process_all_extracted_text
# -------------------------------

from src.embeddings import process_all_multimodal
from src.vectorstore import MultimodalIndexer
from src.agent import run_agent_query

logger = get_logger(__name__)


def indexing_mode():
    """
    MODO 1: Indexación (poblar ChromaDB)
    
    Pipeline completo (Corregido):
    1. Excel → Imágenes PNG (Solo Excel)
    2. (PDFs se procesan directamente)
    3. PDF/PNG → API ADE → Markdown
    4. Markdown → LLM → JSON Estructurado
    5. Todo → CLIP → Embeddings
    6. Embeddings → ChromaDB
    """
    logger.info("=" * 60)
    logger.info("INICIANDO MODO INDEXACIÓN (Pipeline ADE Corregido)")
    logger.info("=" * 60)
    
    try:
        # PASO 1: Convertir Excel a imágenes
        logger.info("\n[PASO 1] Convirtiendo Excel a imágenes PNG...")
        excel_results = process_all_excels()
        logger.info(
            f"Resultado: {len([r for r in excel_results if r.get('status') == 'success'])} Excel procesados"
        )
        
        # (PASO 2 eliminado)
        # Antes: convertir PDF a imágenes (fallback). Ahora los PDFs se procesan directamente en el PASO 3.
        
        # --- PASO 3 CORREGIDO ---
        # Ahora extrae Markdown de los PDF originales y las imágenes de Excel
        logger.info("\n[PASO 3] Extrayendo Markdown estructurado (con API ADE)...")
        ocr_results = process_all_documents()  # Llamada a la función moderna
        logger.info(
            f"Resultado: {len([r for r in ocr_results if r.get('status') == 'success'])} documentos procesados por ADE"
        )
        
        # --- PASO 4 CORREGIDO ---
        # Convierte el Markdown (de ADE) a JSON (con LLM)
        logger.info("\n[PASO 4] Convirtiendo Markdown a JSON (con LLM)...")
        parser_results = process_all_extracted_text()
        logger.info(
            f"Resultado: {len([r for r in parser_results if r.get('status') == 'success'])} JSONs estructurados"
        )
        
        # PASO 5: Generar embeddings con CLIP
        logger.info("\n[PASO 5] Generando embeddings multimodales con CLIP...")
        embedding_results = process_all_multimodal()
        logger.info(f"Resultado: {embedding_results}")
        
        # PASO 6: Indexar en ChromaDB (reemplazo de index_all_embeddings)
        logger.info("\n[PASO 6] Indexando en ChromaDB...")
        indexer = MultimodalIndexer()  # Instanciamos el indexador una sola vez
        
        try:
            # Definimos rutas base de embeddings
            embeddings_dir = Path("data/embeddings")
            image_embed_file = embeddings_dir / "image_embeddings.json"
            text_embed_file = embeddings_dir / "text_embeddings.json"

            all_documents_data = []

            # Cargar embeddings de imágenes
            if image_embed_file.exists():
                with open(image_embed_file, "r", encoding="utf-8") as f:
                    image_data = json.load(f)
                    all_documents_data.extend(image_data)
                    logger.info(f"Cargados {len(image_data)} embeddings de imágenes.")
            else:
                logger.warning(f"No se encontró el archivo de embeddings: {image_embed_file}")

            # Cargar embeddings de texto
            if text_embed_file.exists():
                with open(text_embed_file, "r", encoding="utf-8") as f:
                    text_data = json.load(f)
                    all_documents_data.extend(text_data)
                    logger.info(f"Cargados {len(text_data)} embeddings de texto.")
            else:
                logger.warning(f"No se encontró el archivo de embeddings: {text_embed_file}")

            # Proceder a indexar si hay datos
            if not all_documents_data:
                logger.warning(
                    "No se encontraron embeddings para indexar. El pipeline se completó sin indexar."
                )
                index_results = {
                    "status": "warning",
                    "message": "No embeddings found to index",
                }
            else:
                logger.info(
                    f"Indexando un total de {len(all_documents_data)} documentos en ChromaDB..."
                )
                # Asumimos que index_batch toma la lista de diccionarios con embedding, id/document/metadata
                index_results = indexer.index_batch(all_documents_data)
                logger.info(f"Resultado de indexación: {index_results}")

        except Exception as e:
            logger.error(f"Falló el Paso 6 (Indexación manual de JSONs): {e}", exc_info=True)
            index_results = {"status": "error", "error": str(e)}
        
        # Mostrar estadísticas finales
        logger.info("\n" + "=" * 60)
        logger.info("INDEXACIÓN COMPLETADA")
        logger.info("=" * 60)
        
        # (Esta parte de 'chroma_mgr' no existe, se mantiene comentada para evitar errores)
        # from src.vectorstore import get_chroma_manager
        # chroma_mgr = get_chroma_manager()
        # collection_info = chroma_mgr.get_collection_info()
        # logger.info(f"Documentos en ChromaDB: {collection_info.get('document_count', 0)}")
        
        # Usamos el indexer ya instanciado arriba
        stats = indexer.get_collection_stats()
        logger.info(f"Estadísticas: {stats}")
        
        return {"status": "success", "mode": "indexing"}
    
    except Exception as e:
        logger.error(f"Error fatal durante indexación: {e}", exc_info=True)
        return {"status": "error", "mode": "indexing", "error": str(e)}


def query_mode(user_query: str = None):
    """
    MODO 2: Consulta (usar el agente)
    """
    logger.info("=" * 60)
    logger.info("INICIANDO MODO CONSULTA")
    logger.info("=" * 60)
    
    try:
        if user_query is None:
            # Interfaz interactiva
            logger.info("\nIngresa tu pregunta (o 'salir' para terminar):")
            user_query = input("> ").strip()
            
            if user_query.lower() == "salir":
                logger.info("Terminando...")
                return {"status": "exit"}
        
        logger.info(f"Query: {user_query}")
        
        # Ejecutar agente
        result = run_agent_query(user_query)
        
        logger.info("\n" + "=" * 60)
        logger.info("RESPUESTA DEL AGENTE")
        logger.info("=" * 60)
        
        if result.get("status") == "success":
            response = result.get("response", {})
            logger.info(f"Respuesta: {response.get('answer', 'Sin respuesta')}")
            logger.info(f"Documentos usados: {len(response.get('sources', []))}")
            logger.info(f"Confianza: {response.get('quality_score', 0):.2%}")
        else:
            logger.error(f"Error: {result.get('error', 'Error desconocido')}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error durante consulta: {e}")
        return {"status": "error", "mode": "query", "error": str(e)}


def interactive_mode():
    """
    Modo interactivo continuo

    """
    logger.info("=" * 60)
    logger.info("MODO INTERACTIVO")
    logger.info("=" * 60)
    logger.info("Escribe 'indexar' para indexar documentos")
    logger.info("Escribe 'consultar' para hacer una pregunta")
    logger.info("Escribe 'stats' para ver estadísticas de ChromaDB")
    logger.info("Escribe 'salir' para terminar")
    logger.info("=" * 60)
    
    while True:
        try:
            command = input("\n> Comando: ").strip().lower()
            
            if command == "salir":
                logger.info("¡Hasta luego!")
                break
            elif command == "indexar":
                indexing_mode()
            elif command == "consultar":
                while True:
                    result = query_mode()
                    if result.get("status") == "exit":
                        break
            elif command == "stats":
                try:
                    indexer = MultimodalIndexer()
                    stats = indexer.get_collection_stats()
                    logger.info(f"Estadísticas de ChromaDB: {stats}")
                except Exception as e:
                    logger.error(f"Error al obtener estadísticas: {e}")
            else:
                logger.info("Comando no reconocido")
        
        except KeyboardInterrupt:
            logger.info("\n¡Hasta luego!")
            break
        except Exception as e:
            logger.error(f"Error en modo interactivo: {e}")


def main():
    """Función principal (Sin cambios)"""
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "indexar":
            indexing_mode()
        elif mode == "consultar":
            query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None
            query_mode(query)
        elif mode == "interactivo":
            interactive_mode()
        else:
            logger.error(f"Modo no reconocido: {mode}")
            logger.info("Modos disponibles: indexar, consultar, interactivo")
    else:
        interactive_mode()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nAplicación interrumpida")
    except Exception as e:
        logger.error(f"Error fatal: {e}")
        sys.exit(1)
