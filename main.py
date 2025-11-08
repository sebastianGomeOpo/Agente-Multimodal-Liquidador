"""
main.py - Punto de entrada principal del MultiDoc-Agent
"""

import sys
from pathlib import Path
from src.utils.logger import get_logger
from src.preprocessors import process_all_excels, process_all_pdfs
from src.extractors import process_all_images, process_all_extracted_text
from src.embeddings import process_all_multimodal
from src.vectorstore import index_all_embeddings, get_chroma_manager, MultimodalIndexer
from src.agent import run_agent_query

logger = get_logger(__name__)


def indexing_mode():
    """
    MODO 1: Indexación (poblar ChromaDB)
    
    Pipeline completo:
    1. Excel/PDF → Imágenes
    2. Imágenes → OCR → Texto
    3. Texto → Parser → Estructura
    4. Todo → CLIP → Embeddings
    5. Embeddings → ChromaDB
    """
    logger.info("=" * 60)
    logger.info("INICIANDO MODO INDEXACIÓN")
    logger.info("=" * 60)
    
    try:
        # PASO 1: Convertir Excel a imágenes
        logger.info("\n[PASO 1] Convirtiendo Excel a imágenes...")
        excel_results = process_all_excels()
        logger.info(f"Resultado: {len([r for r in excel_results if r.get('status') == 'success'])} Excel procesados")
        
        # PASO 2: Convertir PDF a imágenes
        logger.info("\n[PASO 2] Convirtiendo PDF a imágenes...")
        pdf_results = process_all_pdfs()
        logger.info(f"Resultado: {len([r for r in pdf_results if r.get('status') == 'success'])} PDF procesados")
        
        # PASO 3: Extraer texto con OCR
        logger.info("\n[PASO 3] Extrayendo texto con OCR...")
        ocr_results = process_all_images()
        logger.info(f"Resultado: {len([r for r in ocr_results if r.get('status') == 'success'])} imágenes procesadas")
        
        # PASO 4: Parsear y estructurar texto
        logger.info("\n[PASO 4] Parseando y estructurando texto...")
        parser_results = process_all_extracted_text()
        logger.info(f"Resultado: {len([r for r in parser_results if r.get('status') == 'success'])} textos estructurados")
        
        # PASO 5: Generar embeddings con CLIP
        logger.info("\n[PASO 5] Generando embeddings multimodales con CLIP...")
        embedding_results = process_all_multimodal()
        logger.info(f"Resultado: {embedding_results}")
        
        # PASO 6: Indexar en ChromaDB
        logger.info("\n[PASO 6] Indexando en ChromaDB...")
        index_results = index_all_embeddings()
        logger.info(f"Resultado: {index_results}")
        
        # Mostrar estadísticas finales
        logger.info("\n" + "=" * 60)
        logger.info("INDEXACIÓN COMPLETADA")
        logger.info("=" * 60)
        
        chroma_mgr = get_chroma_manager()
        collection_info = chroma_mgr.get_collection_info()
        logger.info(f"Documentos en ChromaDB: {collection_info.get('document_count', 0)}")
        
        indexer = MultimodalIndexer()
        stats = indexer.get_collection_stats()
        logger.info(f"Estadísticas: {stats}")
        
        return {"status": "success", "mode": "indexing"}
    
    except Exception as e:
        logger.error(f"Error durante indexación: {e}")
        return {"status": "error", "mode": "indexing", "error": str(e)}


def query_mode(user_query: str = None):
    """
    MODO 2: Consulta (usar el agente)
    
    Pipeline:
    1. Recibir query del usuario
    2. Query → Embedding CLIP
    3. Buscar en ChromaDB
    4. LLM razona
    5. Formatea respuesta
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
                    logger.info(f"Estadísticas: {stats}")
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
    """Función principal"""
    
    if len(sys.argv) > 1:
        # Modo por línea de comandos
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
        # Modo interactivo por defecto
        interactive_mode()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nAplicación interrumpida")
    except Exception as e:
        logger.error(f"Error fatal: {e}")
        sys.exit(1)