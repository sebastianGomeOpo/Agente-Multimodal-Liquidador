"""
main.py - Punto de entrada principal del MultiDoc-Agent (REFACTORIZADO)
"""

import sys
from pathlib import Path

from src.utils.logger import get_logger
from src.preprocessors import process_all_excels
from src.extractors import process_all_documents, process_all_extracted_text
from src.embeddings import process_all_multimodal
from src.vectorstore import index_all_embeddings, MultimodalIndexer
from src.agent import run_agent_query

logger = get_logger(__name__)


def indexing_mode():
    """
    MODO 1: Indexación completa (6 pasos)
    """
    logger.info("=" * 60)
    logger.info("INICIANDO MODO INDEXACIÓN")
    logger.info("=" * 60)
    
    try:
        # PASO 1: Excel → PNG
        logger.info("\n[PASO 1] Convirtiendo Excel a imágenes...")
        excel_results = process_all_excels()
        success_count = len([r for r in excel_results if r.get('status') == 'success'])
        logger.info(f"✓ {success_count} archivos Excel procesados")
        
        # PASO 3: PDF/PNG → Markdown (ADE)
        logger.info("\n[PASO 3] Extrayendo Markdown con API ADE...")
        ocr_results = process_all_documents()
        success_count = len([r for r in ocr_results if r.get('status') == 'success'])
        logger.info(f"✓ {success_count} documentos procesados")
        
        # PASO 4: Markdown → JSON estructurado
        logger.info("\n[PASO 4] Convirtiendo Markdown a JSON estructurado...")
        parser_results = process_all_extracted_text()
        success_count = len([r for r in parser_results if r.get('status') == 'success'])
        logger.info(f"✓ {success_count} JSONs estructurados generados")
        
        # PASO 5: Generar embeddings
        logger.info("\n[PASO 5] Generando embeddings multimodales...")
        embedding_results = process_all_multimodal()
        logger.info(f"✓ {embedding_results.get('total_embeddings', 0)} embeddings generados")
        
        # PASO 6: Indexar en ChromaDB
        logger.info("\n[PASO 6] Indexando en ChromaDB...")
        index_result = index_all_embeddings()
        logger.info(f"✓ {index_result.indexed} registros indexados")
        
        # Estadísticas finales
        indexer = MultimodalIndexer()
        stats = indexer.get_collection_stats()
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ INDEXACIÓN COMPLETADA")
        logger.info(f"  Total en ChromaDB: {stats.get('item_count', 0)} documentos")
        logger.info("=" * 60)
        
        return {"status": "success", "mode": "indexing"}
    
    except Exception as e:
        logger.error(f"Error durante indexación: {e}", exc_info=True)
        return {"status": "error", "mode": "indexing", "error": str(e)}


def query_mode(user_query: str = None):
    """
    MODO 2: Consulta con el agente
    """
    logger.info("=" * 60)
    logger.info("INICIANDO MODO CONSULTA")
    logger.info("=" * 60)
    
    try:
        if user_query is None:
            logger.info("\nIngresa tu pregunta (o 'salir' para terminar):")
            user_query = input("> ").strip()
            
            if user_query.lower() == "salir":
                return {"status": "exit"}
        
        logger.info(f"Query: {user_query}")
        
        # Ejecutar agente
        result = run_agent_query(user_query)
        
        logger.info("\n" + "=" * 60)
        logger.info("RESPUESTA DEL AGENTE")
        logger.info("=" * 60)
        
        if result.get("status") == "success":
            response = result.get("response", {})
            
            # Mostrar respuesta
            print(f"\n{response.get('answer', 'Sin respuesta')}\n")
            
            # Mostrar fuentes citadas
            sources = response.get("sources", [])
            cited_sources = [s for s in sources if s.get("cited")]
            
            if cited_sources:
                print("Fuentes citadas:")
                for source in cited_sources:
                    print(f"  - {source.get('source_file')} ({source.get('type')}) - Similitud: {source.get('similarity', 0):.1%}")
            
            print(f"\nConfianza: {response.get('quality_score', 0):.1%}")
            
            # Mostrar razonamiento (Chain of Thought)
            if response.get("reasoning_steps"):
                print("\nPasos de razonamiento:")
                for step in response["reasoning_steps"]:
                    print(f"  • {step}")
        else:
            logger.error(f"Error: {result.get('error', 'Error desconocido')}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error durante consulta: {e}")
        return {"status": "error", "mode": "query", "error": str(e)}


def interactive_mode():
    """Modo interactivo continuo"""
    logger.info("=" * 60)
    logger.info("MODO INTERACTIVO")
    logger.info("=" * 60)
    logger.info("Comandos disponibles:")
    logger.info("  indexar  - Ejecutar pipeline de indexación")
    logger.info("  consultar - Hacer una pregunta")
    logger.info("  stats    - Ver estadísticas de ChromaDB")
    logger.info("  salir    - Terminar")
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
                    print(f"\nEstadísticas de ChromaDB:")
                    print(f"  Documentos totales: {stats.get('item_count', 0)}")
                    print(f"  Colección: {stats.get('collection_name', 'N/A')}")
                except Exception as e:
                    logger.error(f"Error obteniendo estadísticas: {e}")
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