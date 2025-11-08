"""
nodes.py - Nodos del grafo de LangGraph
"""

from typing import Any
from src.utils.logger import get_logger
from src.embeddings.clip_encoder import CLIPEncoder
from src.vectorstore.multimodal_indexer import MultimodalIndexer
from src.agent.tools import (
    calculate_totals, validate_dates, compare_values,
    extract_key_info, search_by_metadata
)

logger = get_logger(__name__)


class State:
    """Estado del agente"""
    def __init__(self):
        self.query = ""
        self.query_embedding = None
        self.retrieved_documents = []
        self.llm_response = ""
        self.final_response = ""
        self.metadata = {}


def query_node(state: dict) -> dict:
    """
    NODO 1: Procesa la query del usuario
    
    Args:
        state: Estado del agente
    
    Returns:
        dict: Estado actualizado
    """
    try:
        logger.info(f"Query node - Query: {state.get('query', '')[:50]}")
        
        # La query ya viene en el estado
        query = state.get("query", "")
        
        if not query:
            logger.warning("Query vacía recibida")
            state["error"] = "Query vacía"
            return state
        
        # Validar y preparar query
        prepared_query = query.strip()
        
        logger.info(f"Query preparada: {prepared_query}")
        state["prepared_query"] = prepared_query
        
        return state
    
    except Exception as e:
        logger.error(f"Error en query_node: {e}")
        state["error"] = str(e)
        return state


def retrieve_node(state: dict) -> dict:
    """
    NODO 2: Recupera documentos relevantes de ChromaDB
    
    Args:
        state: Estado del agente
    
    Returns:
        dict: Estado actualizado
    """
    try:
        logger.info("Retrieve node - Buscando documentos")
        
        prepared_query = state.get("prepared_query", "")
        
        try:
            # Inicializar encoder y indexer
            encoder = CLIPEncoder()
            indexer = MultimodalIndexer()
            
            # Convertir query a embedding con CLIP
            query_embedding = encoder.encode_text(prepared_query)
            
            if query_embedding is None:
                logger.warning("No se pudo codificar la query")
                state["retrieved_documents"] = []
                return state
            
            state["query_embedding"] = query_embedding.tolist()
            
            # Buscar en ChromaDB
            search_results = indexer.search_by_embedding(query_embedding.tolist())
            
            retrieved_docs = search_results.get("documents", [])
            logger.info(f"Documentos recuperados: {len(retrieved_docs)}")
            
            state["retrieved_documents"] = retrieved_docs
            
        except Exception as e:
            logger.error(f"Error durante retrieval: {e}")
            state["retrieved_documents"] = []
        
        return state
    
    except Exception as e:
        logger.error(f"Error en retrieve_node: {e}")
        state["error"] = str(e)
        return state


def reason_node(state: dict) -> dict:
    """
    NODO 3: LLM razona sobre documentos recuperados
    
    Args:
        state: Estado del agente
    
    Returns:
        dict: Estado actualizado
    """
    try:
        logger.info("Reason node - Razonando con LLM")
        
        query = state.get("prepared_query", "")
        documents = state.get("retrieved_documents", [])
        
        if not documents:
            logger.warning("No hay documentos para razonar")
            state["llm_response"] = "No se encontraron documentos relevantes"
            return state
        
        # Construir contexto
        context = self._build_context(documents)
        
        # Aquí iría la llamada al LLM (OpenAI, DeepSeek, etc)
        # Por ahora, usamos una respuesta simulada
        
        logger.info("LLM processing simulado")
        
        state["llm_response"] = self._process_with_llm(query, context)
        state["reasoning_complete"] = True
        
        return state
    
    except Exception as e:
        logger.error(f"Error en reason_node: {e}")
        state["error"] = str(e)
        return state


def format_node(state: dict) -> dict:
    """
    NODO 4: Formatea la respuesta final
    
    Args:
        state: Estado del agente
    
    Returns:
        dict: Estado actualizado
    """
    try:
        logger.info("Format node - Formateando respuesta")
        
        llm_response = state.get("llm_response", "")
        documents = state.get("retrieved_documents", [])
        
        # Formatear respuesta
        final_response = {
            "answer": llm_response,
            "sources": [
                {
                    "id": doc.get("id"),
                    "type": doc.get("metadata", {}).get("type"),
                    "confidence": 1.0 - doc.get("distance", 0)
                }
                for doc in documents
            ],
            "query": state.get("query", "")
        }
        
        # Validar calidad
        if not llm_response:
            logger.warning("Respuesta vacía del LLM")
            final_response["quality_score"] = 0.0
        else:
            final_response["quality_score"] = min(1.0, len(documents) * 0.3 + 0.7)
        
        state["final_response"] = final_response
        
        logger.info("Respuesta formateada")
        return state
    
    except Exception as e:
        logger.error(f"Error en format_node: {e}")
        state["error"] = str(e)
        return state


def _build_context(documents: list) -> str:
    """
    Construye contexto para el LLM
    
    Args:
        documents: Documentos recuperados
    
    Returns:
        str: Contexto formateado
    """
    context = "DOCUMENTOS RECUPERADOS:\n"
    context += "=" * 50 + "\n"
    
    for i, doc in enumerate(documents, 1):
        context += f"\n[Documento {i}]\n"
        context += f"Tipo: {doc.get('metadata', {}).get('type', 'desconocido')}\n"
        context += f"Relevancia: {1.0 - doc.get('distance', 0):.2f}\n"
        context += f"Contenido: {doc.get('document', '')[:200]}...\n"
    
    return context


def _process_with_llm(query: str, context: str) -> str:
    """
    Procesa query con LLM (simulado)
    
    Args:
        query: Query del usuario
        context: Contexto de documentos
    
    Returns:
        str: Respuesta del LLM
    """
    # Aquí iría integración con OpenAI API
    # Por ahora retorna respuesta simulada
    
    logger.info("Simulando respuesta del LLM")
    
    response = f"""
    En base a los documentos recuperados, he encontrado la siguiente información:
    
    Tu pregunta: "{query}"
    
    {context}
    
    Respuesta: He analizado los documentos relevantes y puedo proporcionarte la información solicitada.
    """
    
    return response