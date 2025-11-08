"""
nodes.py - Nodos del grafo ReAct con Chain of Thought
"""

from typing import Dict, Any, List
from src.utils.logger import get_logger
from src.utils.config import RETRIEVE_TOP_K, SIMILARITY_THRESHOLD, LLM_MODEL, LLM_TEMPERATURE, REACT_MAX_ITERATIONS
from src.vectorstore.multimodal_indexer import MultimodalIndexer
from src.data_models import RetrievalResult
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

logger = get_logger(__name__)


def query_node(state: dict) -> dict:
    """
    NODO 1: Analiza la query del usuario usando Chain of Thought.
    
    Responsabilidades:
    - Normalizar la query
    - Identificar el tipo de pregunta (numérica, descriptiva, temporal)
    - Decidir estrategia de búsqueda
    
    Args:
        state: Estado del agente
    
    Returns:
        dict: Estado actualizado con análisis de la query
    """
    try:
        logger.info("=" * 60)
        logger.info("NODO 1: Análisis de Query (CoT)")
        logger.info("=" * 60)
        
        query = state.get("query", "").strip()
        
        if not query:
            logger.error("Query vacía")
            state["error"] = "Query vacía"
            return state
        
        # ============= CHAIN OF THOUGHT: ANÁLISIS DE QUERY =============
        reasoning_steps = []
        
        # Paso 1: Identificar palabras clave
        query_lower = query.lower()
        
        # Palabras clave para preguntas numéricas
        numeric_keywords = ["cuánto", "cuanto", "tonelaje", "cantidad", "total", "suma"]
        is_numeric = any(kw in query_lower for kw in numeric_keywords)
        
        # Palabras clave para preguntas temporales
        temporal_keywords = ["cuándo", "cuando", "fecha", "inicio", "fin", "periodo"]
        is_temporal = any(kw in query_lower for kw in temporal_keywords)
        
        # Palabras clave para entidades
        entities = {
            "nave": ["nave", "barco", "buque"],
            "bodega": ["bodega", "hold"],
            "lote": ["lote", "batch"],
            "cliente": ["cliente", "customer"],
            "recalada": ["recalada", "escala"]
        }
        
        detected_entities = []
        for entity_type, keywords in entities.items():
            if any(kw in query_lower for kw in keywords):
                detected_entities.append(entity_type)
        
        reasoning_steps.append(f"Query original: '{query}'")
        reasoning_steps.append(f"Tipo de pregunta: {'Numérica' if is_numeric else 'Temporal' if is_temporal else 'Descriptiva'}")
        reasoning_steps.append(f"Entidades detectadas: {', '.join(detected_entities) if detected_entities else 'ninguna'}")
        
        # Paso 2: Decidir estrategia de búsqueda
        if is_numeric or is_temporal:
            # Priorizar documentos de texto (tienen los datos estructurados)
            search_strategy = "text_prioritized"
            reasoning_steps.append("Estrategia: Priorizar documentos de texto (contienen datos numéricos/fechas)")
        else:
            # Búsqueda híbrida (texto + imágenes)
            search_strategy = "hybrid"
            reasoning_steps.append("Estrategia: Búsqueda híbrida (texto e imágenes)")
        
        # Guardar análisis en el estado
        state["prepared_query"] = query
        state["query_analysis"] = {
            "is_numeric": is_numeric,
            "is_temporal": is_temporal,
            "entities": detected_entities,
            "search_strategy": search_strategy
        }
        state["reasoning_steps"] = reasoning_steps
        
        logger.info("✓ Análisis de query completado")
        for step in reasoning_steps:
            logger.info(f"  {step}")
        
        return state
    
    except Exception as e:
        logger.error(f"Error en query_node: {e}")
        state["error"] = str(e)
        return state


def retrieve_node(state: dict) -> dict:
    """
    NODO 2: Recupera documentos relevantes usando la estrategia decidida.
    
    Responsabilidades:
    - Ejecutar búsqueda semántica
    - Aplicar filtros según estrategia
    - Rankear resultados por similitud
    
    Args:
        state: Estado del agente
    
    Returns:
        dict: Estado actualizado con documentos recuperados
    """
    try:
        logger.info("=" * 60)
        logger.info("NODO 2: Recuperación de Documentos")
        logger.info("=" * 60)
        
        prepared_query = state.get("prepared_query", "")
        query_analysis = state.get("query_analysis", {})
        reasoning_steps = state.get("reasoning_steps", [])
        
        indexer = MultimodalIndexer()
        
        search_strategy = query_analysis.get("search_strategy", "hybrid")
        
        retrieved_docs: List[RetrievalResult] = []
        
        # ============= ESTRATEGIA: TEXT PRIORITIZED =============
        if search_strategy == "text_prioritized":
            reasoning_steps.append("Buscando en documentos de texto...")
            
            # Buscar solo en documentos de texto
            text_results = indexer.semantic_search(
                query_text=prepared_query,
                n_results=RETRIEVE_TOP_K,
                doc_type="text",
                min_similarity=SIMILARITY_THRESHOLD
            )
            
            retrieved_docs.extend(text_results)
            reasoning_steps.append(f"Encontrados {len(text_results)} documentos de texto relevantes")
            
            # Si no hay suficientes resultados, buscar en imágenes también
            if len(retrieved_docs) < RETRIEVE_TOP_K:
                reasoning_steps.append("Ampliando búsqueda a imágenes...")
                image_results = indexer.semantic_search(
                    query_text=prepared_query,
                    n_results=RETRIEVE_TOP_K - len(retrieved_docs),
                    doc_type="excel_image",
                    min_similarity=SIMILARITY_THRESHOLD
                )
                retrieved_docs.extend(image_results)
                reasoning_steps.append(f"Encontradas {len(image_results)} imágenes adicionales")
        
        # ============= ESTRATEGIA: HYBRID =============
        else:
            reasoning_steps.append("Buscando en todos los tipos de documentos...")
            
            # Buscar sin filtro de tipo
            all_results = indexer.semantic_search(
                query_text=prepared_query,
                n_results=RETRIEVE_TOP_K,
                doc_type=None,
                min_similarity=SIMILARITY_THRESHOLD
            )
            
            retrieved_docs.extend(all_results)
            reasoning_steps.append(f"Encontrados {len(all_results)} documentos relevantes")
        
        # Ordenar por similitud (mayor a menor)
        retrieved_docs.sort(key=lambda x: x.similarity, reverse=True)
        
        # Logging de resultados
        logger.info(f"✓ Recuperados {len(retrieved_docs)} documentos:")
        for i, doc in enumerate(retrieved_docs, 1):
            logger.info(f"  [{i}] Tipo: {doc.metadata.get('type')}, Similitud: {doc.similarity:.3f}, ID: {doc.id}")
        
        state["retrieved_documents"] = [doc.dict() for doc in retrieved_docs]
        state["reasoning_steps"] = reasoning_steps
        
        return state
    
    except Exception as e:
        logger.error(f"Error en retrieve_node: {e}")
        state["error"] = str(e)
        state["retrieved_documents"] = []
        return state


def reason_node(state: dict) -> dict:
    """
    NODO 3: Razona sobre documentos recuperados usando LLM con ReAct.
    
    Responsabilidades:
    - Analizar documentos recuperados
    - Generar respuesta basada SOLO en documentos
    - Citar fuentes explícitamente
    
    Args:
        state: Estado del agente
    
    Returns:
        dict: Estado actualizado con respuesta del LLM
    """
    try:
        logger.info("=" * 60)
        logger.info("NODO 3: Razonamiento con LLM (ReAct)")
        logger.info("=" * 60)
        
        query = state.get("prepared_query", "")
        documents = state.get("retrieved_documents", [])
        reasoning_steps = state.get("reasoning_steps", [])
        
        if not documents:
            logger.warning("No hay documentos para razonar")
            state["llm_response"] = "No encontré documentos relevantes para responder tu pregunta."
            state["citations"] = []
            return state
        
        # ============= CONSTRUIR CONTEXTO PARA LLM =============
        context_parts = []
        context_parts.append("DOCUMENTOS RECUPERADOS (ordenados por relevancia):\n")
        
        for i, doc in enumerate(documents, 1):
            doc_type = doc["metadata"].get("type", "unknown")
            source = doc["metadata"].get("source_file", "unknown")
            similarity = doc.get("similarity", 0)
            content = doc.get("document", "")
            
            context_parts.append(f"\n[Documento {i}]")
            context_parts.append(f"Fuente: {source}")
            context_parts.append(f"Tipo: {doc_type}")
            context_parts.append(f"Relevancia: {similarity:.1%}")
            context_parts.append(f"Contenido: {content[:500]}...")  # Primeros 500 chars
            context_parts.append("-" * 50)
        
        context = "\n".join(context_parts)
        
        # ============= PROMPT PARA LLM (ReAct + CoT) =============
        system_prompt = """Eres un asistente experto en análisis de documentos portuarios.

INSTRUCCIONES CRÍTICAS:
1. Responde SOLO basándote en los documentos proporcionados
2. SIEMPRE cita la fuente usando el formato: [Documento X]
3. Si la información no está en los documentos, di explícitamente "No encontré esta información"
4. Sé preciso con números, fechas y nombres
5. Si hay múltiples fuentes, menciona todas

FORMATO DE RESPUESTA:
- Responde en español de forma clara y concisa
- Usa viñetas si hay múltiples puntos
- Cita al final de cada afirmación: [Documento X]

NO INVENTES información que no esté en los documentos."""

        user_prompt = f"""PREGUNTA DEL USUARIO:
{query}

{context}

RESPONDE LA PREGUNTA usando los documentos anteriores. Recuerda citar las fuentes."""

        # ============= LLAMADA AL LLM =============
        try:
            llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            reasoning_steps.append(f"Invocando LLM ({LLM_MODEL}) con {len(documents)} documentos como contexto")
            logger.info(f"Llamando a LLM: {LLM_MODEL}...")
            
            response = llm.invoke(messages)
            llm_answer = response.content
            
            reasoning_steps.append(f"LLM respondió con {len(llm_answer)} caracteres")
            logger.info(f"✓ Respuesta del LLM recibida ({len(llm_answer)} chars)")
            
            # Extraer citaciones del texto (formato [Documento X])
            import re
            citations = re.findall(r'\[Documento (\d+)\]', llm_answer)
            citations = list(set(citations))  # Únicos
            
            state["llm_response"] = llm_answer
            state["citations"] = [int(c) for c in citations]
            state["reasoning_steps"] = reasoning_steps
            
            logger.info(f"Citaciones encontradas: {citations}")
        
        except Exception as e:
            logger.error(f"Error llamando al LLM: {e}")
            state["llm_response"] = f"Error al procesar con LLM: {str(e)}"
            state["citations"] = []
        
        return state
    
    except Exception as e:
        logger.error(f"Error en reason_node: {e}")
        state["error"] = str(e)
        return state


def format_node(state: dict) -> dict:
    """
    NODO 4: Formatea la respuesta final con quality score.
    
    Responsabilidades:
    - Calcular quality score basado en similitudes reales
    - Formatear fuentes citadas
    - Ensamblar respuesta final
    
    Args:
        state: Estado del agente
    
    Returns:
        dict: Estado actualizado con respuesta final
    """
    try:
        logger.info("=" * 60)
        logger.info("NODO 4: Formateo de Respuesta Final")
        logger.info("=" * 60)
        
        llm_response = state.get("llm_response", "")
        documents = state.get("retrieved_documents", [])
        citations = state.get("citations", [])
        reasoning_steps = state.get("reasoning_steps", [])
        
        # ============= CALCULAR QUALITY SCORE =============
        if not documents:
            quality_score = 0.0
        else:
            # Score basado en similitudes de documentos citados
            cited_docs = [documents[i-1] for i in citations if 0 < i <= len(documents)]
            
            if cited_docs:
                # Promedio de similitudes de documentos citados
                avg_similarity = sum(doc.get("similarity", 0) for doc in cited_docs) / len(cited_docs)
                
                # Bonus por múltiples fuentes (max +0.1)
                multi_source_bonus = min(0.1, (len(cited_docs) - 1) * 0.03)
                
                quality_score = min(1.0, avg_similarity + multi_source_bonus)
            else:
                # Usar promedio de todos los documentos si no hay citaciones
                avg_similarity = sum(doc.get("similarity", 0) for doc in documents) / len(documents)
                quality_score = avg_similarity * 0.8  # Penalizar por falta de citaciones
        
        reasoning_steps.append(f"Quality score calculado: {quality_score:.2%}")
        
        # ============= FORMATEAR FUENTES =============
        sources = []
        for i, doc in enumerate(documents, 1):
            sources.append({
                "id": doc.get("id"),
                "type": doc["metadata"].get("type"),
                "source_file": doc["metadata"].get("source_file"),
                "similarity": doc.get("similarity"),
                "cited": i in citations
            })
        
        # ============= RESPUESTA FINAL =============
        final_response = {
            "answer": llm_response,
            "sources": sources,
            "quality_score": quality_score,
            "reasoning_steps": reasoning_steps,
            "total_documents_retrieved": len(documents),
            "documents_cited": len(citations)
        }
        
        state["final_response"] = final_response
        
        logger.info(f"✓ Respuesta formateada:")
        logger.info(f"  - Quality Score: {quality_score:.2%}")
        logger.info(f"  - Documentos recuperados: {len(documents)}")
        logger.info(f"  - Documentos citados: {len(citations)}")
        
        return state
    
    except Exception as e:
        logger.error(f"Error en format_node: {e}")
        state["error"] = str(e)
        return state