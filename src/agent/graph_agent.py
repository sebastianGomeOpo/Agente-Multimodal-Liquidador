"""
graph_agent.py - Definición del grafo del agente con LangGraph
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from src.utils.logger import get_logger
from src.agent.nodes import query_node, retrieve_node, reason_node, format_node

logger = get_logger(__name__)


class AgentStateGraph:
    """Definición del grafo del agente con LangGraph"""
    
    def __init__(self):
        """Inicializa el grafo del agente"""
        try:
            logger.info("Inicializando AgentStateGraph")
            self.graph = self._build_graph()
            logger.info("Grafo inicializado exitosamente")
        except Exception as e:
            logger.error(f"Error al inicializar grafo: {e}")
            raise
    
    def _build_graph(self) -> StateGraph:
        """
        Construye el grafo del agente
        
        ESTRUCTURA DEL GRAFO:
        START → query_node → retrieve_node → reason_node → format_node → END
        
        Returns:
            StateGraph: Grafo compilado
        """
        
        # Crear StateGraph
        graph = StateGraph(dict)
        
        # Agregar nodos
        graph.add_node("query_node", query_node)
        graph.add_node("retrieve_node", retrieve_node)
        graph.add_node("reason_node", reason_node)
        graph.add_node("format_node", format_node)
        
        # Definir edges (transiciones)
        graph.add_edge(START, "query_node")
        graph.add_edge("query_node", "retrieve_node")
        graph.add_edge("retrieve_node", "reason_node")
        graph.add_edge("reason_node", "format_node")
        graph.add_edge("format_node", END)
        
        # Compilar el grafo
        compiled_graph = graph.compile()
        
        logger.info("Grafo compilado con éxito")
        return compiled_graph
    
    def execute(self, query: str) -> Dict[str, Any]:
        """
        Ejecuta el agente con una query
        
        Args:
            query: Query del usuario
        
        Returns:
            dict: Respuesta del agente
        """
        try:
            logger.info(f"Ejecutando agente con query: {query[:50]}")
            
            # Estado inicial
            initial_state = {
                "query": query,
                "query_embedding": None,
                "retrieved_documents": [],
                "llm_response": "",
                "final_response": {}
            }
            
            # Ejecutar el grafo
            result = self.graph.invoke(initial_state)
            
            logger.info("Ejecución del agente completada")
            
            return {
                "status": "success",
                "query": query,
                "response": result.get("final_response", {}),
                "documents_retrieved": len(result.get("retrieved_documents", []))
            }
        
        except Exception as e:
            logger.error(f"Error ejecutando agente: {e}")
            return {
                "status": "error",
                "query": query,
                "error": str(e)
            }
    
    def get_graph_info(self) -> dict:
        """
        Obtiene información sobre el grafo
        
        Returns:
            dict: Información del grafo
        """
        try:
            return {
                "nodes": ["query_node", "retrieve_node", "reason_node", "format_node"],
                "edges": [
                    ("START", "query_node"),
                    ("query_node", "retrieve_node"),
                    ("retrieve_node", "reason_node"),
                    ("reason_node", "format_node"),
                    ("format_node", "END")
                ],
                "compiled": True
            }
        except Exception as e:
            logger.error(f"Error al obtener info del grafo: {e}")
            return {"error": str(e)}


# Instancia global del agente
_agent = None


def get_agent() -> AgentStateGraph:
    """
    Obtiene la instancia global del agente
    
    Returns:
        AgentStateGraph: Instancia del agente
    """
    global _agent
    if _agent is None:
        _agent = AgentStateGraph()
    return _agent


def run_agent_query(query: str) -> Dict[str, Any]:
    """
    Ejecuta una query en el agente
    
    Args:
        query: Query del usuario
    
    Returns:
        dict: Respuesta del agente
    """
    agent = get_agent()
    return agent.execute(query)