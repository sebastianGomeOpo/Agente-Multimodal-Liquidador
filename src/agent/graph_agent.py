"""
graph_agent.py - Definición del grafo del agente con LangGraph
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from src.utils.logger import get_logger
from src.agent.nodes import query_node, retrieve_node, reason_node, format_node

logger = get_logger(__name__)


class AgentStateGraph:
    """Grafo del agente con ReAct + CoT"""
    
    def __init__(self):
        try:
            logger.info("Inicializando AgentStateGraph (ReAct)")
            self.graph = self._build_graph()
            logger.info("✓ Grafo inicializado")
        except Exception as e:
            logger.error(f"Error inicializando grafo: {e}")
            raise
    
    def _build_graph(self) -> StateGraph:
        """
        Construye el grafo del agente
        
        FLUJO:
        START → query_node → retrieve_node → reason_node → format_node → END
        """
        graph = StateGraph(dict)
        
        graph.add_node("query_node", query_node)
        graph.add_node("retrieve_node", retrieve_node)
        graph.add_node("reason_node", reason_node)
        graph.add_node("format_node", format_node)
        
        graph.add_edge(START, "query_node")
        graph.add_edge("query_node", "retrieve_node")
        graph.add_edge("retrieve_node", "reason_node")
        graph.add_edge("reason_node", "format_node")
        graph.add_edge("format_node", END)
        
        compiled_graph = graph.compile()
        logger.info("✓ Grafo compilado")
        return compiled_graph
    
    def execute(self, query: str) -> Dict[str, Any]:
        """Ejecuta el agente con una query"""
        try:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"EJECUTANDO AGENTE: {query[:50]}...")
            logger.info(f"{'=' * 60}")
            
            initial_state = {
                "query": query,
                "prepared_query": "",
                "query_analysis": {},
                "retrieved_documents": [],
                "llm_response": "",
                "citations": [],
                "reasoning_steps": [],
                "final_response": {}
            }
            
            result = self.graph.invoke(initial_state)
            
            logger.info("✓ Ejecución completada")
            
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


_agent = None


def get_agent() -> AgentStateGraph:
    """Obtiene la instancia global del agente"""
    global _agent
    if _agent is None:
        _agent = AgentStateGraph()
    return _agent


def run_agent_query(query: str) -> Dict[str, Any]:
    """Ejecuta una query en el agente"""
    agent = get_agent()
    return agent.execute(query)