"""
test_agent_nodes.py - Tests de los nodos del agente
"""

import pytest
from unittest.mock import patch, MagicMock

from src.agent.nodes import query_node, retrieve_node, reason_node, format_node


class TestQueryNode:
    """Tests para query_node"""
    
    def test_query_node_analyzes_numeric_question(self):
        """Test: Analiza correctamente preguntas numéricas"""
        state = {"query": "¿Cuánto tonelaje tiene la bodega B1?"}
        
        result = query_node(state)
        
        assert result["prepared_query"] == "¿Cuánto tonelaje tiene la bodega B1?"
        assert result["query_analysis"]["is_numeric"] is True
        assert "bodega" in result["query_analysis"]["entities"]
        assert result["query_analysis"]["search_strategy"] == "text_prioritized"
    
    def test_query_node_analyzes_temporal_question(self):
        """Test: Analiza correctamente preguntas temporales"""
        state = {"query": "¿Cuándo fue la operación de la nave X?"}
        
        result = query_node(state)
        
        assert result["query_analysis"]["is_temporal"] is True
        assert "nave" in result["query_analysis"]["entities"]
    
    def test_query_node_analyzes_descriptive_question(self):
        """Test: Analiza correctamente preguntas descriptivas"""
        state = {"query": "Describe la operación de la recalada R-2024-001"}
        
        result = query_node(state)
        
        assert result["query_analysis"]["is_numeric"] is False
        assert result["query_analysis"]["is_temporal"] is False
        assert result["query_analysis"]["search_strategy"] == "hybrid"
    
    def test_query_node_handles_empty_query(self):
        """Test: Maneja correctamente query vacía"""
        state = {"query": ""}
        
        result = query_node(state)
        
        assert "error" in result
        assert result["error"] == "Query vacía"
    
    def test_query_node_adds_reasoning_steps(self):
        """Test: Agrega pasos de razonamiento (CoT)"""
        state = {"query": "¿Cuánto tonelaje en B1?"}
        
        result = query_node(state)
        
        assert "reasoning_steps" in result
        assert len(result["reasoning_steps"]) > 0


class TestRetrieveNode:
    """Tests para retrieve_node"""
    
    @patch('src.agent.nodes.MultimodalIndexer')
    def test_retrieve_node_text_prioritized_strategy(self, mock_indexer_class, sample_embedding):
        """Test: Estrategia text_prioritized busca primero en texto"""
        mock_indexer = MagicMock()
        mock_indexer.semantic_search.return_value = [
            MagicMock(
                id="text_doc1",
                similarity=0.85,
                distance=0.15,
                metadata={"type": "text", "source_file": "doc1"},
                document="Content 1",
                dict=lambda: {
                    "id": "text_doc1",
                    "similarity": 0.85,
                    "distance": 0.15,
                    "metadata": {"type": "text", "source_file": "doc1"},
                    "document": "Content 1"
                }
            )
        ]
        mock_indexer_class.return_value = mock_indexer
        
        state = {
            "prepared_query": "¿Cuánto tonelaje?",
            "query_analysis": {
                "search_strategy": "text_prioritized",
                "is_numeric": True
            },
            "reasoning_steps": []
        }
        
        result = retrieve_node(state)
        
        # Verificar que se llamó semantic_search con doc_type="text"
        calls = mock_indexer.semantic_search.call_args_list
        assert any(call.kwargs.get("doc_type") == "text" for call in calls)
        
        assert len(result["retrieved_documents"]) > 0
    
    @patch('src.agent.nodes.MultimodalIndexer')
    def test_retrieve_node_hybrid_strategy(self, mock_indexer_class):
        """Test: Estrategia híbrida busca en todos los tipos"""
        mock_indexer = MagicMock()
        mock_indexer.semantic_search.return_value = []
        mock_indexer_class.return_value = mock_indexer
        
        state = {
            "prepared_query": "Describe la operación",
            "query_analysis": {
                "search_strategy": "hybrid"
            },
            "reasoning_steps": []
        }
        
        result = retrieve_node(state)
        
        # Verificar que se llamó sin filtro de tipo
        mock_indexer.semantic_search.assert_called_once()
        call_kwargs = mock_indexer.semantic_search.call_args.kwargs
        assert call_kwargs.get("doc_type") is None
    
    @patch('src.agent.nodes.MultimodalIndexer')
    def test_retrieve_node_sorts_by_similarity(self, mock_indexer_class):
        """Test: Ordena resultados por similitud"""
        mock_indexer = MagicMock()
        mock_indexer.semantic_search.return_value = [
            MagicMock(similarity=0.6, dict=lambda: {"similarity": 0.6}),
            MagicMock(similarity=0.9, dict=lambda: {"similarity": 0.9}),
            MagicMock(similarity=0.75, dict=lambda: {"similarity": 0.75})
        ]
        mock_indexer_class.return_value = mock_indexer
        
        state = {
            "prepared_query": "test",
            "query_analysis": {"search_strategy": "hybrid"},
            "reasoning_steps": []
        }
        
        result = retrieve_node(state)
        
        # Verificar orden descendente
        similarities = [doc["similarity"] for doc in result["retrieved_documents"]]
        assert similarities == sorted(similarities, reverse=True)


class TestReasonNode:
    """Tests para reason_node"""
    
    @patch('src.agent.nodes.ChatOpenAI')
    def test_reason_node_calls_llm(self, mock_llm_class, mock_llm_response):
        """Test: reason_node llama al LLM con contexto correcto"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_llm_response
        mock_llm_class.return_value = mock_llm
        
        state = {
            "prepared_query": "¿Cuánto tonelaje?",
            "retrieved_documents": [
                {
                    "id": "doc1",
                    "similarity": 0.85,
                    "metadata": {"type": "text", "source_file": "test.json"},
                    "document": "Bodega B1: 1234.5 toneladas"
                }
            ],
            "reasoning_steps": []
        }
        
        result = reason_node(state)
        
        assert "llm_response" in result
        assert mock_llm.invoke.called
    
    @patch('src.agent.nodes.ChatOpenAI')
    def test_reason_node_extracts_citations(self, mock_llm_class, mock_llm_response):
        """Test: Extrae citaciones del formato [Documento X]"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_llm_response
        mock_llm_class.return_value = mock_llm
        
        state = {
            "prepared_query": "test",
            "retrieved_documents": [
                {"id": "doc1", "similarity": 0.8, "metadata": {}, "document": "Content"}
            ],
            "reasoning_steps": []
        }
        
        result = reason_node(state)
        
        assert "citations" in result
        # mock_llm_response contiene [Documento 1] y [Documento 2]
        assert 1 in result["citations"]
        assert 2 in result["citations"]
    
    def test_reason_node_handles_no_documents(self):
        """Test: Maneja correctamente cuando no hay documentos"""
        state = {
            "prepared_query": "test",
            "retrieved_documents": [],
            "reasoning_steps": []
        }
        
        result = reason_node(state)
        
        assert "llm_response" in result
        assert "No encontré documentos relevantes" in result["llm_response"]


class TestFormatNode:
    """Tests para format_node"""
    
    def test_format_node_calculates_quality_score(self):
        """Test: Calcula quality score basado en similitudes"""
        state = {
            "llm_response": "Respuesta del LLM",
            "retrieved_documents": [
                {"id": "doc1", "similarity": 0.9, "metadata": {"type": "text", "source_file": "test"}},
                {"id": "doc2", "similarity": 0.8, "metadata": {"type": "text", "source_file": "test"}}
            ],
            "citations": [1, 2],
            "reasoning_steps": []
        }
        
        result = format_node(state)
        
        assert "final_response" in result
        assert result["final_response"]["quality_score"] > 0
        # Promedio de 0.9 y 0.8 = 0.85, más bonus por múltiples fuentes
        assert result["final_response"]["quality_score"] >= 0.85
    
    def test_format_node_penalizes_no_citations(self):
        """Test: Penaliza cuando no hay citaciones"""
        state = {
            "llm_response": "Respuesta sin citar fuentes",
            "retrieved_documents": [
                {"id": "doc1", "similarity": 0.9, "metadata": {"type": "text", "source_file": "test"}}
            ],
            "citations": [],  # Sin citaciones
            "reasoning_steps": []
        }
        
        result = format_node(state)
        
        # Score debería ser menor por falta de citaciones (penalización de 0.8)
        assert result["final_response"]["quality_score"] <= 0.9 * 0.8
    
    def test_format_node_formats_sources(self):
        """Test: Formatea fuentes correctamente"""
        state = {
            "llm_response": "Test",
            "retrieved_documents": [
                {
                    "id": "doc1",
                    "similarity": 0.85,
                    "metadata": {"type": "text", "source_file": "test.json"}
                }
            ],
            "citations": [1],
            "reasoning_steps": []
        }
        
        result = format_node(state)
        
        sources = result["final_response"]["sources"]
        assert len(sources) == 1
        assert sources[0]["id"] == "doc1"
        assert sources[0]["cited"] is True
        assert sources[0]["similarity"] == 0.85
    
    def test_format_node_includes_reasoning_steps(self):
        """Test: Incluye pasos de razonamiento en respuesta final"""
        state = {
            "llm_response": "Test",
            "retrieved_documents": [],
            "citations": [],
            "reasoning_steps": ["Paso 1", "Paso 2", "Paso 3"]
        }
        
        result = format_node(state)
        
        assert len(result["final_response"]["reasoning_steps"]) == 3