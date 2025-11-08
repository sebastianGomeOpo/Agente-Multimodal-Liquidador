"""
multimodal_indexer.py - Indexador multimodal con IDs estables
"""

from typing import List, Dict, Any, Optional

from src.utils.logger import get_logger
from src.utils.config import CHROMA_COLLECTION_NAME, EMBEDDING_DIMENSION
from src.vectorstore.chroma_manager import get_chroma_manager
from src.embeddings.clip_encoder import CLIPEncoder
from src.data_models import EmbeddingRecord, IndexingResult, RetrievalResult

logger = get_logger(__name__)


class MultimodalIndexer:
    """
    Gestiona la indexación y búsqueda de embeddings multimodales en ChromaDB.
    
    RESPONSABILIDADES:
    - Generar IDs estables basados en fuente + tipo + chunk/page
    - Indexar registros en formato EmbeddingRecord
    - Búsqueda semántica con filtros por tipo
    """
    
    def __init__(self, collection_name: str = CHROMA_COLLECTION_NAME):
        """Inicializa el indexador con la colección de ChromaDB"""
        try:
            self.manager = get_chroma_manager()
            self.collection = self.manager.get_collection()
            self.encoder = CLIPEncoder()  # Singleton
            logger.info(f"✓ MultimodalIndexer inicializado con colección '{collection_name}'")
        except Exception as e:
            logger.error(f"Error inicializando MultimodalIndexer: {e}")
            raise
    
    def _generate_stable_id(self, metadata: Dict[str, Any]) -> str:
        """
        Genera un ID estable basado SOLO en atributos inmutables.
        
        FORMATO:
        - excel_image: excel_{source_file}_{chunk}
        - pdf: pdf_{source_file}_p{page}
        - text: text_{source_file}
        
        Args:
            metadata: Metadatos del documento
        
        Returns:
            str: ID único y estable
        """
        doc_type = metadata.get("type", "unknown")
        source_file = metadata.get("source_file", "unknown")
        
        # Limpiar nombre de archivo (quitar extensión, normalizar)
        source_clean = source_file.replace(".xlsx", "").replace(".pdf", "").replace(".", "_")
        
        if doc_type == "excel_image":
            chunk = metadata.get("chunk", "r0_c0")
            return f"excel_{source_clean}_{chunk}"
        
        elif doc_type == "pdf":
            page = metadata.get("page", 0)
            return f"pdf_{source_clean}_p{page}"
        
        elif doc_type == "text":
            return f"text_{source_clean}"
        
        else:
            # Fallback (no debería usarse)
            return f"{doc_type}_{source_clean}"
    
    def index_batch(self, records: List[EmbeddingRecord]) -> IndexingResult:
        """
        Indexa un lote de registros en ChromaDB.
        
        Args:
            records: Lista de EmbeddingRecord
        
        Returns:
            IndexingResult: Resultado de la indexación
        """
        if not records:
            logger.warning("No hay registros para indexar")
            return IndexingResult(status="warning", indexed=0)
        
        ids: List[str] = []
        embeddings: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        errors: List[str] = []
        
        logger.info(f"Indexando {len(records)} registros...")
        
        for record in records:
            try:
                # Validar dimensión del embedding
                if len(record.embedding) != EMBEDDING_DIMENSION:
                    error_msg = f"Dimensión incorrecta: {len(record.embedding)} != {EMBEDDING_DIMENSION}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    continue
                
                # Generar ID estable
                doc_id = self._generate_stable_id(record.metadata)
                
                ids.append(doc_id)
                embeddings.append(record.embedding)
                documents.append(record.document)
                metadatas.append(record.metadata)
            
            except Exception as e:
                error_msg = f"Error procesando registro: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        if not ids:
            logger.error("Ningún registro válido para indexar")
            return IndexingResult(status="error", indexed=0, errors=errors)
        
        # Indexar en ChromaDB
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"✓ {len(ids)} registros indexados exitosamente")
            
            return IndexingResult(
                status="success",
                indexed=len(ids),
                ids=ids,
                errors=errors
            )
        
        except Exception as e:
            logger.error(f"Error indexando en ChromaDB: {e}")
            return IndexingResult(
                status="error",
                indexed=0,
                errors=errors + [str(e)]
            )
    
    def semantic_search(
        self,
        query_text: str,
        n_results: int = 5,
        doc_type: Optional[str] = None,
        min_similarity: float = 0.0
    ) -> List[RetrievalResult]:
        """
        Búsqueda semántica usando un texto de consulta.
        
        Args:
            query_text: Texto de la consulta
            n_results: Número máximo de resultados
            doc_type: Filtrar por tipo ("excel_image", "pdf", "text", None=todos)
            min_similarity: Similitud mínima (0-1)
        
        Returns:
            List[RetrievalResult]: Documentos recuperados
        """
        try:
            # Generar embedding de la query
            query_embedding = self.encoder.encode_text(query_text)
            if query_embedding is None:
                logger.error("No se pudo generar embedding para la query")
                return []
            
            return self.search_by_embedding(
                embedding=query_embedding.tolist(),
                n_results=n_results,
                doc_type=doc_type,
                min_similarity=min_similarity
            )
        
        except Exception as e:
            logger.error(f"Error en búsqueda semántica: {e}")
            return []
    
    def search_by_embedding(
        self,
        embedding: List[float],
        n_results: int = 5,
        doc_type: Optional[str] = None,
        min_similarity: float = 0.0
    ) -> List[RetrievalResult]:
        """
        Búsqueda directa usando un embedding ya calculado.
        
        Args:
            embedding: Vector de embedding
            n_results: Número máximo de resultados
            doc_type: Filtrar por tipo
            min_similarity: Similitud mínima (0-1)
        
        Returns:
            List[RetrievalResult]: Documentos recuperados ordenados por similitud
        """
        try:
            # Construir filtro where si se especifica tipo
            where_clause = {"type": doc_type} if doc_type else None
            
            # Realizar búsqueda
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
                where=where_clause,
                include=["metadatas", "documents", "distances"]
            )
            
            # Formatear resultados
            retrieved: List[RetrievalResult] = []
            
            if results and results.get("ids"):
                ids_list = results["ids"][0]
                distances_list = results["distances"][0]
                metadatas_list = results["metadatas"][0]
                documents_list = results["documents"][0]
                
                for i in range(len(ids_list)):
                    distance = distances_list[i]
                    similarity = max(0.0, 1.0 - distance)  # Convertir distancia a similitud
                    
                    # Filtrar por similitud mínima
                    if similarity < min_similarity:
                        continue
                    
                    retrieved.append(RetrievalResult(
                        id=ids_list[i],
                        distance=distance,
                        similarity=similarity,
                        metadata=metadatas_list[i],
                        document=documents_list[i]
                    ))
            
            logger.info(f"✓ Búsqueda devolvió {len(retrieved)} resultados")
            return retrieved
        
        except Exception as e:
            logger.error(f"Error en búsqueda por embedding: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la colección"""
        try:
            count = self.collection.count()
            return {
                "status": "success",
                "item_count": count,
                "collection_name": self.manager.collection_name
            }
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {"status": "error", "error": str(e), "item_count": 0}


def index_all_embeddings() -> IndexingResult:
    """
    PASO 6: Indexa todos los embeddings generados en el Paso 5.
    
    Returns:
        IndexingResult: Resultado consolidado de la indexación
    """
    logger.info("=" * 60)
    logger.info("PASO 6: Indexando embeddings en ChromaDB")
    logger.info("=" * 60)
    
    from pathlib import Path
    import json
    from src.utils.config import EMBEDDINGS_DIR
    
    indexer = MultimodalIndexer()
    
    all_records: List[EmbeddingRecord] = []
    
    # Cargar embeddings de imágenes
    image_file = EMBEDDINGS_DIR / "image_embeddings.json"
    if image_file.exists():
        with open(image_file, 'r', encoding='utf-8') as f:
            image_data = json.load(f)
            all_records.extend([EmbeddingRecord(**record) for record in image_data])
        logger.info(f"✓ Cargados {len(image_data)} embeddings de imágenes")
    else:
        logger.warning(f"No se encontró: {image_file}")
    
    # Cargar embeddings de texto
    text_file = EMBEDDINGS_DIR / "text_embeddings.json"
    if text_file.exists():
        with open(text_file, 'r', encoding='utf-8') as f:
            text_data = json.load(f)
            all_records.extend([EmbeddingRecord(**record) for record in text_data])
        logger.info(f"✓ Cargados {len(text_data)} embeddings de texto")
    else:
        logger.warning(f"No se encontró: {text_file}")
    
    if not all_records:
        logger.warning("No hay embeddings para indexar")
        return IndexingResult(status="warning", indexed=0)
    
    # Indexar todo en un lote
    logger.info(f"\nIndexando {len(all_records)} registros totales...")
    result = indexer.index_batch(all_records)
    
    logger.info("\n" + "=" * 60)
    logger.info(f"✓ PASO 6 COMPLETADO")
    logger.info(f"  - Indexados: {result.indexed}")
    logger.info(f"  - Errores: {len(result.errors)}")
    logger.info("=" * 60)
    
    return result