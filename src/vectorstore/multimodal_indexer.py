"""
multimodal_indexer.py - Indexador multimodal (CLIP + ChromaDB)
"""

import hashlib
from typing import List, Dict, Any, Sequence, Optional

from src.utils.logger import get_logger
from src.utils.config import CHROMA_COLLECTION_NAME
from .chroma_manager import ChromaManager
from ..embeddings.clip_encoder import CLIPEncoder  # Import corregido (mayúsculas)

logger = get_logger(__name__)


class MultimodalIndexer:
    """
    Gestiona la indexación de embeddings multimodales (texto e imagen) en ChromaDB.
    Expone una API compatible con el pipeline actual:
      - index_batch(records)  donde records = [{embedding, document, metadata}, ...]
      - index_batch(embeddings, documents, metadatas)  (modo tradicional)
      - get_collection_stats()
      - semantic_search(query_text, ...)
      - search_by_embedding(embedding, ...)  (wrapper útil)
    """

    def __init__(self, collection_name: str = CHROMA_COLLECTION_NAME):
        self.manager = ChromaManager(collection_name=collection_name)
        self.collection = self.manager.collection
        try:
            self.encoder = CLIPEncoder()
        except Exception as e:
            logger.error(f"Error al inicializar CLIPEncoder: {e}")
            self.encoder = None

    # --------------------------
    # Helpers
    # --------------------------
    def _generate_unique_id(self, metadata: Dict[str, Any], document: Optional[str] = None) -> str:
        """
        Genera un ID único para un documento basado en sus metadatos.
        Mantiene IDs legibles para los tipos esperados y un fallback robusto.
        """
        doc_type = metadata.get("type", "unknown")
        source = str(metadata.get("source_file", "unknown_source"))
        # Normaliza el 'source' para evitar separadores problemáticos
        source_norm = source.replace("\\", "/").split("/")[-1] or "unknown_source"

        if doc_type == "pdf_image":
            page = metadata.get("page", 0)
            return f"pdf_{source_norm}_p{page}"

        if doc_type == "excel_image":
            chunk = metadata.get("chunk", "c0")
            return f"excel_{source_norm}_{chunk}"

        if doc_type == "text":
            doc_id = metadata.get("doc_id", "doc")
            return f"text_{source_norm}_{doc_id}"

        # Fallback: hash estable basado en metadatos clave + (opcionalmente) el contenido/document
        key_parts = [
            doc_type,
            source_norm,
            str(metadata.get("page", "")),
            str(metadata.get("chunk", "")),
            str(metadata.get("doc_id", "")),
            str(metadata.get("hash", "")),
            str(metadata.get("id", "")),
        ]
        # Evita depender de metadata['document']; usa 'document' explícito si se pasó
        if document:
            key_parts.append(str(document)[:256])  # limita el tamaño para evitar hashes enormes
        key = "|".join(key_parts)
        content_hash = hashlib.md5(key.encode("utf-8")).hexdigest()
        return f"{doc_type}_{source_norm}_{content_hash[:8]}"

    # --------------------------
    # Indexación
    # --------------------------
    def index_document(self, embedding: Sequence[float], document: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Indexa un único documento (texto o imagen) en ChromaDB.
        """
        try:
            doc_id = self._generate_unique_id(metadata, document)
            self.collection.add(
                ids=[doc_id],
                embeddings=[list(embedding)],
                documents=[document],
                metadatas=[metadata],
            )
            logger.info(f"Documento indexado con ID: {doc_id}")
            return {"status": "success", "indexed": 1, "ids": [doc_id]}
        except Exception as e:
            logger.error(f"Error al indexar documento {metadata.get('source_file')}: {e}")
            return {"status": "error", "error": str(e), "indexed": 0}

    def index_batch(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Modos soportados:
          1) index_batch(records)
             records = [
               {"embedding": [...], "document": "...", "metadata": {...}},
               ...
             ]
          2) index_batch(embeddings, documents, metadatas)
        """
        # Modo 1: lista de dicts
        if len(args) == 1 and isinstance(args[0], list):
            records = args[0]
            if records and isinstance(records[0], dict):
                try:
                    embeddings = []
                    documents = []
                    metadatas = []
                    for r in records:
                        emb = r.get("embedding")
                        doc = r.get("document")
                        meta = r.get("metadata")
                        if emb is None or doc is None or meta is None:
                            logger.warning("Registro inválido en index_batch; se requieren 'embedding', 'document' y 'metadata'. Saltando.")
                            continue
                        embeddings.append(list(emb))
                        documents.append(doc)
                        metadatas.append(meta)
                    return self._index_batch_triples(embeddings, documents, metadatas)
                except Exception as e:
                    logger.error(f"Error procesando records en index_batch: {e}")
                    return {"status": "error", "error": str(e), "indexed": 0}

        # Modo 2: tres listas paralelas
        if len(args) == 3:
            embeddings, documents, metadatas = args
            return self._index_batch_triples(embeddings, documents, metadatas)

        raise TypeError(
            "index_batch espera (records:list[dict]) o (embeddings, documents, metadatas)."
        )

    def _index_batch_triples(
        self,
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        ids: List[str] = []
        valid_embeddings: List[List[float]] = []
        valid_documents: List[str] = []
        valid_metadatas: List[Dict[str, Any]] = []

        try:
            for emb, doc, meta in zip(embeddings, documents, metadatas):
                try:
                    # Preserva 'type' específico (excel_image/pdf_image/text); no sobrescribir con 'image'
                    doc_id = self._generate_unique_id(meta, doc)
                    ids.append(doc_id)
                    valid_embeddings.append(list(emb))
                    valid_documents.append(doc)
                    valid_metadatas.append(meta)
                except Exception as inner_e:
                    logger.warning(
                        f"Error procesando documento para batch {meta.get('source_file')}: {inner_e}. Saltando."
                    )
        except Exception as e:
            logger.error(f"Error preparando lote para indexación: {e}")
            return {"status": "error", "error": str(e), "indexed": 0}

        if not ids:
            logger.warning("No hay documentos válidos para indexar en el lote.")
            return {"status": "warning", "indexed": 0}

        try:
            self.collection.add(
                ids=ids,
                embeddings=valid_embeddings,
                documents=valid_documents,
                metadatas=valid_metadatas,
            )
            logger.info(f"Lote de {len(ids)} documentos indexados exitosamente.")
            return {"status": "success", "indexed": len(ids), "ids": ids}
        except Exception as e:
            logger.error(f"Error al indexar lote en ChromaDB: {e}")
            return {"status": "error", "error": str(e), "indexed": 0}

    # --------------------------
    # Búsqueda
    # --------------------------
    def semantic_search(self, query_text: str, n_results: int = 5, doc_type: str = "all") -> Dict[str, Any]:
        """
        Realiza una búsqueda semántica usando un embedding de texto.
        Devuelve {"documents": [{id, distance, metadata, document}, ...]}
        """
        if not self.encoder:
            logger.error("Encoder no inicializado. No se puede realizar la búsqueda.")
            return {"documents": []}

        try:
            query_embedding = self.encoder.encode_text(query_text)
            if query_embedding is None:
                logger.error("No se pudo generar embedding para la query.")
                return {"documents": []}

            where_clause = {}
            if doc_type != "all":
                where_clause = {"type": doc_type}

            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where_clause,
                include=["metadatas", "documents", "distances", "ids"],
            )

            formatted_docs: List[Dict[str, Any]] = []
            if results and results.get("ids"):
                ids_list = results.get("ids", [[]])[0]
                distances_list = results.get("distances", [[]])[0]
                metadatas_list = results.get("metadatas", [[]])[0]
                documents_list = results.get("documents", [[]])[0]

                for i in range(len(ids_list)):
                    formatted_docs.append(
                        {
                            "id": ids_list[i],
                            "distance": distances_list[i],
                            "metadata": metadatas_list[i],
                            "document": documents_list[i],
                        }
                    )

            logger.info(f"Búsqueda semántica devolvió {len(formatted_docs)} resultados.")
            return {"documents": formatted_docs}
        except Exception as e:
            logger.error(f"Error durante la búsqueda semántica: {e}")
            return {"documents": []}

    def search_by_embedding(
        self,
        embedding: Sequence[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Búsqueda directa por embedding ya calculado (útil como wrapper).
        Devuelve {"documents": [{id, distance, metadata, document}, ...]}
        """
        try:
            results = self.collection.query(
                query_embeddings=[list(embedding)],
                n_results=n_results,
                where=where or {},
                include=["metadatas", "documents", "distances", "ids"],
            )

            formatted_docs: List[Dict[str, Any]] = []
            if results and results.get("ids"):
                ids_list = results.get("ids", [[]])[0]
                distances_list = results.get("distances", [[]])[0]
                metadatas_list = results.get("metadatas", [[]])[0]
                documents_list = results.get("documents", [[]])[0]

                for i in range(len(ids_list)):
                    formatted_docs.append(
                        {
                            "id": ids_list[i],
                            "distance": distances_list[i],
                            "metadata": metadatas_list[i],
                            "document": documents_list[i],
                        }
                    )

            logger.info(f"Búsqueda por embedding devolvió {len(formatted_docs)} resultados.")
            return {"documents": formatted_docs}
        except Exception as e:
            logger.error(f"Error durante la búsqueda por embedding: {e}")
            return {"documents": []}

    # --------------------------
    # Estadísticas
    # --------------------------
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la colección (conteo de ítems).
        Reutiliza el método del ChromaManager.
        """
        try:
            return self.manager.get_collection_status()
        except Exception as e:
            logger.error(f"Error al obtener estadísticas de la colección: {e}")
            return {"error": str(e), "item_count": 0}