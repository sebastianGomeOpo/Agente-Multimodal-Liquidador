import logging
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np
import json
from pathlib import Path

# Importaciones de módulos del proyecto
from .chroma_manager import ChromaManager, get_chroma_manager # ¡Importación añadida!
# (Importar ClipEncoder desde el módulo de embeddings)
from ..embeddings.clip_encoder import CLIPEncoder 

# --- ¡NUEVAS IMPORTACIONES! ---
# Importar configs para instanciación por defecto Y para leer embeddings
from ..utils.config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    CHROMA_DISTANCE_METRIC,
    CLIP_MODEL_NAME,
    EMBEDDINGS_DIR # <- ¡Importante para leer los JSON!
)
# --- FIN DE NUEVAS IMPORTACIONES ---

# Configuración del logger para este módulo
logger = logging.getLogger(__name__)

class MultimodalIndexer:
    """
    Orquesta la indexación y búsqueda de documentos multimodales (imágenes y texto)
    en ChromaDB, utilizando un codificador CLIP para generar embeddings.
    
    Esta clase es el puente entre el codificador (CLIP) y la base de datos (ChromaDB).
    """
    
    def __init__(self, chroma_manager: Optional[ChromaManager] = None, clip_encoder: Optional[CLIPEncoder] = None):
        """
        Inicializa el indexador.
        
        Si no se proveen chroma_manager o clip_encoder, se crearán
        instancias por defecto usando la configuración del proyecto.
        
        Args:
            chroma_manager (Optional[ChromaManager]): Instancia del gestor de ChromaDB.
            clip_encoder (Optional[ClipEncoder]): Instancia del codificador CLIP.
        """
        if clip_encoder:
            self.encoder = clip_encoder
            logger.info("MultimodalIndexer inicializado con ClipEncoder provisto.")
        else:
            logger.info("No se proveyó ClipEncoder, creando instancia por defecto...")
            # Asumimos que ClipEncoder puede ser instanciado sin argumentos
            # o que usa CLIP_MODEL_NAME por defecto desde config.
            self.encoder = CLIPEncoder() 
            logger.info("Instancia de ClipEncoder por defecto creada.")
        
        if chroma_manager:
            self.manager = chroma_manager
            logger.info("MultimodalIndexer inicializado con ChromaManager provisto.")
        else:
            logger.info("No se proveyó ChromaManager, usando get_chroma_manager() por defecto...")
            # --- CORRECCIÓN ---
            # Usar la función factory para obtener la instancia global
            self.manager = get_chroma_manager() 
            # ------------------
            logger.info("Instancia de ChromaManager por defecto obtenida.")

        self.collection = self.manager.get_collection()
        logger.info("MultimodalIndexer inicializado y listo.")

    def _generate_unique_id(self, metadata: Dict[str, Any], doc_type: str) -> str:
        """
        Genera un ID único basado en la metadata para evitar duplicados.
        Ejemplo: "pdf_doc_final_page_3" o "excel_reporte_ventas_sheet_1"
        """
        doc_type = metadata.get("type", doc_type) # Usar el doc_type como fallback
        source = metadata.get("source", "unknown").replace(" ", "_").replace("/", "_")
        
        # Usar Path(source).stem para quitar extensiones (ej. .png, .json)
        source_stem = Path(source).stem
        
        page = metadata.get("page", 0)
        
        if doc_type == "image": # Tipo unificado para todas las imágenes
            return f"img_{source_stem}"
        elif doc_type == "text":
            return f"txt_{source_stem}"
        else:
            # Fallback simple
            return f"{doc_type}_{hash(metadata.get('source', ''))}"

    def index_document(self, document: Any, metadata: Dict[str, Any], doc_type: str):
        """
        Indexa un único documento (imagen o texto) en ChromaDB.
        Genera el embedding apropiado basado en el tipo.
        
        Args:
            document (Any): El documento a indexar. Puede ser un str (texto)
                            o una instancia de PIL.Image (imagen).
            metadata (Dict[str, Any]): Metadatos asociados (fuente, fecha, etc.).
            doc_type (str): El tipo de documento ('text' o 'image').
        """
        try:
            embedding = None
            document_content = None # El "documento" que ChromaDB almacena (puede ser texto)

            if doc_type == 'image' and isinstance(document, Image.Image):
                embedding = self.encoder.encode_image(document)
                # Almacenamos la ruta o un descriptor, no la imagen en sí
                document_content = f"Imagen: {metadata.get('source', 'imagen desconocida')}"
                metadata['type'] = 'image'
                
            elif doc_type == 'text' and isinstance(document, str):
                embedding = self.encoder.encode_text(document)
                document_content = document # Almacenamos el texto completo
                metadata['type'] = 'text'
                
            else:
                logger.warning(f"Tipo de documento no soportado o 'document' no coincide con 'doc_type': {doc_type}")
                return

            if embedding is None:
                logger.error("No se pudo generar el embedding.")
                return

            # Generar ID único
            doc_id = self._generate_unique_id(metadata, doc_type) # <- doc_type añadido

            # Añadir a ChromaDB
            self.collection.add(
                embeddings=[embedding.tolist()], # ChromaDB espera una lista de listas
                documents=[document_content],
                metadatas=[metadata],
                ids=[doc_id]
            )
            logger.info(f"Documento indexado exitosamente con ID: {doc_id}")

        except Exception as e:
            logger.error(f"Error al indexar documento: {e}", exc_info=True) # ID puede no estar definido aquí

    def index_batch(self, documents: List[Any], metadatas: List[Dict[str, Any]], doc_types: List[str]):
        """
        Indexa un lote de documentos multimodales.
        
        Args:
            documents (List[Any]): Lista de documentos (PIL.Image o str).
            metadatas (List[Dict[str, Any]]): Lista de metadatos.
            doc_types (List[str]): Lista de tipos ('image' o 'text').
        """
        if not (len(documents) == len(metadatas) == len(doc_types)):
            logger.error("Error en batch: Las listas de documentos, metadatos y tipos deben tener el mismo tamaño.")
            return

        embeddings_list = []
        documents_list = []
        metadatas_list = []
        ids_list = []

        for doc, meta, doc_type in zip(documents, metadatas, doc_types):
            try:
                embedding = None
                doc_content = None
                
                if doc_type == 'image' and isinstance(doc, Image.Image):
                    embedding = self.encoder.encode_image(doc)
                    doc_content = f"Imagen: {meta.get('source', 'imagen desconocida')}"
                    meta['type'] = 'image'
                
                elif doc_type == 'text' and isinstance(doc, str):
                    embedding = self.encoder.encode_text(doc)
                    doc_content = doc
                    meta['type'] = 'text'
                
                else:
                    logger.warning(f"Ítem de lote omitido: tipo no soportado {doc_type}")
                    continue

                if embedding is not None:
                    embeddings_list.append(embedding.tolist())
                    documents_list.append(doc_content)
                    metadatas_list.append(meta)
                    ids_list.append(self._generate_unique_id(meta, doc_type)) # <- doc_type añadido

            except Exception as e:
                logger.error(f"Error procesando ítem de lote: {meta.get('source', 'unknown')}. Error: {e}", exc_info=True)

        # Añadir el lote completo a ChromaDB si hay algo que añadir
        if ids_list:
            try:
                self.collection.add(
                    embeddings=embeddings_list,
                    documents=documents_list,
                    metadatas=metadatas_list,
                    ids=ids_list
                )
                logger.info(f"Lote de {len(ids_list)} documentos indexado exitosamente.")
            except Exception as e:
                logger.error(f"Error al añadir lote a ChromaDB: {e}", exc_info=True)

    def semantic_search(self, query_text: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> Dict[str, List[Any]]:
        """
        Realiza una búsqueda semántica multimodal.
        Codifica el texto de la consulta con CLIP y busca en ChromaDB.
        
        Args:
            query_text (str): La consulta de búsqueda en lenguaje natural.
            k (int): El número de resultados a devolver.
            filters (Optional[Dict[str, Any]]): Filtros de metadata para ChromaDB (ej. {"type": "image"}).
            
        Returns:
            Dict[str, List[Any]]: Un diccionario con los resultados de la búsqueda (documentos, metadatos, distancias).
        """
        try:
            # 1. Convertir la consulta de texto a un embedding multimodal
            query_embedding = self.encoder.encode_text(query_text)
            
            if query_embedding is None:
                logger.error("No se pudo generar embedding para la consulta.")
                return {}

            query_embeddings_list = [query_embedding.tolist()]

            # 2. Realizar la consulta en ChromaDB
            results = self.collection.query(
                query_embeddings=query_embeddings_list,
                n_results=k,
                where=filters if filters else {} # Aplicar filtros si se proveen
            )
            
            logger.info(f"Búsqueda semántica para '{query_text[:50]}' completada. Encontrados {len(results.get('ids', [[]])[0])} resultados.")
            
            # --- CORRECCIÓN ---
            # Los resultados de ChromaDB están anidados en una lista extra
            # (porque se puede consultar por múltiples embeddings a la vez).
            # Debemos aplanar la salida para el agente.
            
            if not results or not results.get('ids') or not results['ids'][0]:
                logger.info("La búsqueda no arrojó resultados.")
                return {"count": 0, "documents": []}

            # Aplanar la lista de resultados (tomamos el índice [0])
            ids = results['ids'][0]
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]

            formatted_results = []
            for id_val, doc, meta, dist in zip(ids, documents, metadatas, distances):
                formatted_results.append({
                    "id": id_val,
                    "document": doc,
                    "metadata": meta,
                    "distance": dist
                })

            return {"count": len(formatted_results), "documents": formatted_results}
            # --- FIN DE CORRECCIÓN ---

        except Exception as e:
            logger.error(f"Error durante la búsqueda semántica: {e}", exc_info=True)
            return {}

    # --- CÓDIGO AÑADIDO (Función 1) ---
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas detalladas de la colección,
        contando por tipo de documento en los metadatos.
        """
        logger.info(f"Obteniendo estadísticas de la colección '{self.collection.name}'...")
        try:
            stats = {"total_documents": 0, "image_documents": 0, "text_documents": 0, "multimodal": False}
            
            # count() es la forma más rápida de obtener el total
            stats["total_documents"] = self.collection.count()
            
            if stats["total_documents"] == 0:
                logger.info("La colección está vacía.")
                return stats

            # get() sin 'where' para obtener metadatos de todo
            # (Limitado a 5000 por seguridad, ajustar si es necesario)
            all_metadatas = self.collection.get(limit=5000, include=["metadatas"])['metadatas']

            for meta in all_metadatas:
                doc_type = meta.get('type', 'unknown')
                if doc_type == 'image':
                    stats['image_documents'] += 1
                elif doc_type == 'text':
                    stats['text_documents'] += 1
            
            stats['multimodal'] = stats['image_documents'] > 0 and stats['text_documents'] > 0
            
            logger.info(f"Estadísticas obtenidas: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error al obtener estadísticas de la colección: {e}", exc_info=True)
            return {"error": str(e)}

    # --- CÓDIGO AÑADIDO (Función 2) ---
    def list_documents(self, limit: int = 10, offset: int = 0, doc_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Lista los documentos actualmente en la colección.
        """
        try:
            where_filter = {}
            if doc_type:
                where_filter = {"type": doc_type}
                
            results = self.collection.get(
                limit=limit,
                offset=offset,
                where=where_filter,
                include=["metadatas", "documents"]
            )
            
            formatted_results = [
                {"id": id_val, "metadata": meta, "document": doc}
                for id_val, meta, doc in zip(results['ids'], results['metadatas'], results['documents'])
            ]
            
            return {"count": len(formatted_results), "documents": formatted_results}
        
        except Exception as e:
            logger.error(f"Error al listar documentos: {e}", exc_info=True)
            return {"error": str(e)}

# --- CÓDIGO AÑADIDO (Función 3) ---
def index_all_embeddings() -> Dict[str, Any]:
    """
    Función orquestadora (Paso 5)
    Carga los archivos JSON de embeddings generados por el Paso 4 (CLIP)
    y los indexa en ChromaDB.
    """
    logger.info("Iniciando PASO 5: Indexación de todos los embeddings...")
    
    try:
        indexer = MultimodalIndexer() # Obtiene instancia por defecto
        
        image_embeddings_file = EMBEDDINGS_DIR / "image_embeddings.json"
        text_embeddings_file = EMBEDDINGS_DIR / "text_embeddings.json"
        
        stats = {"images_indexed": 0, "texts_indexed": 0, "total_indexed": 0, "errors": []}
        
        # 1. Indexar Imágenes
        if image_embeddings_file.exists():
            with open(image_embeddings_file, 'r', encoding='utf-8') as f:
                image_data = json.load(f)
            
            logger.info(f"Procesando {len(image_data)} embeddings de imágenes...")
            img_embeddings, img_metadatas, img_ids, img_documents = [], [], [], []
            
            for key, data in image_data.items():
                try:
                    img_embeddings.append(data['embedding'])
                    meta = {"source": data['source'], "type": "image"}
                    img_metadatas.append(meta)
                    img_documents.append(f"Imagen de Excel: {data['source']}") # Documento es solo texto
                    img_ids.append(indexer._generate_unique_id(meta, "image"))
                except Exception as e:
                    logger.warning(f"Error procesando embedding de imagen {key}: {e}")
                    stats['errors'].append(f"img:{key} - {e}")

            if img_ids:
                indexer.collection.add(
                    embeddings=img_embeddings,
                    documents=img_documents,
                    metadatas=img_metadatas,
                    ids=img_ids
                )
                stats['images_indexed'] = len(img_ids)
                logger.info(f"{len(img_ids)} embeddings de imagen indexados.")
        else:
            logger.warning("No se encontró 'image_embeddings.json'.")

        # 2. Indexar Textos (JSONs Estructurados)
        if text_embeddings_file.exists():
            with open(text_embeddings_file, 'r', encoding='utf-8') as f:
                text_data = json.load(f)
                
            logger.info(f"Procesando {len(text_data)} embeddings de texto...")
            txt_embeddings, txt_metadatas, txt_ids, txt_documents = [], [], [], []
            
            for key, data in text_data.items():
                try:
                    txt_embeddings.append(data['embedding'])
                    # El 'content' es el JSON estructurado como string
                    doc_content = data['content'] 
                    # Los metadatos los extraemos del propio JSON
                    meta = {"source": data.get('source', 'unknown.json'), "type": "text"}
                    try:
                        # Intentar parsear el JSON para metadatos más ricos
                        content_json = json.loads(doc_content)
                        meta['numero_factura'] = content_json.get('numero_factura')
                        meta['cliente_nombre'] = content_json.get('cliente_nombre')
                    except:
                        pass # Si falla, solo usamos la fuente
                    
                    txt_metadatas.append(meta)
                    txt_documents.append(doc_content) # El documento es el JSON como string
                    txt_ids.append(indexer._generate_unique_id(meta, "text"))
                except Exception as e:
                    logger.warning(f"Error procesando embedding de texto {key}: {e}")
                    stats['errors'].append(f"txt:{key} - {e}")

            if txt_ids:
                indexer.collection.add(
                    embeddings=txt_embeddings,
                    documents=txt_documents,
                    metadatas=txt_metadatas,
                    ids=txt_ids
                )
                stats['texts_indexed'] = len(txt_ids)
                logger.info(f"{len(txt_ids)} embeddings de texto indexados.")
        else:
            logger.warning("No se encontró 'text_embeddings.json'.")

        stats['total_indexed'] = stats['images_indexed'] + stats['texts_indexed']
        logger.info(f"Indexación completada. Total: {stats['total_indexed']} documentos.")
        return stats

    except Exception as e:
        logger.error(f"Error fatal durante index_all_embeddings: {e}", exc_info=True)
        return {"error": str(e)}

# --- FIN DE CÓDIGO AÑADIDO ---


# Ejemplo de uso (requeriría mocks de ChromaManager y ClipEncoder)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Probando MultimodalIndexer (con mocks)...")
    
    # --- Mocking ---
    class MockClipEncoder:
        def encode_text(self, text: str) -> np.ndarray:
            print(f"MOCK: Codificando texto: '{text[:20]}...'")
            return np.random.rand(512) # Dimensión de embedding simulada
            
        def encode_image(self, image: Any) -> np.ndarray:
            print("MOCK: Codificando imagen...")
            return np.random.rand(512)

    class MockChromaCollection:
        def add(self, embeddings, documents, metadatas, ids):
            print(f"MOCK DB: Añadiendo {len(ids)} ítems. ID[0]: {ids[0]}")
        
        def query(self, query_embeddings, n_results, where):
            print(f"MOCK DB: Consultando {n_results} resultados con {len(query_embeddings)} embeddings.")
            return {
                'ids': [['mock_id_1', 'mock_id_2']],
                'documents': [['Documento mock 1', 'Imagen mock 2']],
                'metadatas': [[{'source': 'mock1.txt'}, {'source': 'mock2.png'}]],
                'distances': [[0.123, 0.456]]
            }
        
        def count(self):
            print("MOCK DB: Contando documentos.")
            return 2
        
        def get(self, limit, include, offset=None, where=None):
            print("MOCK DB: Obteniendo metadatos.")
            return {'metadatas': [{'type': 'image'}, {'type': 'text'}]}

    
    class MockChromaManager:
        def __init__(self, *args, **kwargs):
            self.collection = MockChromaCollection()
            
        def get_collection(self):
            return self.collection

    # --- Fin del Mocking ---
    
    # 1. Inicializar mocks
    mock_encoder = MockClipEncoder()
    mock_manager = MockChromaManager()
    
    # 2. Sobrescribir 'get_chroma_manager' para que devuelva el mock
    # (Esto es solo para el test __main__)
    # Asignamos directamente en el ámbito de módulo sin usar 'global'
    get_chroma_manager = lambda: mock_manager
    
    # 3. Inicializar el indexador (ahora usará los mocks)
    indexer = MultimodalIndexer(mock_manager, mock_encoder)
    
    # 4. Probar indexación
    print("\n--- Probando indexación ---")
    
    mock_image = Image.new('RGB', (100, 100), color = 'red')
    meta_image = {"source": "reporte.xlsx", "type": "excel_image"}
    mock_text = "Esto es el contenido de una liquidación de prueba."
    meta_text = {"source": "liquidacion.pdf", "page": 1, "type": "text"}
    
    indexer.index_batch(
        documents=[mock_image, mock_text],
        metadatas=[meta_image, meta_text],
        doc_types=['image', 'text']
    )
    
    # 5. Probar búsqueda
    print("\n--- Probando búsqueda ---")
    query = "¿Cuál es el total del reporte?"
    results = indexer.semantic_search(query_text=query, k=2)
    
    print("\nResultados de búsqueda:")
    print(json.dumps(results, indent=2))
    
    # 6. Probar estadísticas
    print("\n--- Probando estadísticas ---")
    stats = indexer.get_collection_stats()
    print("\nResultados de estadísticas:")
    print(json.dumps(stats, indent=2))