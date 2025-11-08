import logging
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np

# Importaciones de módulos del proyecto
from .chroma_manager import ChromaManager
from ..embeddings.clip_encoder import ClipEncoder # Dependencia clave

# --- ¡NUEVAS IMPORTACIONES! ---
# Importar configs para instanciación por defecto
from ..utils.config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    CHROMA_DISTANCE_METRIC,
    CLIP_MODEL_NAME
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
    
    # --- ¡CONSTRUCTOR MODIFICADO! ---
    def __init__(self, chroma_manager: Optional[ChromaManager] = None, clip_encoder: Optional[ClipEncoder] = None):
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
            self.encoder = ClipEncoder() 
            logger.info("Instancia de ClipEncoder por defecto creada.")
        
        if chroma_manager:
            self.manager = chroma_manager
            logger.info("MultimodalIndexer inicializado con ChromaManager provisto.")
        else:
            logger.info("No se proveyó ChromaManager, creando instancia por defecto...")
            # Usamos str() en el Path por si ChromaDB lo requiere
            self.manager = ChromaManager(
                persist_directory=str(CHROMA_PERSIST_DIR),
                collection_name=CHROMA_COLLECTION_NAME,
                distance_metric=CHROMA_DISTANCE_METRIC
            )
            logger.info("Instancia de ChromaManager por defecto creada.")

        self.collection = self.manager.get_collection()
        logger.info("MultimodalIndexer inicializado y listo.")
    # --- FIN DE CONSTRUCTOR MODIFICADO ---

    def _generate_unique_id(self, metadata: Dict[str, Any]) -> str:
        """
        Genera un ID único basado en la metadata para evitar duplicados.
        Ejemplo: "pdf_doc_final_page_3" o "excel_reporte_ventas_sheet_1"
        """
        doc_type = metadata.get("type", "unknown")
        source = metadata.get("source", "unknown").replace(" ", "_").replace("/", "_")
        page = metadata.get("page", 0)
        
        if doc_type == "pdf_image":
            return f"pdf_{source}_p{page}"
        elif doc_type == "excel_image":
            return f"excel_{source}"
        elif doc_type == "text":
            return f"text_{source}"
        else:
            # Fallback simple basado en el hash del documento (si está disponible)
            return f"{doc_type}_{hash(metadata.get('document', ''))}"

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
            doc_id = self._generate_unique_id(metadata)

            # Añadir a ChromaDB
            self.collection.add(
                embeddings=[embedding.tolist()], # ChromaDB espera una lista de listas
                documents=[document_content],
                metadatas=[metadata],
                ids=[doc_id]
            )
            logger.info(f"Documento indexado exitosamente con ID: {doc_id}")

        except Exception as e:
            logger.error(f"Error al indexar documento (ID: {doc_id}): {e}", exc_info=True)

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
                    ids_list.append(self._generate_unique_id(meta))

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
            
            logger.info(f"Búsqueda semántica para '{query_text}' completada. Encontrados {len(results.get('ids', [[]])[0])} resultados.")
            return results

        except Exception as e:
            logger.error(f"Error durante la búsqueda semántica: {e}", exc_info=True)
            return {}

# Ejemplo de uso (requeriría mocks de ChromaManager y ClipEncoder)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Probando MultimodalIndexer (requiere mocks)...")
    
    # --- Mocking ---
    # En un escenario real, estas clases serían importadas y no mockeadas
    
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
    
    class MockChromaManager:
        def get_collection(self):
            return MockChromaCollection()

    # --- Fin del Mocking ---
    
    # 1. Inicializar mocks
    mock_encoder = MockClipEncoder()
    mock_manager = MockChromaManager()
    
    # 2. Inicializar el indexador
    indexer = MultimodalIndexer(mock_manager, mock_encoder)
    
    # 3. Probar indexación
    print("\n--- Probando indexación ---")
    
    # Simular una imagen PIL
    mock_image = Image.new('RGB', (100, 100), color = 'red')
    meta_image = {"source": "reporte.xlsx", "type": "excel_image"}
    
    # Simular texto
    mock_text = "Esto es el contenido de una liquidación de prueba."
    meta_text = {"source": "liquidacion.pdf", "page": 1, "type": "text"}
    
    indexer.index_batch(
        documents=[mock_image, mock_text],
        metadatas=[meta_image, meta_text],
        doc_types=['image', 'text']
    )
    
    # 4. Probar búsqueda
    print("\n--- Probando búsqueda ---")
    query = "¿Cuál es el total del reporte?"
    results = indexer.semantic_search(query_text=query, k=2)
    
    print("\nResultados de búsqueda:")
    print(results)