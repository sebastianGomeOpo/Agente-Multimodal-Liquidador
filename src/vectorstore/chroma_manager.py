import chromadb
from chromadb.config import Settings
import logging
from typing import Optional, Dict, Any

# Configuración del logger para este módulo
logger = logging.getLogger(__name__)

class ChromaManager:
    """
    Gestiona la conexión y configuración de una colección en ChromaDB.
    Actúa como una capa de abstracción para inicializar, conectar
    y resetear la base de datos vectorial.
    """
    
    def __init__(self, persist_directory: str, collection_name: str, embedding_function: Optional[Any] = None, distance_metric: str = "cosine"):
        """
        Inicializa el cliente de ChromaDB en modo persistente.
        
        Args:
            persist_directory (str): Ruta al directorio donde se almacenarán los datos de ChromaDB.
            collection_name (str): Nombre de la colección a usar.
            embedding_function (Optional[Any]): La función de embedding a utilizar.
                                                 Para nuestro caso de CLIP, lo manejaremos externamente
                                                 y pasaremos 'None', ya que los vectores se generan antes.
            distance_metric (str): Métrica de distancia para la colección (ej. "cosine", "l2").
        """
        logger.info(f"Inicializando ChromaManager para el directorio: {persist_directory}")
        try:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False) # Deshabilitar telemetría
            )
            
            self.collection_name = collection_name
            self.distance_metric = distance_metric
            
            # Metadata para la creación de la colección
            collection_metadata = {"hnsw:space": self.distance_metric}
            
            # Obtener o crear la colección
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata=collection_metadata,
                # No pasamos embedding_function aquí si vamos a proveer vectores manualmente
            )
            
            logger.info(f"Colección '{self.collection_name}' cargada/creada exitosamente con métrica '{self.distance_metric}'.")

        except Exception as e:
            logger.error(f"Error al inicializar ChromaManager o conectar con la colección: {e}", exc_info=True)
            raise

    def get_collection(self) -> chromadb.Collection:
        """
        Devuelve el objeto de la colección de ChromaDB.
        
        Returns:
            chromadb.Collection: La instancia de la colección activa.
        """
        if not self.collection:
            logger.error("La colección no está inicializada. Llama a __init__ primero.")
            raise ConnectionError("La colección de ChromaDB no está disponible.")
        return self.collection

    def reset_database(self):
        """
        Resetea la base de datos eliminando todas las colecciones.
        ¡Usar con precaución!
        """
        logger.warning("Reseteando la base de datos completa...")
        try:
            self.client.reset()
            logger.info("Base de datos reseteada exitosamente.")
            # Volver a crear la colección después de resetear
            self.__init__(self.client.path, self.collection_name, distance_metric=self.distance_metric)
        except Exception as e:
            logger.error(f"Error al resetear la base de datos: {e}", exc_info=True)

    def delete_collection(self):
        """
        Elimina la colección específica gestionada por esta instancia.
        """
        logger.warning(f"Eliminando la colección '{self.collection_name}'...")
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Colección '{self.collection_name}' eliminada.")
            self.collection = None
        except Exception as e:
            logger.error(f"Error al eliminar la colección: {e}", exc_info=True)
            
    def get_collection_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual de la colección (ej. número de ítems).
        
        Returns:
            Dict[str, Any]: Un diccionario con el conteo de ítems.
        """
        try:
            count = self.collection.count()
            logger.debug(f"La colección '{self.collection_name}' tiene {count} ítems.")
            return {"item_count": count}
        except Exception as e:
            logger.error(f"Error al obtener el estado de la colección: {e}", exc_info=True)
            return {"item_count": -1}

# Ejemplo de uso (si se ejecuta este archivo directamente)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Probando ChromaManager...")
    
    # Usar un directorio temporal para la prueba
    test_persist_dir = "./chromadb_storage_test"
    test_collection_name = "test_collection"
    
    try:
        manager = ChromaManager(
            persist_directory=test_persist_dir,
            collection_name=test_collection_name
        )
        
        print(f"Cliente ChromaDB creado: {manager.client}")
        print(f"Colección obtenida: {manager.collection.name}")
        print(f"Estado inicial: {manager.get_collection_status()}")
        
        # Añadir un ítem de prueba (requiere embedding)
        # Como no configuramos una embedding_function, debemos proveer el embedding
        manager.collection.add(
            embeddings=[[0.1, 0.2, 0.3]], # Embedding de ejemplo
            documents=["Documento de prueba"],
            metadatas=[{"source": "test"}],
            ids=["test_id_1"]
        )
        
        print(f"Estado después de añadir: {manager.get_collection_status()}")
        
        # Limpiar
        manager.delete_collection()
        print("Colección de prueba eliminada.")
        
    except Exception as e:
        print(f"Error en la prueba: {e}")
    
    finally:
        # Eliminar el directorio de prueba si es necesario (limpieza manual por ahora)
        print(f"Prueba completada. Limpia el directorio '{test_persist_dir}' manualmente si es necesario.")