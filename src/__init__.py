"""
Paquete principal 'src' para el MultiDoc-Agent.

Este archivo __init__.py permite que Python trate el directorio 'src'
como un paquete, permitiendo importaciones absolutas como:

from src.preprocessors import excel_to_image
from src.agent import run_agent_query

Asegúrate de que la raíz del proyecto (el directorio que contiene 'src')
esté en tu PYTHONPATH.
"""

import logging

# Configurar un logger base para el paquete 'src'
# Los loggers de los módulos (ej. src.utils.logger) heredarán esto
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Opcionalmente, puedes exponer módulos clave aquí, pero es más limpio
# mantener las importaciones explícitas como están.
# Ejemplo (opcional):
# from . import utils
# from . import preprocessors
# from . import extractors
# from . import embeddings
# from . import vectorstore
# from . import agent