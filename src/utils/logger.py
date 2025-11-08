"""
logger.py - Sistema centralizado de logging
"""

import logging
import sys
from pathlib import Path
from src.utils.config import LOG_LEVEL, LOG_FORMAT, LOG_FILE


def get_logger(name: str) -> logging.Logger:
    """
    Obtiene un logger configurado para un módulo específico
    
    Args:
        name: Nombre del módulo (__name__)
    
    Returns:
        logging.Logger: Logger configurado
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    # Evitar añadir múltiples handlers
    if logger.handlers:
        return logger
    
    # Formato de logs
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para archivo
    try:
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(LOG_LEVEL)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"No se pudo configurar file handler: {e}")
    
    return logger


# Logger global del proyecto
main_logger = get_logger(__name__)