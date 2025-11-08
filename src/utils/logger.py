# src/utils/logger.py - ACTUALIZAR

import logging
import sys
from pathlib import Path
from src.utils.config import LOG_LEVEL, LOG_FORMAT, LOG_FILE


def get_logger(name: str) -> logging.Logger:
    """
    Obtiene un logger configurado para un módulo específico
    CON SOPORTE UNICODE EN WINDOWS
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(LOG_FORMAT)
    
    # ============= FIX PARA WINDOWS =============
    # Forzar UTF-8 en la consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(formatter)
    
    # CRÍTICO: Forzar encoding UTF-8
    if hasattr(console_handler.stream, 'reconfigure'):
        console_handler.stream.reconfigure(encoding='utf-8')
    
    logger.addHandler(console_handler)
    # ============================================
    
    # Handler para archivo (siempre UTF-8)
    try:
        file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
        file_handler.setLevel(LOG_LEVEL)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"No se pudo configurar file handler: {e}")
    
    return logger