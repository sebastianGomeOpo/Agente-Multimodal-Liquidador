"""
test_excel_to_image.py - Tests de conversión Excel a imagen
"""

import pytest
from pathlib import Path

from src.preprocessors.excel_to_image import (
    excel_to_image,
    _is_temp_file,
    process_all_excels
)


class TestExcelToImage:
    """Tests para excel_to_image"""
    
    def test_excel_to_image_creates_chunks(self, mock_excel_file, temp_test_dir):
        """Test: Crea chunks de imagen correctamente"""
        with pytest.mock.patch('src.preprocessors.excel_to_image.EXCEL_IMAGES_DIR', temp_test_dir):
            result = excel_to_image(str(mock_excel_file))
        
        assert result["status"] == "success"
        assert "metadata" in result
        assert len(result["metadata"]["image_files"]) > 0
    
    def test_excel_to_image_handles_nonexistent_file(self, temp_test_dir):
        """Test: Maneja correctamente archivo inexistente"""
        with pytest.mock.patch('src.preprocessors.excel_to_image.EXCEL_IMAGES_DIR', temp_test_dir):
            result = excel_to_image(str(temp_test_dir / "nonexistent.xlsx"))
        
        assert result["status"] == "error"
        assert "no encontrado" in result["message"].lower()
    
    def test_is_temp_file_detects_temp_files(self):
        """Test: Detecta archivos temporales de Excel"""
        assert _is_temp_file(Path("~$document.xlsx")) is True
        assert _is_temp_file(Path("document.xlsx")) is False
        assert _is_temp_file(Path("~$~temp.xlsx")) is True
    
    def test_excel_chunk_metadata_includes_all_info(self, mock_excel_file, temp_test_dir):
        """Test: Metadata incluye toda la información necesaria"""
        with pytest.mock.patch('src.preprocessors.excel_to_image.EXCEL_IMAGES_DIR', temp_test_dir):
            result = excel_to_image(str(mock_excel_file))
        
        metadata = result["metadata"]
        assert "original_file" in metadata
        assert "image_files" in metadata
        assert "chunk_size" in metadata
        assert "timestamp" in metadata
        assert metadata["chunk_size"] == "50x50"


class TestProcessAllExcels:
    """Tests para process_all_excels"""
    
    def test_process_all_excels_filters_temp_files(self, temp_test_dir, mock_excel_file):
        """Test: Filtra archivos temporales"""
        # Crear un archivo temporal
        temp_file = temp_test_dir / "~$temp.xlsx"
        temp_file.touch()
        
        with pytest.mock.patch('src.preprocessors.excel_to_image.INPUT_EXCEL_DIR', temp_test_dir):
            with pytest.mock.patch('src.preprocessors.excel_to_image.EXCEL_IMAGES_DIR', temp_test_dir):
                results = process_all_excels()
        
        # No debe procesar el archivo temporal
        processed_files = [r.get("metadata", {}).get("original_file") for r in results if r.get("status") == "success"]
        assert "~$temp.xlsx" not in processed_files