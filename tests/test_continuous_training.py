"""
Unit tests for continuous training functionality.
"""

import unittest
import tempfile
from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the function to test
from main import check_model_exists


class TestContinuousTraining(unittest.TestCase):
    """Test cases for continuous training feature."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_check_model_exists_no_directory(self):
        """Test check_model_exists returns False for non-existent directory."""
        result = check_model_exists(str(self.model_path / "nonexistent"))
        self.assertFalse(result)
    
    def test_check_model_exists_empty_directory(self):
        """Test check_model_exists returns False for empty directory."""
        result = check_model_exists(str(self.model_path))
        self.assertFalse(result)
    
    def test_check_model_exists_only_config(self):
        """Test check_model_exists returns False with only config.json."""
        (self.model_path / "config.json").write_text('{}')
        result = check_model_exists(str(self.model_path))
        self.assertFalse(result)
    
    def test_check_model_exists_config_and_safetensors(self):
        """Test check_model_exists returns True with config.json and model.safetensors."""
        (self.model_path / "config.json").write_text('{}')
        (self.model_path / "model.safetensors").write_text('dummy')
        result = check_model_exists(str(self.model_path))
        self.assertTrue(result)
    
    def test_check_model_exists_config_and_pytorch(self):
        """Test check_model_exists returns True with config.json and pytorch_model.bin."""
        (self.model_path / "config.json").write_text('{}')
        (self.model_path / "pytorch_model.bin").write_text('dummy')
        result = check_model_exists(str(self.model_path))
        self.assertTrue(result)
    
    def test_check_model_exists_config_and_both_models(self):
        """Test check_model_exists returns True with config.json and both model files."""
        (self.model_path / "config.json").write_text('{}')
        (self.model_path / "model.safetensors").write_text('dummy')
        (self.model_path / "pytorch_model.bin").write_text('dummy')
        result = check_model_exists(str(self.model_path))
        self.assertTrue(result)
    
    def test_check_model_exists_only_model_files(self):
        """Test check_model_exists returns False with model files but no config."""
        (self.model_path / "model.safetensors").write_text('dummy')
        (self.model_path / "pytorch_model.bin").write_text('dummy')
        result = check_model_exists(str(self.model_path))
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
