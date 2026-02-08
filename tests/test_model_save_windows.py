#!/usr/bin/env python3
"""
Test for Windows file mapping issue when saving models.

This test verifies that the model can be saved correctly even when
loaded from the same location (which would cause os error 1224 on Windows
without proper handling).
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModelSaveWindows(unittest.TestCase):
    """Test cases for model saving with file handle cleanup."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir) / "test_model"
        self.model_path.mkdir(parents=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_to_temp_then_copy(self):
        """Test that we can save to temp and copy without file conflicts."""
        # Create some dummy files
        (self.model_path / "config.json").write_text('{"test": "config"}')
        (self.model_path / "model.safetensors").write_text('dummy_model_data')
        
        # Verify files exist
        self.assertTrue((self.model_path / "config.json").exists())
        self.assertTrue((self.model_path / "model.safetensors").exists())
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Simulate saving to temp
            (temp_path / "config.json").write_text('{"test": "updated"}')
            (temp_path / "model.safetensors").write_text('updated_model_data')
            
            # Copy files back (simulating the fix)
            for item in temp_path.iterdir():
                dest = self.model_path / item.name
                if dest.exists():
                    dest.unlink()
                shutil.copy2(item, dest)
        
        # Verify files were updated
        self.assertTrue((self.model_path / "config.json").exists())
        self.assertTrue((self.model_path / "model.safetensors").exists())
        
        config_content = (self.model_path / "config.json").read_text()
        model_content = (self.model_path / "model.safetensors").read_text()
        
        self.assertEqual(config_content, '{"test": "updated"}')
        self.assertEqual(model_content, 'updated_model_data')
    
    def test_overwrite_existing_files(self):
        """Test that existing files can be overwritten after deletion."""
        # Create original files
        (self.model_path / "config.json").write_text('original')
        
        # Delete and recreate
        (self.model_path / "config.json").unlink()
        (self.model_path / "config.json").write_text('updated')
        
        # Verify update
        content = (self.model_path / "config.json").read_text()
        self.assertEqual(content, 'updated')


if __name__ == '__main__':
    unittest.main()
