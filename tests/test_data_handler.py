"""
Unit tests for the ConversationDataHandler class.
"""

import unittest
import json
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly to avoid loading ML dependencies
from src.data_handler import ConversationDataHandler


class TestConversationDataHandler(unittest.TestCase):
    """Test cases for ConversationDataHandler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = ConversationDataHandler()
        self.sample_conversation = {
            "input": "What is Python?",
            "output": "Python is a programming language."
        }
    
    def test_init(self):
        """Test initialization."""
        self.assertIsInstance(self.handler, ConversationDataHandler)
        self.assertEqual(len(self.handler), 0)
    
    def test_add_conversation(self):
        """Test adding a single conversation."""
        self.handler.add_conversation(self.sample_conversation)
        self.assertEqual(len(self.handler), 1)
    
    def test_add_conversation_invalid(self):
        """Test adding an invalid conversation raises error."""
        with self.assertRaises(ValueError):
            self.handler.add_conversation({"invalid": "data"})
    
    def test_add_conversations(self):
        """Test adding multiple conversations."""
        conversations = [
            {"input": "Q1", "output": "A1"},
            {"input": "Q2", "output": "A2"},
        ]
        self.handler.add_conversations(conversations)
        self.assertEqual(len(self.handler), 2)
    
    def test_format_for_training(self):
        """Test formatting conversations for training."""
        self.handler.add_conversation(self.sample_conversation)
        formatted = self.handler.format_for_training()
        self.assertEqual(len(formatted), 1)
        self.assertIn("User:", formatted[0])
        self.assertIn("Assistant:", formatted[0])
    
    def test_get_batch(self):
        """Test getting a batch of conversations."""
        conversations = [
            {"input": f"Q{i}", "output": f"A{i}"}
            for i in range(10)
        ]
        self.handler.add_conversations(conversations)
        batch = self.handler.get_batch(batch_size=3, start_idx=2)
        self.assertEqual(len(batch), 3)
        self.assertEqual(batch[0]["input"], "Q2")
    
    def test_save_and_load_json(self):
        """Test saving and loading from JSON."""
        # Add data
        self.handler.add_conversation(self.sample_conversation)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.handler.save_to_json(temp_path)
            
            # Load into new handler
            new_handler = ConversationDataHandler()
            new_handler.load_from_json(temp_path)
            
            self.assertEqual(len(new_handler), 1)
            self.assertEqual(new_handler.conversations[0], self.sample_conversation)
        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main()
