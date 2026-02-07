"""
Unit tests for ChatInterface conversation logging functionality.
"""

import unittest
import json
import tempfile
import time
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestChatLogging(unittest.TestCase):
    """Test cases for ChatInterface logging functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "test_chat.json"
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('src.chat_interface.ORTModelForCausalLM')
    @patch('src.chat_interface.AutoTokenizer')
    def test_init_without_logging(self, mock_tokenizer, mock_model):
        """Test initialization without logging enabled."""
        from src.chat_interface import ChatInterface
        
        # Mock the model and tokenizer
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Create temp model directory
        model_path = Path(self.temp_dir) / "model"
        model_path.mkdir()
        
        chat = ChatInterface(
            str(model_path),
            log_conversations=False
        )
        
        self.assertFalse(chat.log_conversations)
        self.assertIsNone(chat.log_file)
        self.assertEqual(len(chat.conversation_log), 0)
        self.assertIsNone(chat.log_thread)
    
    @patch('src.chat_interface.ORTModelForCausalLM')
    @patch('src.chat_interface.AutoTokenizer')
    def test_init_with_logging(self, mock_tokenizer, mock_model):
        """Test initialization with logging enabled."""
        from src.chat_interface import ChatInterface
        
        # Mock the model and tokenizer
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Create temp model directory
        model_path = Path(self.temp_dir) / "model"
        model_path.mkdir()
        
        chat = ChatInterface(
            str(model_path),
            log_conversations=True,
            log_file=str(self.log_file)
        )
        
        self.assertTrue(chat.log_conversations)
        self.assertEqual(chat.log_file, self.log_file)
        self.assertIsNotNone(chat.log_thread)
        self.assertTrue(chat.log_thread.is_alive())
        
        # Cleanup
        chat.__del__()
    
    @patch('src.chat_interface.ORTModelForCausalLM')
    @patch('src.chat_interface.AutoTokenizer')
    def test_log_conversation(self, mock_tokenizer, mock_model):
        """Test logging a single conversation."""
        from src.chat_interface import ChatInterface
        
        # Mock the model and tokenizer
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Create temp model directory
        model_path = Path(self.temp_dir) / "model"
        model_path.mkdir()
        
        chat = ChatInterface(
            str(model_path),
            log_conversations=True,
            log_file=str(self.log_file)
        )
        
        # Log a conversation
        chat._log_conversation("Test prompt", "Test response")
        
        # Check in-memory log
        self.assertEqual(len(chat.conversation_log), 1)
        self.assertEqual(chat.conversation_log[0]["input"], "Test prompt")
        self.assertEqual(chat.conversation_log[0]["output"], "Test response")
        self.assertIn("timestamp", chat.conversation_log[0])
        
        # Cleanup
        chat.__del__()
    
    @patch('src.chat_interface.ORTModelForCausalLM')
    @patch('src.chat_interface.AutoTokenizer')
    def test_get_conversation_log(self, mock_tokenizer, mock_model):
        """Test getting conversation log."""
        from src.chat_interface import ChatInterface
        
        # Mock the model and tokenizer
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Create temp model directory
        model_path = Path(self.temp_dir) / "model"
        model_path.mkdir()
        
        chat = ChatInterface(
            str(model_path),
            log_conversations=True,
            log_file=str(self.log_file)
        )
        
        # Add conversations
        chat._log_conversation("Q1", "A1")
        chat._log_conversation("Q2", "A2")
        
        # Get log
        log = chat.get_conversation_log()
        
        self.assertEqual(len(log), 2)
        self.assertIsNot(log, chat.conversation_log)  # Should be a copy
        
        # Cleanup
        chat.__del__()
    
    @patch('src.chat_interface.ORTModelForCausalLM')
    @patch('src.chat_interface.AutoTokenizer')
    def test_clear_conversation_log(self, mock_tokenizer, mock_model):
        """Test clearing conversation log."""
        from src.chat_interface import ChatInterface
        
        # Mock the model and tokenizer
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Create temp model directory
        model_path = Path(self.temp_dir) / "model"
        model_path.mkdir()
        
        chat = ChatInterface(
            str(model_path),
            log_conversations=True,
            log_file=str(self.log_file)
        )
        
        # Add conversations
        chat._log_conversation("Q1", "A1")
        chat._log_conversation("Q2", "A2")
        
        # Clear
        chat.clear_conversation_log()
        
        self.assertEqual(len(chat.conversation_log), 0)
        self.assertEqual(chat._conversation_count, 0)
        
        # Cleanup
        chat.__del__()
    
    @patch('src.chat_interface.ORTModelForCausalLM')
    @patch('src.chat_interface.AutoTokenizer')
    def test_save_conversation_log(self, mock_tokenizer, mock_model):
        """Test manually saving conversation log."""
        from src.chat_interface import ChatInterface
        
        # Mock the model and tokenizer
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Create temp model directory
        model_path = Path(self.temp_dir) / "model"
        model_path.mkdir()
        
        chat = ChatInterface(
            str(model_path),
            log_conversations=True,
            log_file=str(self.log_file)
        )
        
        # Add conversations
        chat._log_conversation("Q1", "A1")
        chat._log_conversation("Q2", "A2")
        
        # Save manually
        chat.save_conversation_log()
        
        # Verify file exists and contains data
        self.assertTrue(self.log_file.exists())
        
        with open(self.log_file, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["input"], "Q1")
        self.assertEqual(data[1]["input"], "Q2")
        
        # Cleanup
        chat.__del__()
    
    @patch('src.chat_interface.ORTModelForCausalLM')
    @patch('src.chat_interface.AutoTokenizer')
    def test_async_save_with_flush(self, mock_tokenizer, mock_model):
        """Test async save with auto-flush."""
        from src.chat_interface import ChatInterface
        
        # Mock the model and tokenizer
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Create temp model directory
        model_path = Path(self.temp_dir) / "model"
        model_path.mkdir()
        
        chat = ChatInterface(
            str(model_path),
            log_conversations=True,
            log_file=str(self.log_file),
            auto_flush_interval=2  # Flush after 2 conversations
        )
        
        # Add conversations to trigger flush
        chat._log_conversation("Q1", "A1")
        chat._log_conversation("Q2", "A2")  # This should trigger flush
        
        # Wait for async save to complete
        time.sleep(2)
        
        # Verify file was created
        self.assertTrue(self.log_file.exists())
        
        with open(self.log_file, 'r') as f:
            data = json.load(f)
        
        self.assertGreaterEqual(len(data), 2)
        
        # Cleanup
        chat.__del__()
    
    @patch('src.chat_interface.ORTModelForCausalLM')
    @patch('src.chat_interface.AutoTokenizer')
    def test_compatibility_with_data_handler(self, mock_tokenizer, mock_model):
        """Test that logged format is compatible with ConversationDataHandler."""
        from src.chat_interface import ChatInterface
        from src.data_handler import ConversationDataHandler
        
        # Mock the model and tokenizer
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Create temp model directory
        model_path = Path(self.temp_dir) / "model"
        model_path.mkdir()
        
        chat = ChatInterface(
            str(model_path),
            log_conversations=True,
            log_file=str(self.log_file)
        )
        
        # Add conversations
        chat._log_conversation("What is Python?", "Python is a programming language.")
        chat._log_conversation("What is ML?", "Machine learning is AI.")
        
        # Save
        chat.save_conversation_log()
        
        # Load with ConversationDataHandler
        handler = ConversationDataHandler()
        handler.load_from_json(str(self.log_file))
        
        # Verify it loaded correctly
        self.assertEqual(len(handler), 2)
        self.assertEqual(handler.conversations[0]["input"], "What is Python?")
        self.assertEqual(handler.conversations[0]["output"], "Python is a programming language.")
        
        # Cleanup
        chat.__del__()


if __name__ == '__main__':
    unittest.main()
