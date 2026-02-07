"""
Unit tests for data quality validation functionality.
"""

import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_handler import ConversationDataHandler


class TestDataQualityValidation(unittest.TestCase):
    """Test cases for data quality validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = ConversationDataHandler()
    
    def test_repetition_detection(self):
        """Test detection of repetitive text."""
        # Highly repetitive text
        repetitive_text = "hello hello hello hello hello hello"
        self.assertTrue(self.handler._is_mostly_repetitive(repetitive_text))
        
        # Normal text
        normal_text = "This is a normal sentence with different words."
        self.assertFalse(self.handler._is_mostly_repetitive(normal_text))
    
    def test_short_text_detection(self):
        """Test detection of too-short text."""
        short_text = "hi"
        self.assertTrue(self.handler._is_too_short(short_text))
        
        normal_text = "This is a reasonable length response."
        self.assertFalse(self.handler._is_too_short(normal_text))
    
    def test_gibberish_detection(self):
        """Test detection of gibberish."""
        # Empty output
        self.assertTrue(self.handler._is_gibberish(""))
        
        # URL fragment only
        self.assertTrue(self.handler._is_gibberish("http://"))
        
        # Too many special characters
        self.assertTrue(self.handler._is_gibberish("!@#$%^&*()"))
        
        # Normal text
        self.assertFalse(self.handler._is_gibberish("This is normal text."))
    
    def test_echo_detection(self):
        """Test detection of echo responses."""
        # Direct echo
        input_text = "tell me about artificial intelligence"
        output_text = "tell me about artificial intelligence okay"
        self.assertTrue(self.handler._echos_input(input_text, output_text))
        
        # No echo
        input_text = "What is Python?"
        output_text = "It is a programming language."
        self.assertFalse(self.handler._echos_input(input_text, output_text))
    
    def test_validate_good_conversation(self):
        """Test validation of a good quality conversation."""
        conversation = {
            "input": "What is machine learning?",
            "output": "Machine learning is a subset of AI that enables computers to learn from data."
        }
        is_valid, issues = self.handler.validate_conversation_quality(conversation)
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)
    
    def test_validate_repetitive_conversation(self):
        """Test validation of a repetitive conversation."""
        conversation = {
            "input": "hello",
            "output": "hello hello hello hello hello hello hello"
        }
        is_valid, issues = self.handler.validate_conversation_quality(conversation)
        self.assertFalse(is_valid)
        self.assertIn("repetitive", ' '.join(issues).lower())
    
    def test_validate_empty_output(self):
        """Test validation of empty output."""
        conversation = {
            "input": "What is Python?",
            "output": ""
        }
        is_valid, issues = self.handler.validate_conversation_quality(conversation)
        self.assertFalse(is_valid)
        self.assertIn("empty", ' '.join(issues).lower())
    
    def test_validate_gibberish_output(self):
        """Test validation of gibberish output."""
        conversation = {
            "input": "tell me something",
            "output": "http://"
        }
        is_valid, issues = self.handler.validate_conversation_quality(conversation)
        self.assertFalse(is_valid)
        self.assertTrue(any('gibberish' in issue.lower() for issue in issues))
    
    def test_analyze_dataset_quality_good(self):
        """Test quality analysis on good dataset."""
        conversations = [
            {"input": "What is Python?", "output": "Python is a programming language."},
            {"input": "What is AI?", "output": "AI stands for artificial intelligence."},
            {"input": "What is ML?", "output": "ML is machine learning, a subset of AI."}
        ]
        self.handler.add_conversations(conversations)
        
        report = self.handler.analyze_dataset_quality()
        
        self.assertEqual(report['total_conversations'], 3)
        self.assertEqual(report['valid_conversations'], 3)
        self.assertEqual(report['invalid_conversations'], 0)
        self.assertGreaterEqual(report['quality_score'], 0.9)
    
    def test_analyze_dataset_quality_bad(self):
        """Test quality analysis on poor dataset."""
        conversations = [
            {"input": "hello", "output": ""},
            {"input": "test", "output": "test test test test test test"},
            {"input": "question", "output": "http://"}
        ]
        self.handler.add_conversations(conversations)
        
        report = self.handler.analyze_dataset_quality()
        
        self.assertEqual(report['total_conversations'], 3)
        self.assertLess(report['quality_score'], 0.5)
        self.assertGreater(report['invalid_conversations'], 0)
    
    def test_analyze_dataset_quality_mixed(self):
        """Test quality analysis on mixed quality dataset."""
        conversations = [
            {"input": "What is Python?", "output": "Python is a programming language."},
            {"input": "hello", "output": ""},
            {"input": "What is AI?", "output": "AI stands for artificial intelligence."},
            {"input": "test", "output": "test test test test test test"}
        ]
        self.handler.add_conversations(conversations)
        
        report = self.handler.analyze_dataset_quality()
        
        self.assertEqual(report['total_conversations'], 4)
        self.assertEqual(report['valid_conversations'], 2)
        self.assertEqual(report['invalid_conversations'], 2)
        self.assertAlmostEqual(report['quality_score'], 0.5, places=1)
    
    def test_analyze_empty_dataset(self):
        """Test quality analysis on empty dataset."""
        report = self.handler.analyze_dataset_quality()
        
        self.assertEqual(report['total_conversations'], 0)
        self.assertEqual(report['valid_conversations'], 0)
        self.assertEqual(report['quality_score'], 0.0)
    
    def test_repetition_score_calculation(self):
        """Test repetition score calculation."""
        # Highly repetitive
        text = "I am doing what I am doing what I am doing"
        score = self.handler._calculate_repetition_score(text)
        self.assertGreater(score, 0.3)
        
        # Not repetitive
        text = "Each word in this sentence is unique and meaningful"
        score = self.handler._calculate_repetition_score(text)
        self.assertLess(score, 0.2)
    
    def test_problematic_examples_in_report(self):
        """Test that problematic examples are included in report."""
        conversations = [
            {"input": "test1", "output": ""},
            {"input": "test2", "output": "good response here"},
            {"input": "test3", "output": "http://"}
        ]
        self.handler.add_conversations(conversations)
        
        report = self.handler.analyze_dataset_quality()
        
        self.assertIn('problematic_examples', report)
        self.assertGreater(len(report['problematic_examples']), 0)
        
        # Check that examples include index, input, output, and issues
        example = report['problematic_examples'][0]
        self.assertIn('index', example)
        self.assertIn('input', example)
        self.assertIn('output', example)
        self.assertIn('issues', example)


if __name__ == '__main__':
    unittest.main()
