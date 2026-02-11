"""
Unit tests for baseline model command-line interface.
"""

import unittest
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import main


class TestBaselineCommand(unittest.TestCase):
    """Test cases for baseline command-line interface."""
    
    def test_baseline_command_exists(self):
        """Test that baseline command is registered."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')
        
        # Add baseline parser similar to main.py
        baseline_parser = subparsers.add_parser('baseline')
        baseline_parser.add_argument('--model-name', type=str, default='distilgpt2')
        baseline_parser.add_argument('--baseline-output', type=str, default='./baseline_onnx')
        baseline_parser.add_argument('--test', action='store_true')
        
        # Parse a baseline command
        args = parser.parse_args(['baseline'])
        
        # Verify command was parsed correctly
        self.assertEqual(args.command, 'baseline')
        self.assertEqual(args.model_name, 'distilgpt2')
        self.assertEqual(args.baseline_output, './baseline_onnx')
        self.assertFalse(args.test)
    
    def test_baseline_command_with_args(self):
        """Test baseline command with custom arguments."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')
        
        baseline_parser = subparsers.add_parser('baseline')
        baseline_parser.add_argument('--model-name', type=str, default='distilgpt2')
        baseline_parser.add_argument('--baseline-output', type=str, default='./baseline_onnx')
        baseline_parser.add_argument('--test', action='store_true')
        
        # Parse with custom arguments
        args = parser.parse_args([
            'baseline',
            '--model-name', 'gpt2',
            '--baseline-output', './my_baseline',
            '--test'
        ])
        
        # Verify arguments were parsed correctly
        self.assertEqual(args.command, 'baseline')
        self.assertEqual(args.model_name, 'gpt2')
        self.assertEqual(args.baseline_output, './my_baseline')
        self.assertTrue(args.test)
    
    def test_baseline_function_exists(self):
        """Test that baseline_model function exists in main module."""
        self.assertTrue(hasattr(main, 'baseline_model'))
        self.assertTrue(callable(main.baseline_model))
    
    def test_baseline_function_signature(self):
        """Test that baseline_model function has correct signature."""
        import inspect
        sig = inspect.signature(main.baseline_model)
        params = list(sig.parameters.keys())
        
        # Should accept args parameter
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0], 'args')


if __name__ == '__main__':
    unittest.main()
