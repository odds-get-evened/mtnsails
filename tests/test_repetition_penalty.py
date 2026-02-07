#!/usr/bin/env python3
"""
Test to verify that repetition_penalty parameter is properly configured.
This test validates the API signature without requiring ML dependencies.
"""

import unittest
import sys
from pathlib import Path
import inspect

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRepetitionPenalty(unittest.TestCase):
    """Test cases for repetition penalty parameter."""
    
    def test_generate_response_has_repetition_penalty_parameter(self):
        """Test that generate_response method has repetition_penalty parameter."""
        from src.chat_interface import ChatInterface
        
        # Get the signature of generate_response method
        sig = inspect.signature(ChatInterface.generate_response)
        params = sig.parameters
        
        # Check that repetition_penalty parameter exists
        self.assertIn('repetition_penalty', params, 
                     "generate_response should have repetition_penalty parameter")
        
        # Check that it has a default value
        param = params['repetition_penalty']
        self.assertTrue(param.default != inspect.Parameter.empty,
                       "repetition_penalty should have a default value")
        
        # Check that default value is reasonable (1.2 is our chosen default)
        self.assertEqual(param.default, 1.2,
                        "repetition_penalty default should be 1.2")
    
    def test_generate_response_parameter_order(self):
        """Test that parameters are in the correct order."""
        from src.chat_interface import ChatInterface
        
        # Get the signature of generate_response method
        sig = inspect.signature(ChatInterface.generate_response)
        param_names = list(sig.parameters.keys())
        
        # Expected parameter order
        expected_params = [
            'self',
            'prompt', 
            'max_new_tokens',
            'temperature',
            'top_p',
            'do_sample',
            'repetition_penalty'
        ]
        
        self.assertEqual(param_names, expected_params,
                        f"Parameter order should be {expected_params}")
    
    def test_repetition_penalty_docstring(self):
        """Test that generate_response docstring mentions repetition_penalty."""
        from src.chat_interface import ChatInterface
        
        docstring = ChatInterface.generate_response.__doc__
        self.assertIsNotNone(docstring, "generate_response should have a docstring")
        self.assertIn('repetition_penalty', docstring.lower(),
                     "Docstring should mention repetition_penalty parameter")


if __name__ == '__main__':
    unittest.main()
