"""
ONNX Converter for exporting models to ONNX format.
"""

import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM
from pathlib import Path
from typing import Optional

# Suppress warnings from transformers, torch, and optimum during ONNX export
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')
warnings.filterwarnings('ignore', category=FutureWarning, module='optimum')
warnings.filterwarnings('ignore', category=FutureWarning, module='functools')
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')
warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
warnings.filterwarnings('ignore', message='.*pad_token_id.*')
warnings.filterwarnings('ignore', message='.*ONNX initializers.*')
warnings.filterwarnings('ignore', message='.*aten::index.*')


class ONNXConverter:
    """Converts trained models to ONNX format for efficient inference."""
    
    def __init__(self, model_path: str):
        """
        Initialize the ONNX converter.
        
        Args:
            model_path: Path to the trained model (with safetensors)
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
    
    def convert_to_onnx(
        self,
        output_path: str,
        opset_version: int = 14,
        use_io_binding: bool = True
    ) -> str:
        """
        Convert the model to ONNX format.
        
        Args:
            output_path: Path to save the ONNX model
            opset_version: ONNX opset version to use
            use_io_binding: Whether to use IO binding for optimization
            
        Returns:
            Path to the converted ONNX model
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Converting model from {self.model_path} to ONNX format...")
        print(f"Output path: {output_path}")
        
        try:
            # Suppress TracerWarnings during ONNX export
            # These warnings are expected during model tracing and don't indicate actual problems
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
                warnings.filterwarnings('ignore', message='.*Converting a tensor to a Python boolean.*')
                warnings.filterwarnings('ignore', message='.*torch_dtype.*deprecated.*')
                
                # Load and convert the model using Optimum
                model = ORTModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    export=True,
                    use_io_binding=use_io_binding
                )
            
            # Save the converted model
            model.save_pretrained(str(output_path))
            
            # Also copy the tokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            tokenizer.save_pretrained(str(output_path))
            
            print(f"Successfully converted model to ONNX format at {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"Error during conversion: {e}")
            raise
    
    def verify_onnx_model(self, onnx_model_path: str, test_text: str = "Hello, how are you?") -> bool:
        """
        Verify that the ONNX model works correctly.
        
        Args:
            onnx_model_path: Path to the ONNX model
            test_text: Test text for verification
            
        Returns:
            True if model works correctly
        """
        try:
            print(f"Verifying ONNX model at {onnx_model_path}")
            
            # Load the ONNX model
            model = ORTModelForCausalLM.from_pretrained(onnx_model_path)
            tokenizer = AutoTokenizer.from_pretrained(onnx_model_path)
            
            # Test inference
            inputs = tokenizer(test_text, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=10)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"Verification successful! Test output: {result}")
            return True
            
        except Exception as e:
            print(f"Verification failed: {e}")
            return False
    
    @staticmethod
    def get_model_info(model_path: str) -> dict:
        """
        Get information about the model.
        
        Args:
            model_path: Path to the model
            
        Returns:
            Dictionary with model information
        """
        model_path = Path(model_path)
        
        info = {
            'path': str(model_path),
            'exists': model_path.exists(),
            'files': []
        }
        
        if model_path.exists():
            info['files'] = [f.name for f in model_path.iterdir() if f.is_file()]
        
        return info
