"""
Utility functions for ONNX export.
"""

import warnings
from contextlib import contextmanager

import torch


@contextmanager
def suppress_onnx_export_warnings():
    """
    Context manager to suppress expected warnings during ONNX export.
    
    These warnings are expected during torch.jit.trace operations and don't 
    indicate actual problems with the ONNX export. They occur because the 
    transformers library uses dynamic control flow that can't be captured 
    in static ONNX graphs.
    
    Usage:
        with suppress_onnx_export_warnings():
            model = ORTModelForCausalLM.from_pretrained(
                model_name,
                export=True
            )
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
        warnings.filterwarnings('ignore', message='.*Converting a tensor to a Python boolean.*')
        warnings.filterwarnings('ignore', message='.*torch_dtype.*deprecated.*')
        yield
