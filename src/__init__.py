"""
MTN Sails - LLM Training and ONNX Conversion System
"""

__all__ = [
    'ConversationDataHandler',
    'LLMTrainer',
    'ONNXConverter',
    'ChatInterface'
]

# Lazy imports to avoid loading heavy ML dependencies until needed
def __getattr__(name):
    if name == 'ConversationDataHandler':
        from .data_handler import ConversationDataHandler
        return ConversationDataHandler
    elif name == 'LLMTrainer':
        from .trainer import LLMTrainer
        return LLMTrainer
    elif name == 'ONNXConverter':
        from .onnx_converter import ONNXConverter
        return ONNXConverter
    elif name == 'ChatInterface':
        from .chat_interface import ChatInterface
        return ChatInterface
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
