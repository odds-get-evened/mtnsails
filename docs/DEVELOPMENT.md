# Development Guide

## Architecture

### Object-Oriented Design

The system follows OOP principles for modularity and reusability:

```
┌────────────────────────────────┐
│  ConversationDataHandler       │
│  ─────────────────────────     │
│  + load_from_json()            │
│  + add_conversation()          │
│  + format_for_training()       │
│  + get_batch()                 │
│  + save_to_json()              │
└────────────────────────────────┘
                ↓
┌────────────────────────────────┐
│  LLMTrainer                    │
│  ─────────────────────────     │
│  + __init__(model_name)        │
│  + prepare_dataset()           │
│  + train(texts, epochs)        │
│  + save_model()                │
│  + load_trained_model()        │
└────────────────────────────────┘
                ↓
┌────────────────────────────────┐
│  ONNXConverter                 │
│  ─────────────────────────     │
│  + convert_to_onnx()           │
│  + verify_onnx_model()         │
│  + get_model_info()            │
└────────────────────────────────┘
                ↓
┌────────────────────────────────┐
│  ChatInterface                 │
│  ─────────────────────────     │
│  + generate_response()         │
│  + chat(interactive)           │
│  + batch_generate()            │
└────────────────────────────────┘
```

## Class Responsibilities

### ConversationDataHandler
- **Purpose**: Manages conversation data
- **Key Methods**:
  - `load_from_json()`: Load conversations from JSON file
  - `add_conversation()`: Add a single conversation
  - `format_for_training()`: Format data for model training
  - `get_batch()`: Retrieve batches for training
- **Use Cases**: Data preprocessing, batch management

### LLMTrainer
- **Purpose**: Fine-tune pre-trained language models
- **Key Methods**:
  - `__init__()`: Initialize with model name (default: distilgpt2)
  - `prepare_dataset()`: Tokenize and prepare training data
  - `train()`: Execute training loop with specified epochs
  - `save_model()`: Save model with safetensors format
- **Use Cases**: Model fine-tuning on custom conversations

### ONNXConverter
- **Purpose**: Convert PyTorch models to ONNX format
- **Key Methods**:
  - `convert_to_onnx()`: Export model to ONNX with Optimum
  - `verify_onnx_model()`: Test ONNX model inference
  - `get_model_info()`: Get model metadata
- **Use Cases**: Model optimization for deployment

### ChatInterface
- **Purpose**: Interactive chat with ONNX models
- **Key Methods**:
  - `generate_response()`: Generate single response
  - `chat()`: Start interactive session
  - `batch_generate()`: Generate multiple responses
- **Use Cases**: Production inference, testing

## Design Patterns Used

### 1. Single Responsibility Principle
Each class has one clear responsibility:
- Data handling
- Training
- Conversion
- Inference

### 2. Encapsulation
Internal implementation details are hidden:
```python
class LLMTrainer:
    def _load_model(self):  # Private method
        ...
    
    def train(self):  # Public API
        ...
```

### 3. Composition
Classes can be composed for complex workflows:
```python
data = ConversationDataHandler()
trainer = LLMTrainer()
converter = ONNXConverter(trainer.save_model())
chat = ChatInterface(converter.convert_to_onnx())
```

## Testing

### Unit Tests
Run tests for individual components:
```bash
python -m unittest tests.test_data_handler
```

### Validation
Run full validation without ML dependencies:
```bash
python validate.py
```

### Integration Test
Test the full pipeline:
```bash
python main.py pipeline --epochs 1 --batch-size 2
```

## Extending the System

### Adding New Data Sources

Extend `ConversationDataHandler`:
```python
class CustomDataHandler(ConversationDataHandler):
    def load_from_csv(self, file_path):
        # Custom implementation
        pass
```

### Custom Training Configurations

Extend `LLMTrainer`:
```python
class AdvancedTrainer(LLMTrainer):
    def train_with_validation(self, train_texts, val_texts):
        # Add validation logic
        pass
```

### Alternative Model Formats

Extend `ONNXConverter`:
```python
class TensorRTConverter(ONNXConverter):
    def convert_to_tensorrt(self):
        # Convert ONNX to TensorRT
        pass
```

## Performance Considerations

### CPU Optimization
- Uses `distilgpt2` (82M params) by default
- ONNX format for faster inference
- `torch.float32` for CPU compatibility

### Memory Management
- Batch processing for large datasets
- Gradient accumulation for limited RAM
- Model quantization (future enhancement)

### Inference Speed
- ONNX Runtime optimizations
- IO binding for tensor operations
- Cached model loading

## Best Practices

### 1. Model Selection
Choose models based on your hardware:
- CPU: distilgpt2, gpt2-small
- GPU: gpt2-medium, larger models

### 2. Training
- Start with small epochs (1-3)
- Use small batch sizes for CPU (2-4)
- Monitor loss during training

### 3. Data Preparation
- Clean and validate conversation data
- Balance question/answer lengths
- Use diverse training examples

### 4. ONNX Conversion
- Always verify converted models
- Test with sample inputs
- Check output quality

## Troubleshooting

### Out of Memory
- Reduce batch size
- Use smaller model
- Enable gradient checkpointing

### Slow Training
- Reduce max_length
- Use fewer epochs
- Consider GPU if available

### Poor Responses
- Increase training epochs
- Add more diverse data
- Adjust generation parameters (temperature, top_p)

## Future Enhancements

1. **Model Quantization**: Add INT8 quantization for faster inference
2. **Distributed Training**: Support multi-GPU training
3. **Fine-tuning Strategies**: LoRA, QLoRA adapters
4. **Web Interface**: Flask/FastAPI endpoint
5. **Model Zoo**: Pre-trained conversation models
6. **Evaluation Metrics**: BLEU, ROUGE scores
7. **Continuous Training**: Online learning from new conversations
