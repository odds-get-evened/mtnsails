# API Reference - Programming Guide

This document provides a complete programming reference for developers who want to integrate MTN Sails into their applications or extend its functionality.

## Table of Contents

- [Programmatic Usage](#programmatic-usage)
- [Class Reference](#class-reference)
- [Data Format](#data-format)
- [Code Examples](#code-examples)
- [Extending the System](#extending-the-system)

## Programmatic Usage

### Basic Workflow

```python
from src.data_handler import ConversationDataHandler
from src.trainer import LLMTrainer
from src.onnx_converter import ONNXConverter
from src.chat_interface import ChatInterface

# 1. Prepare data
data_handler = ConversationDataHandler()
data_handler.add_conversation({
    "input": "What is Python?",
    "output": "Python is a programming language."
})

# 2. Train model
trainer = LLMTrainer(model_name="distilgpt2", device="cpu")
train_texts = data_handler.format_for_training()
trainer.train(train_texts, num_epochs=3, batch_size=4)
model_path = trainer.save_model()

# 3. Convert to ONNX
converter = ONNXConverter(model_path)
onnx_path = converter.convert_to_onnx("./onnx_model")

# 4. Use for chat
chat = ChatInterface(onnx_path, device="cpu")
response = chat.generate_response("Hello!")
print(response)
```

## Class Reference

### ConversationDataHandler

Manages conversation data loading and formatting.

**Methods:**

```python
# Initialize
data_handler = ConversationDataHandler()

# Load from JSON file
data_handler.load_from_json("conversations.json")

# Add single conversation
data_handler.add_conversation({
    "input": "Question here",
    "output": "Answer here"
})

# Format for training
train_texts = data_handler.format_for_training()

# Get batch of conversations
batch = data_handler.get_batch(start_idx=0, batch_size=10)

# Save to JSON
data_handler.save_to_json("output.json")
```

### LLMTrainer

Handles model fine-tuning on conversation data.

**Methods:**

```python
# Initialize with model
trainer = LLMTrainer(
    model_name="distilgpt2",  # or "gpt2", "microsoft/DialoGPT-small"
    device="cpu"              # or "cuda"
)

# Prepare dataset
dataset = trainer.prepare_dataset(
    texts=train_texts,
    max_length=256
)

# Train the model
trainer.train(
    train_texts=train_texts,
    num_epochs=3,
    batch_size=4,
    learning_rate=5e-5
)

# Save trained model
model_path = trainer.save_model(output_dir="./trained_model")

# Load a trained model
trainer.load_trained_model(model_path)
```

### ONNXConverter

Converts trained models to ONNX format.

**Methods:**

```python
# Initialize with model path
converter = ONNXConverter(model_path="./trained_model")

# Convert to ONNX
onnx_path = converter.convert_to_onnx(
    output_dir="./onnx_model",
    opset_version=14
)

# Verify ONNX model
is_valid = converter.verify_onnx_model(onnx_path)

# Get model information
info = converter.get_model_info()
print(info)
```

### ChatInterface

Provides chat functionality with ONNX models.

**Methods:**

```python
# Initialize with ONNX model
chat = ChatInterface(
    model_path="./onnx_model",
    device="cpu",
    max_length=256
)

# Generate single response
response = chat.generate_response(
    prompt="Your question here",
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2  # Prevents repetitive responses (default: 1.2)
)

# Interactive chat session
chat.chat()

# Batch generation
questions = ["Q1", "Q2", "Q3"]
responses = chat.batch_generate(
    prompts=questions,
    max_new_tokens=50
)

# With conversation logging
chat = ChatInterface(
    "./onnx_model",
    log_conversations=True,
    log_file="chat_history.json"
)
response = chat.generate_response("Hello!")

# Access conversation logs
logs = chat.get_conversation_log()

# Manually save logs
chat.save_conversation_log()
```

## Data Format

### Conversation JSON Format

```json
[
  {
    "input": "What is machine learning?",
    "output": "Machine learning is a subset of AI that enables systems to learn from data."
  },
  {
    "input": "How does it work?",
    "output": "It uses algorithms to identify patterns in data and make predictions."
  }
]
```

### Logged Conversation Format

When using conversation logging, the format includes timestamps:

```json
[
  {
    "input": "What is Python?",
    "output": "Python is a high-level programming language.",
    "timestamp": "2026-02-07T10:30:00"
  }
]
```

## Code Examples

### Example 1: Customer Support Bot

```python
from src.data_handler import ConversationDataHandler
from src.trainer import LLMTrainer
from src.onnx_converter import ONNXConverter
from src.chat_interface import ChatInterface

# Prepare customer support conversations
data = ConversationDataHandler()
data.load_from_json("support_conversations.json")

# Train on support data
trainer = LLMTrainer(model_name="distilgpt2")
trainer.train(data.format_for_training(), num_epochs=5)

# Deploy as ONNX
converter = ONNXConverter(trainer.save_model())
onnx_path = converter.convert_to_onnx("./support_bot")

# Use in production
chat = ChatInterface(onnx_path)
response = chat.generate_response("How do I reset my password?")
print(response)
```

### Example 2: Domain-Specific Assistant

```python
# Train on domain-specific data (e.g., medical, legal, technical)
data = ConversationDataHandler()
data.load_from_json("domain_data.json")

trainer = LLMTrainer(model_name="distilgpt2")
trainer.train(
    data.format_for_training(),
    num_epochs=10,  # More epochs for domain expertise
    batch_size=4
)

model_path = trainer.save_model("./domain_model")
```

### Example 3: Batch Processing

```python
chat = ChatInterface("./onnx_model")

questions = [
    "What is Python?",
    "What is machine learning?",
    "What is ONNX?"
]

responses = chat.batch_generate(questions, max_new_tokens=50)
for q, a in zip(questions, responses):
    print(f"Q: {q}\nA: {a}\n")
```

### Example 4: Using Logged Conversations for Retraining

```python
from src.chat_interface import ChatInterface
from src.data_handler import ConversationDataHandler
from src.trainer import LLMTrainer

# Enable logging during chat
chat = ChatInterface(
    "./onnx_model",
    log_conversations=True,
    log_file="my_chats.json"
)

# Chat naturally - logging happens in the background
response = chat.generate_response("Hello!")

# Later, use logged conversations for retraining
data = ConversationDataHandler()
data.load_from_json("my_chats.json")

trainer = LLMTrainer(model_name="distilgpt2")
trainer.train(data.format_for_training(), num_epochs=3)
```

### Example 5: Custom Training Parameters

```python
trainer = LLMTrainer(model_name="gpt2", device="cpu")

# Fine-tune with custom parameters
trainer.train(
    train_texts=train_texts,
    num_epochs=5,
    batch_size=2,
    learning_rate=3e-5
)

# Adjust generation parameters
chat = ChatInterface("./onnx_model")
response = chat.generate_response(
    "Your question",
    max_new_tokens=100,
    temperature=0.8,  # Higher = more creative
    top_p=0.95
)
```

## Extending the System

### Adding Custom Data Sources

Extend `ConversationDataHandler` to support different data formats:

```python
from src.data_handler import ConversationDataHandler

class CustomDataHandler(ConversationDataHandler):
    def load_from_csv(self, file_path):
        """Load conversations from CSV file"""
        import csv
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.add_conversation({
                    "input": row["question"],
                    "output": row["answer"]
                })
```

### Custom Training Configurations

Extend `LLMTrainer` for advanced training features:

```python
from src.trainer import LLMTrainer

class AdvancedTrainer(LLMTrainer):
    def train_with_validation(self, train_texts, val_texts):
        """Train with validation set"""
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_texts)
        val_dataset = self.prepare_dataset(val_texts)
        
        # Add validation logic to training loop
        # ... custom implementation
        pass
```

### Alternative Model Formats

Extend `ONNXConverter` for different export formats:

```python
from src.onnx_converter import ONNXConverter

class TensorRTConverter(ONNXConverter):
    def convert_to_tensorrt(self, output_path):
        """Convert ONNX model to TensorRT format"""
        # Convert ONNX to TensorRT
        # ... custom implementation
        pass
```

### Custom Chat Interfaces

Extend `ChatInterface` for specialized inference:

```python
from src.chat_interface import ChatInterface

class StreamingChatInterface(ChatInterface):
    def generate_response_streaming(self, prompt):
        """Generate response with token streaming"""
        # Implement streaming response
        # ... custom implementation
        pass
```

## Advanced Usage

### Multi-Model Ensemble

```python
# Load multiple models for ensemble inference
models = [
    ChatInterface("./onnx_model_1"),
    ChatInterface("./onnx_model_2"),
    ChatInterface("./onnx_model_3")
]

# Generate responses from all models
prompt = "What is machine learning?"
responses = [model.generate_response(prompt) for model in models]

# Combine or select best response
# ... custom logic
```

### Continuous Learning Pipeline

```python
import time
from src.data_handler import ConversationDataHandler
from src.trainer import LLMTrainer
from src.chat_interface import ChatInterface

def continuous_learning_loop():
    """Continuously retrain model with new data"""
    chat = ChatInterface("./onnx_model", log_conversations=True)
    
    while True:
        # Use model for inference (logs conversations)
        # ... user interactions
        
        # Periodically retrain
        time.sleep(3600)  # Wait 1 hour
        
        # Load new conversations
        data = ConversationDataHandler()
        data.load_from_json(chat.log_file)
        
        # Retrain model
        trainer = LLMTrainer(model_name="distilgpt2")
        trainer.train(data.format_for_training(), num_epochs=1)
        
        # Update model
        # ... redeploy logic
```

## Python Package Installation

If you want to use MTN Sails as a Python package:

```bash
# Install in development mode
pip install -e .

# Then import in your code
from mtnsails import ConversationDataHandler, LLMTrainer, ONNXConverter, ChatInterface
```

## Type Hints and IDE Support

All classes include comprehensive type hints for better IDE support:

```python
from typing import List, Dict, Optional
from src.data_handler import ConversationDataHandler

def process_data(file_path: str) -> List[str]:
    """Process conversation data with type hints"""
    handler: ConversationDataHandler = ConversationDataHandler()
    handler.load_from_json(file_path)
    texts: List[str] = handler.format_for_training()
    return texts
```

## Error Handling

```python
from src.chat_interface import ChatInterface

try:
    chat = ChatInterface("./onnx_model")
    response = chat.generate_response("Your prompt")
    print(response)
except FileNotFoundError:
    print("Model not found. Please train and convert a model first.")
except Exception as e:
    print(f"Error during inference: {e}")
```

## Performance Tips

### Memory Optimization

```python
# Use smaller batch sizes for limited RAM
trainer = LLMTrainer(model_name="distilgpt2")
trainer.train(train_texts, num_epochs=3, batch_size=2)

# Use distilgpt2 instead of larger models for CPU
trainer = LLMTrainer(model_name="distilgpt2", device="cpu")
```

### Inference Optimization

```python
# Use ONNX Runtime for faster inference
chat = ChatInterface("./onnx_model", device="cpu")

# Batch process for efficiency
prompts = ["Q1", "Q2", "Q3", "Q4", "Q5"]
responses = chat.batch_generate(prompts)
```

## Next Steps

- Review [README.md](README.md) for user guide and workflows
- Check [DEVELOPMENT.md](DEVELOPMENT.md) for architecture details
- See [QUICKSTART.md](QUICKSTART.md) for step-by-step tutorials
- Run [example.py](example.py) for a complete working example
