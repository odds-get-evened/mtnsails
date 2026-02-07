# MTN Sails - LLM Training and ONNX Conversion System

A streamlined, object-oriented Python system for training small language models on conversation data and converting them to ONNX format for efficient CPU inference.

## Features

- **OOP Design**: Clean, reusable classes for data handling, training, conversion, and chat
- **CPU-Friendly**: Uses small models (DistilGPT2) that run efficiently on CPU
- **ONNX Export**: Converts models to ONNX format using safetensors for optimized inference
- **Batch Training**: Train on batches of conversation data
- **Interactive Chat**: Chat interface for testing ONNX models

## Architecture

The system consists of four main components:

1. **ConversationDataHandler**: Manages conversation data loading and formatting
2. **LLMTrainer**: Handles model fine-tuning on conversation batches
3. **ONNXConverter**: Converts trained models to ONNX format
4. **ChatInterface**: Provides chat functionality with ONNX models

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Option 1: Run the Full Pipeline

Train a model, convert to ONNX, and test chat in one command:

```bash
python main.py pipeline --epochs 3 --batch-size 4
```

### Option 2: Step-by-Step

#### 1. Train a Model

```bash
# With your own data
python main.py train --data-file conversations.json --epochs 3

# With example data
python main.py train --epochs 3
```

#### 2. Convert to ONNX

```bash
python main.py convert --model-path ./trained_model --onnx-output ./onnx_model --verify
```

#### 3. Chat with the Model

```bash
# Interactive chat
python main.py chat --model-path ./onnx_model

# Single prompt
python main.py chat --model-path ./onnx_model --prompt "What is Python?"
```

### Option 3: Run the Example Script

```bash
python example.py
```

## Usage Examples

### Programmatic Usage

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

### Data Format

Conversations should be in JSON format:

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

## Command Reference

### Train Command

```bash
python main.py train [OPTIONS]

Options:
  --data-file PATH       Path to conversation data JSON
  --model-name NAME      Base model name (default: distilgpt2)
  --output-dir PATH      Output directory (default: ./trained_model)
  --device DEVICE        Device: cpu or cuda (default: cpu)
  --epochs N             Number of epochs (default: 3)
  --batch-size N         Batch size (default: 4)
  --learning-rate RATE   Learning rate (default: 5e-5)
```

### Convert Command

```bash
python main.py convert [OPTIONS]

Options:
  --model-path PATH      Path to trained model (required)
  --onnx-output PATH     ONNX output path (default: ./onnx_model)
  --opset-version N      ONNX opset version (default: 14)
  --verify               Verify ONNX model after conversion
```

### Chat Command

```bash
python main.py chat [OPTIONS]

Options:
  --model-path PATH      Path to ONNX model (required)
  --device DEVICE        Device: cpu or cuda (default: cpu)
  --max-length N         Max input length (default: 256)
  --max-tokens N         Max tokens to generate (default: 50)
  --prompt TEXT          Single prompt (non-interactive mode)
```

## Model Selection

The default model is **DistilGPT2** which is:
- Small enough to run on CPU (82M parameters)
- Fast inference
- Good for conversational tasks
- Based on GPT-2 architecture

You can use other small models like:
- `gpt2` (124M parameters)
- `distilgpt2` (82M parameters) - recommended for CPU
- `microsoft/DialoGPT-small` (117M parameters)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Optimum with ONNX Runtime
- 4GB+ RAM for CPU training

## Architecture Overview

```
┌─────────────────────────────────────────┐
│     ConversationDataHandler             │
│  - Load/save conversation data          │
│  - Format data for training             │
│  - Batch management                     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│          LLMTrainer                     │
│  - Load pre-trained model               │
│  - Fine-tune on conversation data       │
│  - Save model with safetensors          │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│        ONNXConverter                    │
│  - Load trained model                   │
│  - Convert to ONNX format               │
│  - Verify conversion                    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│        ChatInterface                    │
│  - Load ONNX model                      │
│  - Generate responses                   │
│  - Interactive chat                     │
└─────────────────────────────────────────┘
```

## License

GNU General Public License v3.0

## Contributing

Contributions are welcome! Please ensure code follows the OOP principles used in the project.
