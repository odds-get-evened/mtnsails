# MTN Sails - LLM Training and ONNX Conversion System

A streamlined, object-oriented Python system for training small language models on conversation data and converting them to ONNX format for efficient CPU inference.

## Features

- **OOP Design**: Clean, reusable classes for data handling, training, conversion, and chat
- **CPU-Friendly**: Uses small models (DistilGPT2) that run efficiently on CPU
- **ONNX Export**: Converts models to ONNX format using safetensors for optimized inference
- **Batch Training**: Train on batches of conversation data
- **Interactive Chat**: Chat interface for testing ONNX models
- **Conversation Logging**: Optional async logging for collecting retraining data

## Quick Links

- **Getting Started**: See [Installation](#installation) and [Quick Start](#quick-start) below
- **Programming Guide**: See [API_REFERENCE.md](API_REFERENCE.md) for code examples and API documentation
- **Step-by-Step Tutorial**: See [QUICKSTART.md](QUICKSTART.md) for detailed walkthrough
- **Developer Guide**: See [DEVELOPMENT.md](DEVELOPMENT.md) for architecture and extension details

## Architecture

The system consists of four main components:

1. **ConversationDataHandler**: Manages conversation data loading and formatting
2. **LLMTrainer**: Handles model fine-tuning on conversation batches
3. **ONNXConverter**: Converts trained models to ONNX format
4. **ChatInterface**: Provides chat functionality with ONNX models

## Installation

```bash
# Clone the repository
git clone https://github.com/odds-get-evened/mtnsails.git
cd mtnsails

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Option 1: Run the Example

The fastest way to see MTN Sails in action:

```bash
python example.py
```

This will train a small model, convert to ONNX, and demonstrate chat functionality.

### Option 2: Run the Full Pipeline

Train a model, convert to ONNX, and test chat in one command:

```bash
python main.py pipeline --epochs 3 --batch-size 4
```

### Option 3: Interactive Step-by-Step

See [Step-by-Step Workflow](#workflow-how-to-use-mtn-sails) below for detailed instructions.

## Workflow: How to Use MTN Sails

### Step 1: Prepare Your Data

Create a JSON file with your conversation data. Each conversation should have an `input` and `output` field:

**File: my_conversations.json**
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

### Step 2: Train Your Model

Train a language model on your conversation data:

```bash
python main.py train --data-file my_conversations.json --epochs 3 --batch-size 4
```

**What happens:**
- Loads your conversation data
- Fine-tunes a DistilGPT2 model on your conversations
- Saves the trained model to `./trained_model` directory

**Training tips:**
- Start with 2-3 epochs for quick testing
- Use batch size 2-4 for CPU training
- Monitor the loss values (should decrease)

### Step 3: Convert to ONNX Format

Convert your trained model to ONNX for optimized inference:

```bash
python main.py convert --model-path ./trained_model --onnx-output ./onnx_model --verify
```

**What happens:**
- Converts PyTorch model to ONNX format
- Optimizes for CPU inference
- Verifies the conversion was successful
- Saves to `./onnx_model` directory

### Step 4: Use Your Model

#### Interactive Chat

Start a conversation with your trained model:

```bash
python main.py chat --model-path ./onnx_model
```

Type your questions and the model will respond. Type `exit` or `quit` to end the session.

#### Single Prompt Mode

Get a quick response without interactive mode:

```bash
python main.py chat --model-path ./onnx_model --prompt "What is Python?"
```

### Step 5: Retrain with New Data

As you use your model, you can collect new conversation data to improve it:

#### Enable Conversation Logging

```bash
python main.py chat --model-path ./onnx_model --log-conversations --log-file new_data.json
```

**What happens:**
- All your conversations are automatically saved to `new_data.json`
- Logging is asynchronous and doesn't slow down chat
- Data is saved in the same format as your training data

#### Retrain with Logged Conversations

Use your logged conversations to retrain and improve the model:

```bash
# Train on the new conversation data
python main.py train --data-file new_data.json --epochs 3

# Convert the updated model
python main.py convert --model-path ./trained_model --onnx-output ./onnx_model --verify

# Use the improved model
python main.py chat --model-path ./onnx_model
```

**Retraining tips:**
- Combine old and new data for best results
- Retrain periodically as you collect more conversations
- Use fewer epochs (1-3) when fine-tuning existing models

## Command Reference

### Train Command

Train a model on conversation data:

```bash
python main.py train [OPTIONS]
```

**Key Options:**
- `--data-file PATH` - Path to your conversation JSON file
- `--epochs N` - Number of training epochs (default: 3)
- `--batch-size N` - Batch size for training (default: 4)
- `--output-dir PATH` - Where to save trained model (default: ./trained_model)

**Example:**
```bash
python main.py train --data-file my_data.json --epochs 5 --batch-size 2
```

### Convert Command

Convert trained model to ONNX format:

```bash
python main.py convert [OPTIONS]
```

**Key Options:**
- `--model-path PATH` - Path to trained model (required)
- `--onnx-output PATH` - Where to save ONNX model (default: ./onnx_model)
- `--verify` - Verify the conversion worked correctly

**Example:**
```bash
python main.py convert --model-path ./trained_model --onnx-output ./my_onnx --verify
```

### Chat Command

Chat with your ONNX model:

```bash
python main.py chat [OPTIONS]
```

**Key Options:**
- `--model-path PATH` - Path to ONNX model (required)
- `--prompt TEXT` - Single prompt (non-interactive)
- `--log-conversations` - Enable logging for retraining
- `--log-file PATH` - Where to save logs (default: ./chat_history.json)
- `--max-tokens N` - Maximum response length (default: 50)

**Examples:**
```bash
# Interactive chat
python main.py chat --model-path ./onnx_model

# Single prompt
python main.py chat --model-path ./onnx_model --prompt "Hello!"

# With logging enabled
python main.py chat --model-path ./onnx_model --log-conversations
```

### Pipeline Command

Run the complete workflow in one command:

```bash
python main.py pipeline [OPTIONS]
```

Combines train, convert, and chat into a single command. Useful for quick testing.

## Common Use Cases

### Customer Support Bot

Train a model on your customer support conversations to create an automated assistant that can answer common questions.

### Domain-Specific Assistant

Train on specialized domain knowledge (medical, legal, technical documentation) to create an expert assistant for your field.

### Interactive Tutorial System

Create an educational assistant that can answer questions about specific topics based on your training material.

### FAQ Chatbot

Convert your FAQ documentation into an interactive chatbot that provides natural conversation responses.

## Model Selection

The default model is **DistilGPT2** which is:
- Small enough to run on CPU (82M parameters)
- Fast inference
- Good for conversational tasks
- Based on GPT-2 architecture

You can use other small models with the `--model-name` option:
- `distilgpt2` (82M parameters) - recommended for CPU
- `gpt2` (124M parameters)
- `microsoft/DialoGPT-small` (117M parameters)

**Example:**
```bash
python main.py train --model-name gpt2 --data-file my_data.json
```

## Best Practices

### Data Quality
- Use at least 20 diverse conversations for training
- Keep input/output lengths balanced
- Include various question types and scenarios
- Review your data for accuracy before training

### Training
- Start with 2-3 epochs for initial testing
- Monitor loss values during training (should decrease)
- Increase epochs if responses aren't satisfactory
- Use smaller batch sizes (2-4) for CPU training

### Generation Quality
- Lower `--max-tokens` (30-50) for concise responses
- Higher `--max-tokens` (100-200) for detailed responses
- Adjust based on your use case and response quality

### Retraining Workflow
1. Deploy your model with conversation logging enabled
2. Collect real user conversations over time
3. Review and filter the logged conversations
4. Combine with original training data
5. Retrain periodically (weekly/monthly)
6. Compare new model with old before deploying

## Troubleshooting

### Out of Memory Errors

Reduce batch size:
```bash
python main.py train --batch-size 2
```

Use the smaller DistilGPT2 model (default).

### Poor Response Quality

Train for more epochs:
```bash
python main.py train --epochs 10
```

Add more diverse training data to your JSON file.

### Slow Training

This is normal for CPU training. For faster training:
- Reduce batch size
- Use fewer epochs for testing
- Consider using a GPU if available (`--device cuda`)


## System Requirements

- **Python**: 3.8 or newer
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for models and dependencies
- **OS**: Linux, macOS, or Windows

## Further Reading

- **[API_REFERENCE.md](API_REFERENCE.md)** - Complete programming guide with code examples
- **[QUICKSTART.md](QUICKSTART.md)** - Detailed step-by-step tutorial
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Architecture and extension guide
- **[example.py](example.py)** - Working code example

## License

GNU General Public License v3.0

## Contributing

Contributions are welcome! Please ensure code follows the OOP principles used in the project.

For developers looking to extend or integrate MTN Sails, see [API_REFERENCE.md](API_REFERENCE.md) for detailed programming documentation.
