# Quick Start Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/odds-get-evened/mtnsails.git
cd mtnsails

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## 5-Minute Tutorial

### 1. Quick Test (No Training)
```bash
# Run validation to ensure everything is set up
python validate.py
```

### 2. Run the Example
```bash
# This will train a small model, convert to ONNX, and test chat
python example.py
```

### 3. Interactive Chat
After running the example, try the chat interface:
```bash
python main.py chat --model-path ./example_onnx_model
```

## Step-by-Step Workflow

### Step 1: Prepare Your Data

Create a JSON file with your conversations:
```json
[
  {
    "input": "What is machine learning?",
    "output": "Machine learning is a subset of AI..."
  },
  {
    "input": "How does it work?",
    "output": "It uses algorithms to learn from data..."
  }
]
```

### Step 2: Train the Model

```bash
python main.py train \
  --data-file my_conversations.json \
  --epochs 3 \
  --batch-size 4 \
  --output-dir ./my_model
```

**Expected Output:**
- Training progress with loss values
- Model saved to `./my_model/` with safetensors

### Step 3: Convert to ONNX

```bash
python main.py convert \
  --model-path ./my_model \
  --onnx-output ./my_onnx_model \
  --verify
```

**Expected Output:**
- ONNX model saved to `./my_onnx_model/`
- Verification test passed

### Step 4: Use for Chat

```bash
# Interactive mode
python main.py chat --model-path ./my_onnx_model

# Single prompt mode
python main.py chat \
  --model-path ./my_onnx_model \
  --prompt "Tell me about Python"
```

## One-Command Pipeline

Run everything in one go:
```bash
python main.py pipeline \
  --data-file my_conversations.json \
  --epochs 3 \
  --output-dir ./trained \
  --onnx-output ./onnx
```

## Python API Usage

```python
from src.data_handler import ConversationDataHandler
from src.trainer import LLMTrainer
from src.onnx_converter import ONNXConverter
from src.chat_interface import ChatInterface

# 1. Load data
data = ConversationDataHandler()
data.load_from_json("conversations.json")

# 2. Train
trainer = LLMTrainer(model_name="distilgpt2", device="cpu")
trainer.train(data.format_for_training(), num_epochs=3)
model_path = trainer.save_model("./trained")

# 3. Convert
converter = ONNXConverter(model_path)
onnx_path = converter.convert_to_onnx("./onnx")

# 4. Chat
chat = ChatInterface(onnx_path)
response = chat.generate_response("Hello!")
print(response)
```

## Customization Options

### Use a Different Model
```bash
python main.py train \
  --model-name gpt2 \
  --data-file data.json
```

### Adjust Training Parameters
```bash
python main.py train \
  --epochs 5 \
  --batch-size 2 \
  --learning-rate 3e-5
```

### Control Generation
```python
chat = ChatInterface("./onnx_model")
response = chat.generate_response(
    "Your question",
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.95
)
```

## Common Use Cases

### Use Case 1: Customer Support Bot
```python
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
```

### Use Case 2: Domain-Specific Assistant
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
```

### Use Case 3: Batch Processing
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

## Tips for Best Results

### Data Quality
- Use 20+ diverse conversations
- Keep input/output lengths balanced
- Include various question types

### Training
- Start with 2-3 epochs
- Monitor loss (should decrease)
- Increase epochs if needed

### Generation
- Lower temperature (0.5-0.7) for focused responses
- Higher temperature (0.8-1.0) for creative responses
- Adjust max_new_tokens based on desired length

## Troubleshooting

### "Out of Memory" Error
```bash
# Reduce batch size
python main.py train --batch-size 2

# Use smaller model
python main.py train --model-name distilgpt2
```

### Slow Training
```bash
# Reduce max sequence length
# Edit src/trainer.py, change max_length=256 to max_length=128
```

### Poor Response Quality
```bash
# Train longer
python main.py train --epochs 10

# Add more diverse training data
# Review and expand your conversations.json
```

## Next Steps

1. ✅ Complete this quick start
2. 📖 Read [README.md](README.md) for full documentation
3. 🔧 Check [DEVELOPMENT.md](DEVELOPMENT.md) for advanced topics
4. 🧪 Run tests with `python validate.py`
5. 🚀 Deploy your ONNX model in production

## Getting Help

- Check existing issues on GitHub
- Review the documentation
- Examine the example scripts

Happy training! 🚀
