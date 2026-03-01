# MTN Sails

A CPU-friendly system for fine-tuning small language models on conversation data, converting them to ONNX format, and deploying them for efficient local inference.

## Features

- **Fine-tune DistilGPT-2** (or other GPT-2 variants) on your own conversation data
- **ONNX export** via [Optimum](https://github.com/huggingface/optimum) for 2–4× faster CPU inference
- **Continuous training** — automatically resumes from an existing checkpoint and uses a lower learning rate to prevent catastrophic forgetting
- **Data quality validation** with automatic filtering of low-quality conversations
- **Interactive chat interface** with optional conversation logging for future retraining
- **scrapyer integration** — convert raw scraped text files into training data
- **CLI and Python API** for every workflow step

## Requirements

- Python 3.8 or newer
- 4 GB RAM minimum (8 GB recommended)
- ~2 GB disk space for models and dependencies
- Linux, macOS, or Windows

## Installation

```bash
git clone https://github.com/odds-get-evened/mtnsails.git
cd mtnsails

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Quick Start

### 1. Prepare training data

Create a JSON file containing input/output conversation pairs:

```json
[
  {"input": "What is machine learning?", "output": "Machine learning is a subset of AI that learns from data."},
  {"input": "How do I install Python?", "output": "Download the installer from python.org and follow the setup wizard."}
]
```

### 2. Validate data quality (optional but recommended)

```bash
python main.py validate --data-file my_conversations.json
```

Add `--filter` to automatically remove low-quality conversations and save the cleaned data:

```bash
python main.py validate --data-file my_conversations.json --filter --output clean.json
```

### 3. Train

```bash
python main.py train \
  --data-file my_conversations.json \
  --output-dir ./trained_model \
  --epochs 3 \
  --batch-size 4
```

If a trained model already exists in `--output-dir`, training continues from that checkpoint at a lower learning rate (1e-5) to preserve existing knowledge.

### 4. Convert to ONNX

```bash
python main.py convert \
  --model-path ./trained_model \
  --onnx-output ./onnx_model \
  --verify
```

### 5. Chat

```bash
# Interactive mode
python main.py chat --model-path ./onnx_model

# Single prompt
python main.py chat --model-path ./onnx_model --prompt "Hello, how are you?"

# Log conversations for future retraining
python main.py chat --model-path ./onnx_model --log-conversations --log-file ./chat_history.json
```

### One-command pipeline

Run training, ONNX conversion, and a test chat in one step:

```bash
python main.py pipeline \
  --data-file my_conversations.json \
  --output-dir ./trained_model \
  --onnx-output ./onnx_model \
  --epochs 3
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `train` | Fine-tune a model on conversation data |
| `convert` | Export a trained model to ONNX format |
| `chat` | Chat with an ONNX model (interactive or single-prompt) |
| `validate` | Analyse and optionally filter training data quality |
| `pipeline` | Run train → convert → chat in sequence |
| `baseline` | Export the base model to ONNX without any fine-tuning |
| `reset` | Delete fine-tuned and ONNX model directories |

### `train`

```
python main.py train [OPTIONS]

  --data-file PATH      Path to conversation JSON file
  --model-name NAME     Base model (default: distilgpt2)
  --output-dir PATH     Where to save the trained model (default: ./trained_model)
  --device DEVICE       cpu or cuda (default: cpu)
  --epochs N            Training epochs (default: 3)
  --batch-size N        Batch size (default: 4)
  --learning-rate LR    Initial learning rate (default: 5e-5)
  --force               Skip data quality warnings
```

### `convert`

```
python main.py convert [OPTIONS]

  --model-path PATH     Path to trained model (required)
  --onnx-output PATH    ONNX output directory (default: ./onnx_model)
  --opset-version N     ONNX opset version (default: 14)
  --verify              Run a verification pass after conversion
```

### `chat`

```
python main.py chat [OPTIONS]

  --model-path PATH         Path to ONNX model (required)
  --device DEVICE           cpu or cuda (default: cpu)
  --max-length N            Maximum input length in tokens (default: 256)
  --max-tokens N            Maximum tokens to generate (default: 50)
  --prompt TEXT             Single prompt — non-interactive mode
  --log-conversations       Save chat turns to a JSON file
  --log-file PATH           Log file path (default: ./chat_history.json)
```

### `validate`

```
python main.py validate [OPTIONS]

  --data-file PATH      Path to conversation JSON file (required)
  --filter              Remove invalid conversations and save the result
  --output PATH         Output file for filtered data (default: <input>_filtered.json)
```

### `pipeline`

```
python main.py pipeline [OPTIONS]

  --data-file PATH      Path to conversation JSON file
  --model-name NAME     Base model (default: distilgpt2)
  --output-dir PATH     Trained model directory (default: ./trained_model)
  --onnx-output PATH    ONNX output directory (default: ./onnx_model)
  --device DEVICE       cpu or cuda (default: cpu)
  --epochs N            Training epochs (default: 3)
  --batch-size N        Batch size (default: 4)
  --learning-rate LR    Initial learning rate (default: 5e-5)
  --opset-version N     ONNX opset version (default: 14)
  --verify              Verify ONNX model after conversion
  --max-length N        Maximum input length (default: 256)
  --max-tokens N        Maximum tokens to generate (default: 50)
```

### `baseline`

```
python main.py baseline [OPTIONS]

  --model-name NAME         Base model to export (default: distilgpt2)
  --baseline-output PATH    Output directory (default: ./baseline_onnx)
  --test                    Run a sample generation after export
```

### `reset`

```
python main.py reset [OPTIONS]

  --output-dir PATH     Trained model directory to delete (default: ./trained_model)
  --onnx-output PATH    ONNX model directory to delete (default: ./onnx_model)
  --force               Skip confirmation prompt
```

## Python API

```python
from src.data_handler import ConversationDataHandler
from src.trainer import LLMTrainer
from src.onnx_converter import ONNXConverter
from src.chat_interface import ChatInterface

# 1. Load and inspect data
data = ConversationDataHandler()
data.load_from_json("conversations.json")
report = data.analyze_dataset_quality()
print(f"Quality score: {report['quality_score']:.1%}")

# 2. Train
trainer = LLMTrainer(model_name="distilgpt2", output_dir="./trained", device="cpu")
trainer.train(data.format_for_training(), num_epochs=3, batch_size=4)
model_path = trainer.save_model()

# 3. Convert to ONNX
converter = ONNXConverter(model_path)
onnx_path = converter.convert_to_onnx("./onnx")
converter.verify_onnx_model(onnx_path)

# 4. Chat
chat = ChatInterface(onnx_path)
response = chat.generate_response("Hello!", max_new_tokens=50)
print(response)
```

## Processing Scraped Content

Use `process_scraped_content.py` to convert plain-text files scraped with [scrapyer](https://github.com/odds-get-evened/scrapyer) into training data:

```bash
python process_scraped_content.py /path/to/scraped/files --output chat_data.json
```

Custom prompt templates are supported:

```bash
python process_scraped_content.py /path/to/scraped/files \
  --prompt-template "What is {topic} from {source}?" \
  --output chat_data.json
```

## Architecture

```
mtnsails/
├── main.py                    # CLI entry point
├── process_scraped_content.py # Scraped-content → training data converter
├── validate.py                # Project structure and import validation
├── src/
│   ├── data_handler.py        # ConversationDataHandler — load, validate, format data
│   ├── trainer.py             # LLMTrainer — fine-tune and save models
│   ├── onnx_converter.py      # ONNXConverter — export and verify ONNX models
│   ├── chat_interface.py      # ChatInterface — ONNX inference and conversation logging
│   └── onnx_utils.py          # Shared ONNX utility helpers
├── examples/
│   ├── example.py             # End-to-end demo script
│   └── example_conversations.json
├── tests/                     # Unit and integration tests
├── docs/                      # Extended documentation
│   ├── QUICKSTART.md
│   ├── API_REFERENCE.md
│   ├── DEVELOPMENT.md
│   ├── DATA_QUALITY_GUIDE.md
│   ├── CONTINUOUS_TRAINING.md
│   └── SCRAPYER_INTEGRATION.md
├── requirements.txt
└── setup.py
```

## Continuous Training

Every time `train` or `pipeline` is run, the system checks whether a trained model already exists at `--output-dir`:

- **Model found** — continues fine-tuning from the checkpoint at a reduced learning rate (1e-5) to preserve previously learned knowledge.
- **No model found** — trains from scratch using the specified base model and learning rate.

This makes it easy to incrementally improve the model as new conversation data becomes available.

## Data Quality

The `validate` command scores your dataset on a scale of 0–100 %:

| Score | Status | Recommendation |
|-------|--------|----------------|
| ≥ 70 % | ✅ Good | Proceed with training |
| 50–70 % | ⚠️ Acceptable | Consider filtering problematic conversations |
| < 50 % | ❌ Critical | Filter or replace data before training |

Common issues detected: empty responses, repetitive text, input echoing, and very short outputs.

## Testing

```bash
# Run unit tests
python -m pytest tests/

# Validate project structure and imports (no ML deps required)
python validate.py

# Run the end-to-end demo
python examples/example.py
```

## License

[GPL-3.0](LICENSE)
