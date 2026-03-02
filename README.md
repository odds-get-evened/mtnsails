# MTN Sails

A CPU-friendly system for fine-tuning small language models on your own conversation data and deploying them for efficient local inference.

## Features

- **Fine-tune DistilGPT-2** (or other GPT-2 variants) on your own conversation data
- **Optimized for CPU** via ONNX export for 2–4× faster inference without requiring a GPU
- **Continuous training** — automatically resumes from an existing checkpoint and uses a lower learning rate to prevent catastrophic forgetting
- **Data quality validation** with automatic detection and filtering of low-quality conversations
- **Interactive chat interface** with optional conversation logging for future retraining
- **Feedback mode** — human-in-the-loop approve/correct step during chat that writes labeled JSONL pairs for retraining
- **Background retraining daemon** — watches a JSONL feedback file and automatically retrains + re-exports the model once enough new examples accumulate
- **Taber bridge** — describe environmental forecasts in plain English and have the model translate them into structured predictions
- **scrapyer integration** — convert raw scraped text files into training-ready data

## Requirements

- Python 3.8 or newer
- 4 GB RAM minimum (8 GB recommended)
- Approximately 2 GB of free disk space for models and dependencies
- Linux, macOS, or Windows

## Installation

1. Clone the repository from GitHub.
2. Navigate into the project directory.
3. Install the required Python dependencies.

```bash
pip install -r requirements.txt

# Or install as an editable package
pip install -e .
```

## Getting Started

MTN Sails follows a straightforward workflow: prepare your data, validate its quality, train a model, convert it to ONNX format, and start chatting. Each step is described below.

### 1. Prepare Your Training Data

Training data must be formatted as a list of conversation pairs — each with an input (what the user says) and an output (how the assistant should respond). Store this data in a JSON file before proceeding.

### 2. Validate Data Quality

Before training, it is recommended to check the quality of your conversation data using the `validate` command. This scores your dataset, highlights any problematic conversations (such as empty responses, repetitive text, or inputs that simply echo the question back), and offers recommendations. You can also instruct it to automatically filter out low-quality entries and save the cleaned dataset to a new file.

```bash
mtnsails validate --data-file conversations.json

# Optionally filter and write a cleaned dataset
mtnsails validate --data-file conversations.json --filter --output-file conversations_filtered.json
```

### 3. Train Your Model

The `train` command fine-tunes a base language model on your conversation data. By default it uses DistilGPT-2, which runs comfortably on a CPU with minimal memory. You can configure how many training epochs to run, the batch size, and where to save the resulting model.

If a previously trained model already exists in the output directory, MTN Sails automatically continues from that checkpoint at a lower learning rate to preserve existing knowledge — no manual steps required.

```bash
# Train on your own dataset
mtnsails train --data-file conversations.json --epochs 3

# Train using the default example dataset (if provided by the repo)
mtnsails train --epochs 3
```

### 4. Convert to ONNX

The `convert` command exports your trained model to ONNX format, which enables significantly faster inference on CPU hardware. An optional verification step confirms the converted model produces correct output.

```bash
mtnsails convert --model-path ./trained_model --onnx-output ./onnx_model --verify
```

### 5. Chat with Your Model

The `chat` command opens an interactive session with your ONNX model. You can type messages and receive responses in real time, or supply a single prompt for a one-off query. Conversation logs can optionally be saved to a file and reused as future training data.

```bash
# Interactive chat
mtnsails chat --model-path ./onnx_model

# Single prompt
mtnsails chat --model-path ./onnx_model --prompt "What is Python?"

# Chat with conversation logging enabled
mtnsails chat --model-path ./onnx_model --log-conversations --log-file chat_data.json
```

### 6. One-Command Pipeline

The `pipeline` command chains training, ONNX conversion, and a quick test chat into a single step — ideal for getting started quickly or automating repeated training runs.

```bash
mtnsails pipeline --epochs 3 --batch-size 4
```

## Commands

| Command | Description |
|---------|-------------|
| `validate` | Analyze and optionally filter training data quality |
| `train` | Fine-tune a model on conversation data |
| `convert` | Export a trained model to ONNX format |
| `chat` | Chat with an ONNX model interactively or with a single prompt |
| `pipeline` | Run train, convert, and chat in sequence |
| `daemon` | Background daemon: watch a JSONL feedback file and retrain when enough new examples accumulate |
| `taber` | Translate a natural-language forecast request into a taber_enviro prediction |
| `baseline` | Export the base model to ONNX without any fine-tuning |
| `reset` | Delete fine-tuned and ONNX model directories to start fresh |

### validate

Checks your conversation dataset for quality issues and produces a report showing how many conversations are valid, how many have problems, and an overall quality score. Common issues include empty responses, repetitive text, echoed inputs, and very short outputs. When the filter option is enabled, invalid conversations are removed and the cleaned dataset is saved — either to a path you specify or to a file named after the original with `_filtered` appended.

```bash
# Check data quality
mtnsails validate --data-file conversations.json

# Check and automatically filter out bad conversations
mtnsails validate --data-file conversations.json --filter

# Filter and save to a specific output file
mtnsails validate --data-file conversations.json --filter --output conversations_clean.json
```

### train

Fine-tunes a pre-trained language model on your conversation data. Supports configurable training epochs, batch size, and learning rate. If low-quality data is detected, a warning is displayed and confirmation is requested before training begins (this check can be skipped with the force flag). Automatically resumes from an existing checkpoint when one is present in the output directory.

```bash
# Train with default settings
mtnsails train --data-file conversations.json

# Train with custom epochs, batch size, and output directory
mtnsails train --data-file conversations.json --epochs 5 --batch-size 2 --output-dir ./my_model

# Skip data quality warning and force training
mtnsails train --data-file conversations.json --epochs 3 --force
```

### convert

Converts a fine-tuned model from PyTorch format to ONNX format. ONNX models run faster than standard PyTorch models on CPU hardware. An optional verification pass confirms the conversion succeeded.

```bash
# Convert a trained model to ONNX
mtnsails convert --model-path ./trained_model --onnx-output ./onnx_model

# Convert and verify the output
mtnsails convert --model-path ./trained_model --onnx-output ./onnx_model --verify
```

### chat

Launches a conversation session with an ONNX model. Supports both interactive mode — where you type prompts in a loop — and single-prompt mode for scripted or one-off use. Conversation turns can be logged to a file for later use as additional training data.

```bash
# Interactive chat session
mtnsails chat --model-path ./onnx_model

# Single one-off prompt
mtnsails chat --model-path ./onnx_model --prompt "What is machine learning?"

# Chat with conversation logging
mtnsails chat --model-path ./onnx_model --log-conversations --log-file chat_history.json

# Chat with human feedback collection for retraining
mtnsails chat --model-path ./onnx_model --feedback-file ./live_pairs.jsonl
```

### pipeline

Runs the full workflow — training, ONNX conversion, and a quick chat test — in a single command. Useful for automating the end-to-end process or getting started quickly without running each step individually.

```bash
# Run the full pipeline with defaults
mtnsails pipeline --data-file conversations.json

# Run with custom settings
mtnsails pipeline --data-file conversations.json --epochs 5 --batch-size 2 --output-dir ./my_model --onnx-output ./my_onnx
```

### taber

Bridges MTN Sails with the [taber_enviro](https://github.com/odds-get-evened/taber_enviro) environmental predictor. Describe a forecasting scenario in plain English, and the model translates it into a structured request that is passed to taber_enviro for prediction. Supports interactive and single-prompt modes, and can save request-and-response pairs to disk for future retraining. See the [Taber Bridge](#taber-bridge) section below for full details.

### baseline

Exports the base model (DistilGPT-2 by default) to ONNX format directly, without any fine-tuning. Useful for comparing base-model behavior against a fine-tuned version. An optional test flag runs a short generation after export to confirm the model is working.

```bash
mtnsails baseline --baseline-output ./onnx_baseline --test
```

### reset

Deletes the fine-tuned model directory and the ONNX model directory, returning the project to a clean state ready for a fresh training run. A confirmation prompt is shown before any files are removed unless the force flag is used.

```bash
mtnsails reset
```

### daemon

Starts a long-running background process that watches a JSONL feedback file (produced by chat feedback mode) and automatically retrains the PyTorch model, then exports a new ONNX model, whenever at least N new labeled examples have accumulated since the last retrain cycle.

```bash
# Start the daemon with defaults (watches ./live_pairs.jsonl, retrains every 50 examples)
mtnsails daemon

# Custom settings
mtnsails daemon \
  --feedback-file ./live_pairs.jsonl \
  --threshold 20 \
  --max-steps 100 \
  --poll-interval 60
```

After each retraining cycle the daemon saves the updated model and rotates the ONNX directories:
- `./onnx_model_next` (staging) → `./onnx_model` (production)
- `./onnx_model` (previous) → `./onnx_model_prev` (archive)

A manual restart of `chat` is required to pick up the new model weights.

## Feedback Mode and Live Retraining

MTN Sails supports a human-in-the-loop workflow for continuously improving the model using real conversations.

### Step 1 — Chat in feedback mode

```bash
mtnsails chat --model-path ./onnx_model --feedback-file ./live_pairs.jsonl
```

After each model response you will be prompted:

```
[Accept? Press Enter, or type a correction]:
```

- **Press Enter** to accept the model's response as a training example.
- **Type a correction** to replace the response with a better one before saving.

Each approved or corrected pair is appended to `live_pairs.jsonl` as a single JSON record:

```json
{"input": "What is Python?", "output": "Python is a programming language.", "timestamp": "2024-01-01T12:00:00", "accepted": true}
```

The format is fully compatible with `ConversationDataHandler.load_from_jsonl()` and the existing training pipeline.

### Step 2 — Run the daemon in a separate terminal

```bash
mtnsails daemon --feedback-file ./live_pairs.jsonl --threshold 50 --max-steps 100
```

The daemon polls the JSONL file every 30 seconds (configurable with `--poll-interval`).  Once at least `--threshold` new examples have arrived since the last retrain it:

1. Loads the new pairs from the file.
2. Fine-tunes the existing checkpoint in `./trained_model` (or trains from base `distilgpt2` if no checkpoint exists).  A low learning rate (1e-5) is used automatically when continuing from a checkpoint.
3. Exports the updated model to ONNX and rotates directories atomically.

### Step 3 — Restart chat to use the new model

After the daemon completes a retrain cycle, restart the chat command to load the updated ONNX model.

### Recommended settings

| Setting | Value | Notes |
|---------|-------|-------|
| `--threshold` | 50 | Collect at least 50 good examples before retraining |
| `--max-steps` | 100 | Short training run; fast incremental updates |
| `--poll-interval` | 30 | Seconds between file checks |
| Learning rate | 1e-5 (auto) | Applied automatically when a checkpoint exists |

### Caveats

- Only approve/correct responses — do not accept raw model outputs that are clearly wrong.
- The feedback JSONL file grows indefinitely; archive or rotate it periodically.
- Hot-reload of the chat process is not supported; a manual restart is required after retraining.

## Taber Bridge

The Taber bridge connects MTN Sails to the [taber_enviro](https://github.com/odds-get-evened/taber_enviro) environmental forecasting predictor. Instead of constructing a structured forecast request by hand, you describe what you want in plain English — for example, "Predict temperature and humidity for sensor 7 over the next 24 hours at 1-hour intervals." The language model interprets the request, produces a validated forecast specification, and passes it to the predictor automatically.

```bash
# Interactive taber session
mtnsails taber --model-path ./onnx_model

# One-off request
mtnsails taber --model-path ./onnx_model --prompt "Predict temperature and humidity for sensor 7 over the next 24 hours at 1-hour intervals"

# Save request/response pairs to disk for future retraining
mtnsails taber --model-path ./onnx_model --save-dir ./taber_data
```

**How it works:**

1. Your request is paired with a system prompt that instructs the model to respond with a structured specification.
2. The model's output is parsed to extract that specification.
3. The specification is validated: it must include a sensor query, forecast duration, sampling interval, and output format. Optionally, you can limit which environmental targets (temperature, barometer, light, humidity) are predicted.
4. The validated request is forwarded to taber_enviro, which runs the actual prediction.
5. Results are displayed in your chosen format — JSON, CSV, or a human-readable table.

Optionally, each request and its response can be saved to disk as training data for future model refinement.

**Prerequisites:** The `taber_enviro` package must be installed and accessible before using this feature.

## Data Quality

The `validate` command scores your dataset on a 0–100% scale:

| Score | Status | Recommendation |
|-------|--------|----------------|
| 70% or above | ✅ Good | Proceed with training |
| 50–70% | ⚠️ Acceptable | Consider filtering problematic conversations |
| Below 50% | ❌ Critical | Filter or replace data before training |

Common issues detected: empty responses, repetitive text, input echoing, and very short outputs. Training on critically low-quality data is likely to produce a model that generates poor or nonsensical responses.

## Continuous Training

Every time `train` or `pipeline` is run, MTN Sails checks whether a trained model already exists in the output directory:

- **Model found** — continues fine-tuning from the checkpoint at a reduced learning rate to preserve previously learned knowledge.
- **No model found** — trains from scratch using the specified base model and learning rate.

This makes it easy to incrementally improve the model as new conversation data becomes available, without losing what was learned in earlier sessions.

## Processing Scraped Content

MTN Sails includes a utility (`process_scraped_content.py`) for converting plain-text files scraped with [scrapyer](https://github.com/odds-get-evened/scrapyer) into training-ready conversation data. Each text file is processed into input-output conversation pairs suitable for fine-tuning. Custom prompt templates are supported to control how the input side of each pair is phrased.

```bash
# Convert a directory of scraped text files into training data
python process_scraped_content.py /path/to/scraped/files --output chat_data.json

# Use a custom prompt template
python process_scraped_content.py /path/to/scraped/files --prompt-template "What is {topic} from {source}?" --output chat_data.json

# Disable prompt randomization for consistent output
python process_scraped_content.py /path/to/scraped/files --no-randomize --output chat_data.json
```

## License

[GPL-3.0](LICENSE)