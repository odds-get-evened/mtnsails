# MTN Sails - LLM Training and ONNX Conversion System

A streamlined, object-oriented Python system for training small language models on conversation data and converting them to ONNX format for efficient CPU inference.

## Features

- **OOP Design**: Clean, reusable classes for data handling, training, conversion, and chat
- **CPU-Friendly**: Uses small models (DistilGPT2) that run efficiently on CPU
- **ONNX Export**: Converts models to ONNX format using safetensors for optimized inference
- **Batch Training**: Train on batches of conversation data
- **Interactive Chat**: Chat interface for testing ONNX models
- **Conversation Logging**: Optional async logging for collecting retraining data
- **Data Quality Validation**: Automatic detection of low-quality training data to prevent "garbage in, garbage out" problems
- **Scrapyer Integration**: Process web-scraped content into conversational training data (see [SCRAPYER_INTEGRATION.md](SCRAPYER_INTEGRATION.md))

## Quick Links

- **Getting Started**: See [Installation](#installation) and [Quick Start](#quick-start) below
- **Programming Guide**: See [API_REFERENCE.md](API_REFERENCE.md) for code examples and API documentation
- **Step-by-Step Tutorial**: See [QUICKSTART.md](QUICKSTART.md) for detailed walkthrough
- **Developer Guide**: See [DEVELOPMENT.md](DEVELOPMENT.md) for architecture and extension details
- **Loss Configuration & Training Optimization**: See [LOSS_CONFIGURATION.md](LOSS_CONFIGURATION.md) for improving model performance
- **Scrapyer Integration**: See [SCRAPYER_INTEGRATION.md](SCRAPYER_INTEGRATION.md) for processing web-scraped content

## Architecture

The system consists of four main components:

1. **ConversationDataHandler**: Manages conversation data loading and formatting
2. **LLMTrainer**: Handles model fine-tuning on conversation batches
3. **ONNXConverter**: Converts trained models to ONNX format
4. **ChatInterface**: Provides chat functionality with ONNX models

### Additional Tools

- **process_scraped_content.py**: Converts web content from scrapyer into conversational training data
- **validate.py**: Lightweight validation script for verifying project structure
- **main.py**: Command-line interface for all training and inference operations

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

### Option 3: Build from Web Content (New!)

Quickly create a specialized assistant from web documentation:

```bash
# Install scrapyer
pip install git+https://github.com/odds-get-evened/scrapyer.git

# Scrape multiple pages
scrapyer "https://example.com/docs/page1" /tmp/content/
scrapyer "https://example.com/docs/page2" /tmp/content/
scrapyer "https://example.com/docs/page3" /tmp/content/

# Process into training data
python process_scraped_content.py /tmp/content/ --output docs_training.json

# Validate, train, and deploy
python main.py validate --data-file docs_training.json
python main.py pipeline --data-file docs_training.json --epochs 3
```

This workflow is perfect for building domain-specific assistants from documentation, tutorials, or knowledge bases.

### Option 4: Interactive Step-by-Step

See [Step-by-Step Workflow](#workflow-how-to-use-mtn-sails) below for detailed instructions.

## Workflow: How to Use MTN Sails

### Step 1: Prepare Your Data

You have multiple options for preparing training data:

#### Option A: Create Manual Conversation Data

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

#### Option B: Process Web Content with Scrapyer (New!)

Use [scrapyer](https://github.com/odds-get-evened/scrapyer) to automatically extract content from web pages and convert it to training data:

```bash
# 1. Install scrapyer
pip install git+https://github.com/odds-get-evened/scrapyer.git

# 2. Scrape web content
scrapyer "https://example.com/documentation" /tmp/scraped/

# 3. Convert to conversational format
python process_scraped_content.py /tmp/scraped/ --output my_conversations.json
```

**Benefits of using scrapyer:**
- Quickly build training datasets from documentation, articles, or tutorials
- Automatically extracts and cleans text content
- Generates proper conversation format
- Supports custom prompt templates

See [SCRAPYER_INTEGRATION.md](SCRAPYER_INTEGRATION.md) for complete documentation and examples.

### Step 2: Validate Your Data Quality

**Before training, validate your data quality:**

```bash
python main.py validate --data-file my_conversations.json
```

**What this does:**
- Analyzes all conversations for quality issues
- Detects repetitive, empty, gibberish, or echo responses
- Shows a quality score (0-100%)
- Provides specific examples of problematic conversations
- Gives recommendations for improving data quality

**Example output:**
```
=== Data Quality Validation ===
Total conversations: 42
Valid conversations: 38
Invalid conversations: 4
Quality score: 90.5%

✅ DATA QUALITY: GOOD
Your data quality is good. You can proceed with training.
```

**Automatic Data Filtering:**

If you have a dataset with mixed quality, you can automatically filter out bad conversations:

```bash
python main.py validate --data-file my_conversations.json --filter
```

This will:
- Remove all conversations with quality issues (empty outputs, highly repetitive text, gibberish, echoing, too-short responses)
- Save only valid, high-quality conversations to `my_conversations_filtered.json`
- Show you how many conversations were removed

You can then train on the filtered data:

```bash
python main.py train --data-file my_conversations_filtered.json
```

**Why validate?** Training on low-quality data produces models that generate low-quality responses. This "garbage in, garbage out" problem wastes time and produces poor results. Always validate first!

### Step 3: Train Your Model

Train a language model on your conversation data:

```bash
python main.py train --data-file my_conversations.json --epochs 3 --batch-size 4
```

**What happens:**
- Loads your conversation data
- **Validates data quality automatically** (detects repetitive, empty, or nonsensical responses)
- Shows quality report with warnings if issues are detected
- Fine-tunes a DistilGPT2 model on your conversations
- Saves the trained model to `./trained_model` directory

**Training tips:**
- Start with 2-3 epochs for quick testing
- Use batch size 2-4 for CPU training
- Monitor the loss values (should decrease)

### Step 4: Convert to ONNX Format

Convert your trained model to ONNX for optimized inference:

```bash
python main.py convert --model-path ./trained_model --onnx-output ./onnx_model --verify
```

**What happens:**
- Converts PyTorch model to ONNX format
- Optimizes for CPU inference
- Verifies the conversion was successful
- Saves to `./onnx_model` directory

### Step 5: Use Your Model

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

### Step 6: Retrain with New Data

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

### Validate Command

Validate your training data quality before training:

```bash
python main.py validate --data-file PATH
```

**Required Options:**
- `--data-file PATH` - Path to conversation JSON file to validate

**Optional Options:**
- `--filter` - Automatically remove bad data and save only valid conversations
- `--output PATH` - Specify output file for filtered data (default: `<input>_filtered.json`)

**What it does:**
- Analyzes all conversations for quality issues
- Calculates a quality score (0-100%)
- Identifies specific problems (repetition, gibberish, empty responses, etc.)
- Shows example problematic conversations
- Provides recommendations
- Exits with code 0 if quality is good (≥50%), 1 if critical (<50%)
- **With `--filter` flag:** Automatically removes bad conversations and saves clean data

**Examples:**
```bash
# Validate your data before training
python main.py validate --data-file my_data.json

# Validate and automatically filter bad data
python main.py validate --data-file my_data.json --filter

# Validate, filter, and save to specific file
python main.py validate --data-file my_data.json --filter --output clean_data.json

# Use in scripts (checks exit code)
python main.py validate --data-file data.json && python main.py train --data-file data.json

# Validate, filter, then train on clean data
python main.py validate --data-file data.json --filter && \
python main.py train --data-file data_filtered.json
```

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

### Web Content Training (New!)

Use [scrapyer](https://github.com/odds-get-evened/scrapyer) to download web content and convert it to training data:

```bash
# 1. Install scrapyer
pip install git+https://github.com/odds-get-evened/scrapyer.git

# 2. Scrape content from web pages
scrapyer "https://example.com/article" /tmp/scraped_content/

# 3. Process into conversational format
python process_scraped_content.py /tmp/scraped_content/ --output chat_data.json

# 4. Validate and train
python main.py validate --data-file chat_data.json
python main.py train --data-file chat_data.json --epochs 3
```

See [SCRAPYER_INTEGRATION.md](SCRAPYER_INTEGRATION.md) for detailed documentation.

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

## Loss Function and Training Configuration

MTN Sails uses the **standard cross-entropy loss** for causal language modeling, which is the optimal and recommended approach for GPT-style models. The loss function is automatically handled by the transformers library.

**Note**: If you see a warning about `loss_type`, this is a deprecation warning from the transformers library and can be safely ignored. MTN Sails suppresses this warning automatically.

**To improve model performance**, focus on:
- **Data quality** (most important) - Use `python main.py validate` to check
- **Learning rate tuning** - Adjust with `--learning-rate` flag
- **Training epochs** - Increase with `--epochs` flag
- **Training data quantity** - Use 50+ high-quality conversations

For detailed information about loss functions and training optimization, see [LOSS_CONFIGURATION.md](LOSS_CONFIGURATION.md).

## Best Practices

### Data Quality

**IMPORTANT: Quality data is critical for good model performance!**

MTN Sails automatically validates your training data quality and warns you about:
- **Repetitive responses** - Text that repeats the same phrases over and over
- **Empty responses** - Conversations with no output
- **Gibberish** - Nonsensical text or URL fragments
- **Echo responses** - Outputs that simply repeat the user's input
- **Too-short responses** - Meaningless one or two-word answers

#### Understanding the "Garbage In, Garbage Out" Problem

If you train on low-quality conversation logs (like chat sessions with a poorly performing model), your new model will learn to produce similar low-quality responses. This creates a cycle of degrading performance.

**Example of BAD training data:**
```json
[
  {
    "input": "hello there",
    "output": "yes i"
  },
  {
    "input": "stop repeating",
    "output": "stop repeating stop repeating stop repeating stop repeating"
  },
  {
    "input": "what is AI?",
    "output": ""
  }
]
```

**Example of GOOD training data:**
```json
[
  {
    "input": "What is artificial intelligence?",
    "output": "Artificial intelligence is the simulation of human intelligence by machines, enabling them to perform tasks that typically require human cognition."
  },
  {
    "input": "How does machine learning work?",
    "output": "Machine learning uses algorithms to analyze data, identify patterns, and make predictions or decisions without being explicitly programmed for each specific task."
  }
]
```

#### Data Quality Guidelines

- Use at least 20 diverse, high-quality conversations for training
- Ensure all responses are coherent, complete sentences
- Keep input/output lengths balanced
- Include various question types and scenarios
- **Review your data for accuracy before training**
- **Never train on chat logs from a poorly performing model**
- Filter out empty, repetitive, or nonsensical responses

#### Checking Data Quality

Run the data quality demo to see validation in action:
```bash
python demo_data_quality.py
```

When training, the system will:
1. Automatically analyze your data quality
2. Show a quality report with specific issues found
3. Display problematic conversation examples
4. Warn you if quality is low (< 70%)
5. Block training if quality is critically low (< 50%) unless you use `--force`

**Example quality report:**
```
=== Data Quality Analysis ===
Total conversations: 54
Valid conversations: 12
Invalid conversations: 42
Quality score: 22.2%

Issues detected:
  - Repetitive Outputs: 28
  - Empty Outputs: 8
  - Gibberish Outputs: 6

⚠️  CRITICAL WARNING: DATA QUALITY IS VERY LOW
Training on this data will result in a model that produces nonsense responses.
```

#### What to Do If You Get a Quality Warning

1. **Use the automatic filter feature** - Let the system remove bad conversations:
   ```bash
   python main.py validate --data-file my_data.json --filter
   ```
   Then train on the filtered data:
   ```bash
   python main.py train --data-file my_data_filtered.json
   ```

2. **Review your data file** - Look at the problematic examples shown in the report

3. **Filter out bad conversations** - Remove or fix low-quality responses manually if needed

4. **Don't train on chat logs from failing models** - Use curated, high-quality data instead

5. **Start with example data** - Use `example_conversations.json` as a template

6. **Only use `--force` if you understand the consequences**

### Training
- Start with 2-3 epochs for initial testing
- Monitor loss values during training (should decrease)
- Increase epochs if responses aren't satisfactory
- Use smaller batch sizes (2-4) for CPU training

### Generation Quality
- Lower `--max-tokens` (30-50) for concise responses
- Higher `--max-tokens` (100-200) for detailed responses
- Adjust based on your use case and response quality
- The system includes automatic repetition penalty (default: 1.2) to prevent repetitive responses

### Retraining Workflow
1. Deploy your model with conversation logging enabled
2. Collect real user conversations over time
3. **CRITICAL: Review and filter the logged conversations for quality**
4. Remove repetitive, empty, or nonsensical responses
5. Combine filtered data with original high-quality training data
6. Retrain periodically (weekly/monthly)
7. Compare new model with old before deploying

## Troubleshooting

### Out of Memory Errors

Reduce batch size:
```bash
python main.py train --batch-size 2
```

Use the smaller DistilGPT2 model (default).

### Poor Response Quality / Repetitive / Nonsensical Outputs

**This is almost always a data quality problem!**

If your model produces:
- Repetitive text (e.g., "hello hello hello hello")
- Nonsensical responses
- Empty or very short responses
- Echoes of user input

**Root Cause:** You trained on low-quality data (the "garbage in, garbage out" problem).

**Solution:**
1. **Check your training data quality:**
   ```bash
   python demo_data_quality.py
   ```

2. **Review the quality report** when training - it will show you specific issues

3. **Do NOT train on chat logs from a poorly performing model**
   - If your model is producing bad outputs, don't use those conversations for retraining
   - This creates a downward spiral of quality

4. **Start fresh with high-quality data:**
   ```bash
   # Use the provided examples as a template
   python main.py train --data-file example_conversations.json --epochs 3
   ```

5. **Filter your conversation logs:**
   - Remove empty responses
   - Remove repetitive responses  
   - Remove gibberish
   - Keep only coherent, complete, helpful responses

6. **If you still have issues**, try:
   - Training for more epochs (5-10)
   - Using more diverse training data
   - Starting from the base model again instead of retraining
   - Adjusting the `repetition_penalty` parameter when generating responses (default is 1.2, try values between 1.0-1.5)

**Remember:** High training loss values (> 2.0 after 3 epochs) often indicate your data quality is poor or doesn't match the expected conversation format.

### Data Quality Warning When Training

If you see:
```
⚠️  CRITICAL WARNING: DATA QUALITY IS VERY LOW
```

**Do NOT proceed with training!** Your training data has serious quality issues.

To bypass the warning (not recommended):
```bash
python main.py train --data-file my_data.json --force
```

But you'll likely get a poorly performing model.

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

## Tools and Scripts

### Main Scripts

- **main.py** - Primary CLI for training, conversion, and chat operations
- **example.py** - Demonstration script showing complete workflow
- **validate.py** - Lightweight validation of project structure

### Data Processing Tools

- **process_scraped_content.py** - Convert web-scraped content to training data
  ```bash
  python process_scraped_content.py <directory> --output <file.json>
  ```
  See [SCRAPYER_INTEGRATION.md](SCRAPYER_INTEGRATION.md) for details.

- **demo_data_quality.py** - Demonstrates data quality validation features

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
