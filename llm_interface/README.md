# LLM Interface

## Overview

The enhanced training data generator can now create large-scale, diverse training datasets for the MTN Sails LLM using augmentation and synthetic generation techniques.

## Key Features

### 1. Query Augmentation
Generates variations of existing queries using:
- **Synonym replacement**: 40+ mappings for common terms
- **Paraphrasing**: Natural language variations
- **Capitalization**: Sentence case, lowercase variations

### 2. Synthetic Data Generation
Creates new training examples using templates:
- **Forecast queries**: 20+ templates for prediction tasks
- **Anomaly detection**: 20+ templates for outlier identification
- **Gradient analysis**: 18+ templates for spatial analysis
- **Correlation queries**: 16+ templates for relationship analysis

### 3. Data Quality Controls
- **Deduplication**: Set-based tracking prevents duplicate queries
- **Uniqueness**: All queries are unique (case-insensitive)
- **Validation**: Ensures grammatical correctness
- **Reproducibility**: Random seed for deterministic results

---

## 📊 Training Data Generator Usage

### Quick Start

Generate training data with a specific count:

```bash
python3 llm_interface/training_data_generator.py --count <number_of_samples>
```

### CLI Options

#### Basic Options
```bash
--output FILE          # Output JSON file (default: iot_analyst_training.json)
--pretty               # Pretty-print JSON output
--count N              # Limit output to N examples (legacy option)
--quiet                # Suppress informational output
```

#### Augmentation Options
```bash
--augment-factor N     # Generate N variations of each base example
                       # (default: 2, legacy --augment also supported)
```

#### Synthetic Generation Options
```bash
--synthetic-count N    # Generate N synthetic examples using templates
                       # Distribution: 40% forecast, 30% anomaly, 
                       #              15% gradient, 15% correlation

--diversity-mode       # Enable maximum variation
                       # - Uses 15 metric types (vs 6 in normal mode)
                       # - Includes: temp, humidity, light, pressure, barometer,
                       #   vibration, CO2, noise, motion, proximity, 
                       #   air quality, soil moisture, wind speed, rain, UV index
```

#### Target-Based Generation
```bash
--total-target N       # Generate data until reaching N total examples
                       # Automatically calculates required synthetic count
```

#### Reproducibility
```bash
--seed N               # Set random seed for reproducible generation
                       # Same seed produces identical output
```

### Usage Examples

#### Generate 100+ examples with moderate augmentation
```bash
python3 llm_interface/training_data_generator.py \
  --output training_data.json \
  --augment-factor 3 \
  --synthetic-count 100 \
  --seed 42
```

#### Generate 1,000+ examples with diversity
```bash
python3 llm_interface/training_data_generator.py \
  --output training_1k.json \
  --augment-factor 3 \
  --synthetic-count 1000 \
  --diversity-mode \
  --seed 42
```

#### Generate 10,000+ examples
```bash
python3 llm_interface/training_data_generator.py \
  --output training_10k.json \
  --augment-factor 5 \
  --synthetic-count 10000 \
  --diversity-mode \
  --seed 42
```

#### Target-based generation
```bash
python3 llm_interface/training_data_generator.py \
  --output training_target.json \
  --total-target 5000 \
  --augment-factor 4 \
  --diversity-mode \
  --seed 42
```

#### Generate for CI/CD (quiet mode)
```bash
python3 llm_interface/training_data_generator.py \
  --output training_ci.json \
  --synthetic-count 500 \
  --quiet \
  --seed 42
```

### Output Format

The generator produces JSON files with training examples:

```json
[
  {
    "query": "What will temperature be in 2 hours?",
    "category": "forecast",
    "metric": "temperature",
    "time_range": 120,
    "expected_action": "Run LSTM forecast for temperature with 120 minute prediction window"
  },
  {
    "query": "Are there any anomalies in humidity?",
    "category": "anomaly_detection",
    "metric": "humidity",
    "expected_action": "Run anomaly detection on humidity sensor data"
  },
  {
    "query": "How does temperature correlate with humidity?",
    "category": "correlation",
    "metrics": ["temperature", "humidity"],
    "expected_action": "Calculate correlation between temperature and humidity"
  }
]
```

### Statistics Output

After generation, the tool displays comprehensive statistics:

```
============================================================
TRAINING DATA STATISTICS
============================================================
Total Examples: 3411
Unique Queries: 3363
Vocabulary Size: 155 unique words
Average Query Length: 35.5 characters

Distribution by Category:
  forecast            :   2000 ( 58.6%)
  correlation         :    750 ( 22.0%)
  anomaly_detection   :    300 (  8.8%)
  gradient_analysis   :    270 (  7.9%)
  other               :     91 (  2.7%)

Distribution by Metric:
  temperature         :    305
  humidity            :    278
  pressure            :    269
  ...
============================================================

✓ Output written to training_data.json
✓ File size: 3411 entries
```

---

## 🌉 MTN Sails Bridge Usage

The MTN Sails Bridge connects natural language queries to LSTM predictions, enabling conversational IoT analytics.

### Architecture

```
User Query (Natural Language)
    ↓
MTN Sails LLM (Parse Intent)   ← required; no rule-based fallback
    ↓
Taber LSTM Predictor (Numerical Predictions)
    ↓
MTN Sails LLM (Generate Explanation)   ← required; no template fallback
    ↓
Human-Readable Response
    ↓
[intent_valid=True AND lstm_success=True]
    ↓
Validator-Confirmed Retrain Buffer (JSONL)
    ↓
Retrain Daemon (continual fine-tuning)
```

**The LLM is required for both query parsing and result explanation.**
If the model is not provided, the bridge returns a clear error with installation instructions.

### Installation

#### Required Dependencies

```bash
# Core dependencies (REQUIRED)
pip install numpy pandas onnxruntime

# LLM support (REQUIRED for bridge operation)
pip install optimum[onnxruntime] transformers
```

### Quick Start

#### With MTN Sails LLM (Required)

```bash
python3 llm_interface/mtnsails_bridge.py \
  --mtnsails-model path/to/onnx_model \
  --taber-model outputs/ \
  --interactive
```

#### With Retrain Buffer Logging

```bash
python3 llm_interface/mtnsails_bridge.py \
  --mtnsails-model path/to/onnx_model \
  --taber-model outputs/ \
  --buffer-dir ./llm_interface/retrain_buffer \
  --interactive
```

#### Single Query Mode

```bash
python3 llm_interface/mtnsails_bridge.py \
  --taber-model outputs/ \
  --query "What will temperature be in 2 hours?"
```

### CLI Options

```bash
--mtnsails-model PATH  # Path to MTN Sails ONNX model directory (REQUIRED)
--taber-model PATH     # Path to Taber LSTM model directory (REQUIRED)
--interactive          # Run in interactive mode
--query "TEXT"         # Process a single query
--buffer-dir PATH      # Path to JSONL retrain buffer directory (optional)
```

### Usage Examples

#### Example 1: Temperature Forecasting

```bash
$ python3 llm_interface/mtnsails_bridge.py --taber-model outputs/ --query "What will temperature be in 2 hours?"

🔍 Processing query: What will temperature be in 2 hours?
📋 Extracted intent: {'type': 'forecast', 'metric': 'temp', 'duration': 120}
📊 LSTM output received

🤖 Response:
📊 Temperature Forecast

Current: 23.45°C
In 15 min: 23.52°C (+0.07°C)
In 30 min: 23.61°C (+0.16°C)
In 1 hour: 23.78°C (+0.33°C)
In 120 min: 24.02°C (+0.57°C)

📈 Trend: Rising
```

#### Example 2: Anomaly Detection

```bash
$ python3 llm_interface/mtnsails_bridge.py --taber-model outputs/ --query "Are there any unusual readings?"

✓ Anomaly Detection: LOW SEVERITY

Metric: temp
Anomaly Score: 0.12/1.0
Prediction Error: 0.23

✓ No significant anomalies detected
Sensor operating within normal parameters
```

#### Example 3: Gradient Analysis

```bash
$ python3 llm_interface/mtnsails_bridge.py --taber-model outputs/ --query "Which sensors have the biggest temperature difference?"

📍 Temperature Spatial Gradients

Top sensor pairs with largest differences:

1. sensor_01 - sensor_05: 3.45°C over 12.3m (gradient: 0.281/m)
2. sensor_03 - sensor_07: 2.87°C over 8.7m (gradient: 0.330/m)
3. sensor_02 - sensor_04: 1.92°C over 5.2m (gradient: 0.369/m)

💡 Moderate gradients - consider airflow optimization
```

#### Example 4: Interactive Session

```bash
$ python3 llm_interface/mtnsails_bridge.py --taber-model outputs/ --interactive

============================================================
MTN Sails + Taber LSTM - Natural Language Interface
============================================================

Example queries:
  • What will temperature be in 2 hours?
  • Are there any unusual readings?
  • Which sensors have the biggest temperature difference?

Type 'quit' or 'exit' to stop

💬 You: Show me temp forecast

📊 Temperature Forecast
Current: 23.45°C
In 30 min: 23.61°C (+0.16°C)
...

💬 You: Any anomalies?

✓ Anomaly Detection: LOW SEVERITY
...

💬 You: quit

👋 Goodbye!
```

### Programmatic Usage

#### Python API Example

```python
from llm_interface.mtnsails_bridge import MTNSailsLSTMBridge

# Initialize bridge (no LLM required)
bridge = MTNSailsLSTMBridge(taber_model_path="outputs/")

# Ask questions
response = bridge.ask("What will temperature be in 1 hour?")
print(response)

# Multiple queries
queries = [
    "Forecast humidity for 2 hours",
    "Check for anomalies",
    "Show temperature gradients"
]

for query in queries:
    response = bridge.ask(query)
    print(f"Query: {query}")
    print(f"Response: {response}\n")
```

#### With MTN Sails LLM

```python
from llm_interface.mtnsails_bridge import MTNSailsLSTMBridge

# Initialize with LLM
bridge = MTNSailsLSTMBridge(
    mtnsails_model_path="path/to/onnx_model",
    taber_model_path="outputs/"
)

# LLM handles parsing and explanation
response = bridge.ask("What's going to happen with the temperature?")
print(response)
```

### Query Types Supported

#### 1. **Forecast Queries**

Natural language queries for time-series predictions:

```
• "What will temperature be in 2 hours?"
• "Forecast humidity for the next 30 minutes"
• "Predict light levels in 1 hour"
• "Show me barometer forecast"
```

**Detected patterns**:
- Time expressions: "2 hours", "30 minutes", "1 hour"
- Metrics: temperature, humidity, light, barometer
- Intent keywords: forecast, predict, will be

#### 2. **Anomaly Detection Queries**

Identify unusual sensor behavior:

```
• "Are there any unusual temperature readings?"
• "Is the pump going to fail?"
• "Check for sensor failures"
• "Any anomalies?"
```

**Detected patterns**:
- Keywords: anomaly, unusual, fail, wrong, problem
- Checks prediction error vs. expected error distribution
- Severity levels: Low, Medium, High

#### 3. **Gradient Analysis Queries**

Spatial distribution analysis:

```
• "Which sensors have the biggest temperature difference?"
• "Show me humidity gradients between sensors"
• "Where are the temperature hotspots?"
• "Compare sensors"
```

**Detected patterns**:
- Keywords: gradient, difference, compare, which sensor
- Requires multiple sensors with location data
- Shows top 5 sensor pairs by gradient magnitude

### Example Scripts

#### Run Examples

```bash
# Run all examples
python3 llm_interface/examples/example_queries.py --taber-model outputs/

# Run specific example
python3 llm_interface/examples/example_queries.py \
  --taber-model outputs/ \
  --example forecast

# With MTN Sails LLM
python3 llm_interface/examples/example_queries.py \
  --mtnsails-model onnx_model \
  --taber-model outputs/ \
  --example all
```

**Available examples**:
- `forecast` - Temperature forecasting
- `anomaly` - Anomaly detection
- `gradient` - Spatial gradient analysis
- `conversation` - Multi-turn conversation
- `metrics` - Different sensor metrics
- `all` - Run all examples

---

## 🔧 Integration Guide

### Step 1: Generate Training Data

```bash
# Generate comprehensive training set
python3 llm_interface/training_data_generator.py \
  --output training_data.json \
  --total-target 10000 \
  --diversity-mode \
  --seed 42
```

### Step 2: Train MTN Sails LLM

Use the generated training data to fine-tune your MTN Sails model:

```bash
# Example using Hugging Face transformers
python train_mtnsails.py \
  --train-data training_data.json \
  --model-name distilgpt2 \
  --output-dir mtnsails_model
```

### Step 3: Export to ONNX

```bash
# Export trained model to ONNX format
python export_to_onnx.py \
  --model-dir mtnsails_model \
  --output-dir onnx_model
```

### Step 4: Use the Bridge

```python
from llm_interface.mtnsails_bridge import MTNSailsLSTMBridge

# Initialize with LLM (required) and optional retrain buffer
bridge = MTNSailsLSTMBridge(
    mtnsails_model_path="onnx_model",  # required
    taber_model_path="outputs/",
    buffer_dir="./llm_interface/retrain_buffer",  # optional
)

# Process queries
response = bridge.ask("What will temperature be in 2 hours?")
```

---

## 🔄 Validator-Confirmed Continual Learning

### How It Works

After each successful query (intent parsed + LSTM returned results without errors),
the bridge logs the interaction to a JSONL buffer file:

```
llm_interface/retrain_buffer/
    validated_2026-03-01.jsonl
    validated_2026-03-02.jsonl
    ...
```

Each record stores:
- `timestamp` – UTC ISO-8601
- `user_query` – original user input
- `parsed_intent` – structured intent (type/metric/duration/sensor_id)
- `llm_intent_raw_response` – raw LLM parse output (for debugging)
- `intent_prompt` – prompt sent to LLM
- `intent_valid` / `lstm_success` – always `true` (only validated records stored)
- `training_text` – `"User: ...\nAssistant: ..."` string ready for fine-tuning

Only successful interactions are buffered, preventing the model from
learning from misinterpretations or LSTM failures.

### Retrain Daemon

Start the daemon alongside the bridge to automatically fine-tune the model
when enough validated examples accumulate:

```bash
python -m llm_interface.retrain_daemon \
    --model-dir ./mtnsails_model \
    --buffer-dir ./llm_interface/retrain_buffer \
    --min-examples 200 \
    --check-interval 60 \
    --epochs 1 \
    --batch-size 4 \
    --learning-rate 1e-5
```

**Daemon options**:

| Option | Default | Description |
|--------|---------|-------------|
| `--model-dir` | (required) | MTN Sails model directory |
| `--buffer-dir` | (required) | JSONL buffer directory |
| `--min-examples` | 200 | Validated examples before retraining |
| `--check-interval` | 60 | Seconds between checks |
| `--epochs` | 1 | Fine-tuning epochs per cycle |
| `--batch-size` | 4 | Batch size |
| `--learning-rate` | 1e-5 | Fine-tuning learning rate |
| `--archive-dir` | (none) | Move consumed files here instead of deleting |

**Behaviour**:
- Scans buffer for validated JSONL records on every tick.
- Triggers retraining when total record count ≥ `--min-examples`.
- Uses a filesystem lock (`<model-dir>/.retrain.lock`) to prevent concurrent runs.
- On success, deletes (or archives) the consumed buffer files.

### Programmatic API

```python
from llm_interface.retrain_buffer import (
    append_validated_interaction,
    load_buffer_records,
    count_buffer_records,
)

# Check how many validated records are ready
n = count_buffer_records("./llm_interface/retrain_buffer")
print(f"{n} validated examples in buffer")

# Load all records
records = load_buffer_records("./llm_interface/retrain_buffer")
training_texts = [r["training_text"] for r in records]
```

---

## 🧪 Testing

### Test Training Data Generator

```bash
python3 tests/test_training_data_generator.py -v
```

**Test coverage**:
- Query augmentation tests (16 tests)
- Synthetic generation tests
- Data expansion and deduplication tests
- Statistics calculation tests
- Large-scale generation tests (100+ examples)

### Test MTN Sails Bridge

```bash
# Run with test data
python3 llm_interface/examples/example_queries.py \
  --taber-model outputs/ \
  --example all
```

---

## 📈 Performance

### Training Data Generator

- **Generation speed**: ~500-1000 examples/second
- **Memory usage**: Minimal (streaming generation)
- **Uniqueness guarantee**: 100% (set-based deduplication)
- **Reproducibility**: 100% (with seed)

### MTN Sails Bridge

- **Query processing**: ~100-500ms (rule-based mode)
- **Query processing**: ~500-2000ms (with LLM)
- **Memory footprint**: ~100MB (rule-based), ~500MB (with LLM)
- **Prediction latency**: Depends on LSTM model size

---

## ⚠️ Limitations

### Training Data Generator

1. **Maximum unique examples**: Due to template-based generation, there's a practical limit
   of ~6,000-8,000 truly unique examples without adding more templates or metrics.

2. **Semantic variation**: While queries are grammatically correct and unique, they follow
   template patterns which may limit semantic diversity beyond a certain scale.

3. **Domain-specific**: Templates are optimized for IoT sensor analysis. Other domains
   would require new templates.

### MTN Sails Bridge

1. **Rule-based parsing limitations**: Without LLM, complex or ambiguous queries may be misinterpreted.

2. **Metric support**: Only supports metrics available in the LSTM model (temperature, humidity, light, barometer).

3. **Gradient analysis**: Requires sensor location data and multiple sensors.

---

## 🐛 Troubleshooting

### Training Data Generator Issues

#### Not reaching target count

If `--total-target` doesn't reach the desired count, try:
- Enable `--diversity-mode` for more metric types
- Increase `--augment-factor` for more variations
- Add custom templates to the source code

#### Reproducibility issues

- Ensure same Python version (tested with 3.12)
- Use same `--seed` value
- Use same command-line options

#### Performance issues

- For large datasets (10K+), allow 10-30 seconds for generation
- Use `--quiet` flag to reduce output overhead
- Consider generating in batches for very large datasets (100K+)

### MTN Sails Bridge Issues

#### "Failed to load Taber LSTM predictor"

**Solution**:
1. Verify model directory exists: `ls -la outputs/`
2. Check for model files: `model.onnx`, `scaler.pkl`, `buffer_data.parquet`
3. Ensure dependencies installed: `pip install onnxruntime pandas numpy`

#### "Gradient analyzer not available"

**Solution**:
1. Check for `gradient_analyzer.py` in pipeline directory
2. Ensure sensor location data available
3. Verify multiple sensors in dataset

#### "LLM model is required for query parsing"

The bridge no longer supports rule-based fallback.  Provide `--mtnsails-model`
with a trained ONNX model directory, and install LLM dependencies:

```bash
pip install optimum[onnxruntime] transformers
```

#### Import errors

**Solution**:
```bash
# Run from project root directory
cd /path/to/taber_enviro
python3 -m llm_interface.mtnsails_bridge --taber-model outputs/ --interactive
```

---

## 🚀 Future Enhancements

### Planned Features

1. **Training Data Generator**:
   - Add more template categories (trend analysis, comparative queries)
   - Implement GPT-based paraphrasing for unlimited variation
   - Add configuration file support for custom metrics and templates
   - Support for multi-language generation
   - Integration with active learning

2. **MTN Sails Bridge**:
   - Multi-sensor query support ("Compare temperature across all sensors")
   - Historical data queries ("What was temperature yesterday?")
   - Conditional queries ("Alert me if temperature exceeds 30°C")
   - Multi-metric analysis ("Show correlation between all metrics")
   - REST API endpoint for web integration

---

## 📚 Additional Resources

- **Example Scripts**: `llm_interface/examples/example_queries.py`
- **Test Suite**: `tests/test_training_data_generator.py`
- **Source Code**: 
  - Generator: `llm_interface/training_data_generator.py`
  - Bridge: `llm_interface/mtnsails_bridge.py`

---

## 💡 Tips and Best Practices

### Training Data Generation

1. **Start small**: Test with 1000 examples before generating 10K+
2. **Use reproducible seeds**: Always set `--seed` for consistent results
3. **Enable diversity mode**: For production models, use `--diversity-mode`
4. **Monitor statistics**: Check vocabulary size and category distribution
5. **Validate output**: Manually inspect sample queries for quality

### MTN Sails Bridge

1. **Start without LLM**: Test rule-based mode first to verify LSTM integration
2. **Use interactive mode**: Experiment with query patterns before automation
3. **Check intent extraction**: Monitor console output to verify query parsing
4. **Handle errors gracefully**: Wrap `bridge.ask()` in try-except blocks
5. **Implement caching**: For repeated queries, cache responses

### Production Deployment

1. **Model versioning**: Track both generator settings and trained model versions
2. **Monitoring**: Log query patterns and response quality
3. **Fallback strategies**: Implement fallbacks for failed queries
4. **Rate limiting**: Protect against excessive API calls
5. **Data privacy**: Sanitize queries containing sensitive information

---

## 📄 License

See project LICENSE file.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## 📞 Support

For issues or questions:
1. Check test suite: `python3 tests/test_training_data_generator.py -v`
2. Review examples in this documentation
3. Examine generated examples for quality
4. Open an issue in the repository

---

**Let's have fun coding!** 🎉
