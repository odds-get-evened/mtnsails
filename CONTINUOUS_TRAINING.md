# Continuous Training Guide

## Overview

The MTN Sails system now supports **continuous training**, allowing you to build knowledge cumulatively across multiple training sessions. Instead of starting from scratch each time, the system automatically detects existing models and continues training from where you left off.

## Key Benefits

- 🔄 **Cumulative Learning**: Each training session builds on previous knowledge
- 🧠 **Preserved Knowledge**: Lower learning rate prevents catastrophic forgetting
- 🚀 **Faster Training**: Start from a trained model instead of base model
- 📈 **Progressive Improvement**: Model gets better with each training session
- 💡 **Automatic Detection**: No manual configuration required

## How It Works

### Automatic Model Detection

Before training, the system checks for existing models:

1. **For `train` command**: Checks `--output-dir` for trained model
2. **For `pipeline` command**: Checks both `--onnx-output` and `--output-dir`

A model is considered valid if it has:
- `config.json` (model configuration)
- `model.safetensors` OR `pytorch_model.bin` (model weights)

### Learning Rate Strategy

| Scenario | Learning Rate | Purpose |
|----------|---------------|---------|
| New training (from base) | 5e-5 (default) | Standard training |
| Retraining (from checkpoint) | 1e-5 (automatic) | Fine-tuning without forgetting |

The system automatically uses a lower learning rate (1/5th of default) when retraining to preserve existing knowledge while learning new information.

## Usage Examples

### Example 1: Progressive Training Sessions

**First Session** (train from scratch):
```bash
python main.py pipeline --data-file programming_basics.json --epochs 3
```
Output:
```
🆕 No existing model found. Training from base model 'distilgpt2'
📚 Using standard learning rate (5e-05) for initial training
```

**Second Session** (continue learning):
```bash
python main.py pipeline --data-file web_development.json --epochs 2
```
Output:
```
🔄 Found existing ONNX model at ./onnx_model
🔄 Found existing trained model at ./trained_model
🔄 Continuing training from previous checkpoint...
📚 Using lower learning rate (1e-05) for fine-tuning to preserve existing knowledge
```

**Third Session** (more knowledge):
```bash
python main.py pipeline --data-file databases.json --epochs 2
```
Output:
```
🔄 Found existing trained model at ./trained_model
🔄 Continuing training from this checkpoint...
📚 Using lower learning rate (1e-05) for fine-tuning to preserve existing knowledge
```

Result: Model now understands programming basics, web development, AND databases!

### Example 2: Using Train Command

```bash
# First training
python main.py train --data-file dataset1.json --epochs 3

# Add more knowledge
python main.py train --data-file dataset2.json --epochs 2
```

The `train` command also supports continuous training and will detect existing models at `--output-dir`.

### Example 3: Custom Directories

```bash
# Use custom directories
python main.py pipeline \
  --data-file mydata.json \
  --output-dir ./my_model \
  --onnx-output ./my_onnx \
  --epochs 2
```

The system will check `./my_model` for existing trained models.

## Behavior in Different Scenarios

### Scenario 1: Fresh Start
```
State: No models exist
Behavior: Train from base model (distilgpt2)
Learning Rate: 5e-5 (standard)
```

### Scenario 2: Trained Model Exists
```
State: ./trained_model contains valid model
Behavior: Continue training from checkpoint
Learning Rate: 1e-5 (fine-tuning)
```

### Scenario 3: Both Models Exist
```
State: Both ./trained_model and ./onnx_model exist
Behavior: Continue training from trained model
Learning Rate: 1e-5 (fine-tuning)
Note: ONNX will be updated after training
```

### Scenario 4: ONNX Only (Recovery)
```
State: ./onnx_model exists but ./trained_model deleted
Behavior: Train from base model (can't retrain from ONNX)
Learning Rate: 5e-5 (standard)
Warning: ⚠️  Source trained model not found
```

### Scenario 5: Incomplete Model
```
State: ./trained_model exists but missing required files
Behavior: Treat as fresh start
Learning Rate: 5e-5 (standard)
```

## Best Practices

### ✅ Do's

1. **Keep Trained Models**: Don't delete `./trained_model` if you plan to continue training
2. **Incremental Training**: Add knowledge in stages with multiple training sessions
3. **Monitor Quality**: Use `validate` command to check data quality before each session
4. **Smaller Epochs**: Use fewer epochs (1-2) for retraining sessions
5. **Test After Each Session**: Verify model behavior after each training round

### ❌ Don'ts

1. **Don't Delete Trained Models**: ONNX models can't be used for retraining
2. **Don't Mix Incompatible Data**: Ensure new data is compatible with existing knowledge
3. **Don't Overtrain**: Too many epochs on small datasets can cause overfitting
4. **Don't Skip Validation**: Always validate data quality first

## Advanced Usage

### Override Model Detection

If you want to train from scratch even when models exist:

```bash
# Delete existing models first
rm -rf ./trained_model ./onnx_model

# Or use a different output directory
python main.py pipeline --data-file data.json --output-dir ./new_model
```

### Explicit Model Selection

You can override automatic detection by specifying a different base model:

```bash
python main.py train --model-name gpt2 --data-file data.json
```

This will use `gpt2` regardless of existing models (though detection still happens internally).

### Custom Learning Rate

Override the automatic learning rate if needed:

```bash
python main.py train --data-file data.json --learning-rate 2e-5
```

Note: The system will still detect existing models and load from them, but will use your specified learning rate.

## Troubleshooting

### Problem: "Model exists but training from scratch"

**Cause**: Incomplete model files (missing config.json or model weights)

**Solution**: Ensure your trained_model directory contains:
- `config.json`
- `model.safetensors` OR `pytorch_model.bin`

### Problem: "ONNX exists but source model not found"

**Cause**: Trained model was deleted after ONNX conversion

**Solution**: 
- Keep trained models for retraining
- Or retrain from scratch (system will handle this automatically)

### Problem: "Model quality degraded after retraining"

**Cause**: 
- Low quality training data
- Too many epochs
- Learning rate too high

**Solution**:
- Validate data quality with `validate` command
- Use fewer epochs (1-2) for retraining
- Trust the automatic learning rate (1e-5)

## Technical Details

### Model Validation Function

```python
def check_model_exists(model_path: str) -> bool:
    """Check if a trained model exists at the given path."""
    model_path = Path(model_path)
    
    # Check for required files
    has_config = (model_path / "config.json").exists()
    has_safetensors = (model_path / "model.safetensors").exists()
    has_pytorch = (model_path / "pytorch_model.bin").exists()
    
    return has_config and (has_safetensors or has_pytorch)
```

This function validates that a model directory contains the minimum required files for loading.

### Learning Rate Logic

```python
if check_model_exists(args.output_dir):
    model_to_use = args.output_dir  # Load from existing
    learning_rate_to_use = 1e-5      # Lower rate for fine-tuning
else:
    model_to_use = args.model_name   # Load base model
    learning_rate_to_use = args.learning_rate  # Standard rate
```

### Files Preserved

The following files are preserved in `./trained_model`:
- `config.json` - Model configuration
- `model.safetensors` - Model weights (preferred format)
- `pytorch_model.bin` - Model weights (fallback format)
- `tokenizer.json` - Tokenizer configuration
- `vocab.json` - Vocabulary
- Other tokenizer files

## FAQ

**Q: Can I continue training from an ONNX model?**
A: No, ONNX is for inference only. Keep the trained model in `./trained_model` for retraining.

**Q: How many training sessions can I do?**
A: Unlimited! Each session builds on previous knowledge.

**Q: Will old knowledge be forgotten?**
A: The lower learning rate (1e-5) minimizes forgetting, but some degradation is possible. Monitor quality.

**Q: Can I train on different topics?**
A: Yes! The model will learn multiple topics. Just ensure data quality is good.

**Q: What if I want to start fresh?**
A: Delete `./trained_model` and `./onnx_model` directories, or use a different `--output-dir`.

## Related Documentation

- [QUICKSTART.md](QUICKSTART.md) - Getting started guide
- [DATA_QUALITY_GUIDE.md](DATA_QUALITY_GUIDE.md) - Data quality validation
- [API_REFERENCE.md](API_REFERENCE.md) - Programming API
- [IMPLEMENTATION.md](IMPLEMENTATION.md) - Technical implementation details
