# Loss Configuration Guide

## Understanding Loss Functions in MTN Sails

### Current Implementation

MTN Sails uses the **default cross-entropy loss** for causal language modeling, which is the standard and recommended approach for GPT-style models. This loss function is automatically computed by the model during training.

### Why Cross-Entropy Loss?

Cross-entropy loss is the optimal choice for causal language modeling because:

1. **Probabilistic Foundation**: It measures the difference between predicted token probabilities and actual tokens
2. **Proven Effectiveness**: Used by all major language models (GPT-2, GPT-3, LLaMA, etc.)
3. **Gradient Properties**: Provides stable gradients for effective learning
4. **Automatic Handling**: Transformers library handles it automatically

### The `loss_type` Parameter

**Important**: The `loss_type` parameter in TrainingArguments is **deprecated** in recent versions of transformers and should **not** be used.

- Setting `loss_type=None` triggers a deprecation warning (which MTN Sails suppresses)
- The model automatically uses the correct loss function based on its architecture
- For causal LMs (GPT-2, DistilGPT-2), this is always cross-entropy loss

## Alternative Approaches to Improve Model Performance

Instead of changing the loss function, consider these proven techniques:

### 1. **Improve Data Quality** ⭐ Most Important

Use the built-in data quality validation:

```bash
python main.py validate --data-file data.json --filter
```

High-quality training data has the biggest impact on model performance. See the README's "Best Practices" section for details.

### 2. **Adjust Learning Rate**

The learning rate controls how quickly the model learns:

```bash
# Default learning rate
python main.py train --learning-rate 5e-5

# Lower learning rate (more stable, slower)
python main.py train --learning-rate 2e-5

# Higher learning rate (faster, less stable)
python main.py train --learning-rate 1e-4
```

**Recommendations**:
- Start with default: `5e-5`
- If training is unstable: try `2e-5` or `1e-5`
- If training is too slow: try `1e-4` (monitor for instability)

### 3. **Increase Training Epochs**

More epochs allow the model to learn patterns better:

```bash
# Quick test
python main.py train --epochs 2

# Standard training
python main.py train --epochs 3

# Thorough training
python main.py train --epochs 5-10
```

**Note**: Too many epochs can lead to overfitting on small datasets.

### 4. **Adjust Batch Size**

Batch size affects training stability and speed:

```bash
# Smaller batch (more stable, slower)
python main.py train --batch-size 2

# Default
python main.py train --batch-size 4

# Larger batch (faster, requires more RAM)
python main.py train --batch-size 8
```

### 5. **Use More Training Data**

- Aim for at least 50-100 high-quality conversations
- Diverse examples help the model generalize
- Filter out low-quality data using `--filter` option

### 6. **Try Different Base Models**

```bash
# Smallest (fastest, less capable)
python main.py train --model-name distilgpt2

# Medium (good balance)
python main.py train --model-name gpt2

# Larger (more capable, slower)
python main.py train --model-name gpt2-medium
```

## Advanced: Custom Loss Functions

If you need a custom loss function for specialized use cases, you can extend the `LLMTrainer` class:

```python
from src.trainer import LLMTrainer
from transformers import Trainer
import torch

class CustomLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """Override loss computation for custom behavior."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Standard cross-entropy
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        # Optionally add custom modifications here
        # For example, label smoothing, focal loss, etc.
        
        return (loss, outputs) if return_outputs else loss
```

However, **this is rarely needed** for conversational AI applications.

## Practical Example: Optimizing Training

Here's a step-by-step workflow to optimize your model training:

### Step 1: Start with Baseline

```bash
# Train with default settings
python main.py train --data-file my_data.json

# Test the model
python main.py convert --model-path ./trained_model --onnx-output ./onnx_model
python main.py chat --model-path ./onnx_model --prompt "Hello, how are you?"
```

### Step 2: If Results are Poor

**First, check data quality:**
```bash
python main.py validate --data-file my_data.json
```

If quality score is < 70%, filter your data:
```bash
python main.py validate --data-file my_data.json --filter
python main.py train --data-file my_data_filtered.json
```

### Step 3: Tune Learning Rate

If training seems unstable (loss jumps around):
```bash
python main.py train --data-file my_data.json --learning-rate 2e-5
```

If training is too slow (loss decreases very slowly):
```bash
python main.py train --data-file my_data.json --learning-rate 1e-4
```

### Step 4: Increase Training Duration

For better results with sufficient data:
```bash
python main.py train --data-file my_data.json --epochs 5 --learning-rate 3e-5
```

### Step 5: Monitor Training

Watch the loss values during training. Good training should show:
- Loss starting around 3-5 for untrained model
- Loss decreasing steadily
- Final loss around 1-2 for well-trained model on quality data
- Loss < 1.0 indicates very good fit (may overfit on small datasets)

If loss stays high (>3) after several epochs:
- Your data quality may be poor
- You may need more training epochs
- Consider adjusting learning rate

## Summary

**For 99% of use cases, the default cross-entropy loss is optimal.**

To improve your model's performance, focus on:

1. ✅ **Data quality** - Most important factor
2. ✅ **Learning rate tuning** - Adjust for stability
3. ✅ **Sufficient training epochs** - Allow adequate learning
4. ✅ **Adequate training data** - 50+ high-quality examples
5. ✅ **Model size selection** - Balance capability and speed

**Don't** try to change the loss function unless you have a specific, advanced research use case.

## Suppressing the Loss Type Warning

MTN Sails already suppresses the `loss_type` deprecation warning in `src/trainer.py`:

```python
warnings.filterwarnings('ignore', message='.*loss_type.*')
```

This is intentional and correct. The warning is a deprecation notice from transformers, not an error.

## Questions?

- For general training help: See [README.md](README.md)
- For API documentation: See [API_REFERENCE.md](API_REFERENCE.md)
- For data quality: Run `python demo_data_quality.py`
