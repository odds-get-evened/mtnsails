# Data Quality Validation Guide

## Quick Start: How to Validate Your Data

**Before you train, validate your data quality:**

```bash
python main.py validate --data-file your_data.json
```

This command will:
- ✓ Analyze all conversations in your file
- ✓ Calculate a quality score (0-100%)
- ✓ Show specific issues found (repetition, gibberish, empty responses, etc.)
- ✓ Provide actionable recommendations
- ✓ Exit with status 0 if quality is good, 1 if critical

**Example:**
```bash
# Validate your training data
$ python main.py validate --data-file my_conversations.json

=== Data Quality Validation ===
Loaded 42 conversations from my_conversations.json

Total conversations: 42
Valid conversations: 38
Invalid conversations: 4
Quality score: 90.5%

✅ DATA QUALITY: GOOD
Your data quality is good. You can proceed with training.
```

---

## The Problem: "Garbage In, Garbage Out"

When training language models, **data quality is everything**. If you train on low-quality conversation logs, your model will learn to produce similar low-quality outputs.

### Real-World Example

A user reported that their chatbot was producing nonsense:

**Training Data (from chat logs):**
```json
[
  {
    "input": "hello there. i am chris",
    "output": "yes i"
  },
  {
    "input": "you are?",
    "output": "You are a computer programmer. You are a computer programmer. You are a computer programmer..."
  },
  {
    "input": "stop repeating yourself please",
    "output": "stop repeating yourself please: stop repeating yourself please: stop repeating..."
  },
  {
    "input": "haha",
    "output": "http://"
  },
  {
    "input": "ugh nonsense",
    "output": ""
  }
]
```

**Training Results:**
```
Loss: 2.6021 → 2.3755 → 2.0374 → 1.9466 (still high after 3 epochs)
```

**The Problem:** The model learned to reproduce the same bad patterns from the training data:
- Repetitive responses
- Empty outputs
- Gibberish
- Echo/copying user input

This creates a **downward spiral**: bad model → bad outputs → train on bad outputs → worse model.

## The Solution

MTN Sails now includes **automatic data quality validation** that detects and warns you about problematic training data before you waste time training.

### What Gets Detected

1. **Repetitive Text** - Responses that repeat the same phrases over and over
   - Example: "hello hello hello hello hello"
   
2. **Empty Responses** - Conversations with no output
   - Example: `{"input": "test", "output": ""}`

3. **Gibberish** - Nonsensical text, URL fragments, or mostly non-alphabetic characters
   - Example: "http://", "!@#$%^&*()", ""

4. **Echo Responses** - Outputs that simply repeat the user's input
   - Example: Input: "stop repeating", Output: "stop repeating stop repeating"

5. **Too Short** - Meaningless one or two-word responses
   - Example: "yes i", "ok", "no"

### How It Works

When you run training, the system automatically:

1. **Analyzes all conversations** in your training data
2. **Calculates a quality score** (0% = all bad, 100% = all good)
3. **Identifies specific issues** with examples
4. **Shows a detailed report** before training starts
5. **Warns or blocks training** if quality is too low

### Quality Score Thresholds

- **100% - 70%**: ✓ Good quality, proceed with training
- **69% - 50%**: ⚠️  Warning shown, training proceeds
- **Below 50%**: 🛑 Critical warning, training blocked (unless --force used)

## Usage Examples

### Example 1: Training with Good Data

```bash
$ python main.py train --data-file examples/example_conversations.json --epochs 3

=== Training Model ===
Loaded 8 conversations

=== Data Quality Analysis ===
Total conversations: 8
Valid conversations: 8
Invalid conversations: 0
Quality score: 100.0%

Recommendations:
  ✓ Dataset quality is acceptable

Starting training...
```

### Example 2: Training with Bad Data

```bash
$ python main.py train --data-file tests/fixtures/bad_quality_data.json --epochs 3

=== Training Model ===
Loaded 5 conversations

=== Data Quality Analysis ===
Total conversations: 5
Valid conversations: 0
Invalid conversations: 5
Quality score: 0.0%

Issues detected:
  - Empty Outputs: 1
  - Repetitive Outputs: 2
  - Short Outputs: 2
  - Gibberish Outputs: 1
  - Echo Outputs: 1

Example problematic conversations:

  Conversation #0:
    Input: hello there. i am chris
    Output: yes i
    Issues: Output is too short

  Conversation #1:
    Input: you are?
    Output: You are not a computer programmer. You are a computer programmer...
    Issues: Output is highly repetitive

Recommendations:
  ⚠️  CRITICAL: Less than 50% of conversations are high quality
  Training on this data will likely produce a poorly performing model
  Consider filtering or regenerating your training data
  High repetition detected in 2 conversations
  1 conversations have empty outputs

======================================================================
⚠️  CRITICAL WARNING: DATA QUALITY IS VERY LOW
======================================================================
Training on this data will likely result in a model that produces:
  - Nonsense or gibberish responses
  - Repetitive text
  - Echoes of user input

This is a 'garbage in, garbage out' situation.

RECOMMENDATIONS:
  1. Filter out low-quality conversations
  2. Use only conversations with meaningful, coherent responses
  3. Avoid training on chat logs with nonsense outputs
  4. Start with high-quality example conversations
======================================================================

Do you want to continue anyway? (yes/no): no
Training cancelled. Please improve your data quality first.
```

### Example 3: Running the Quality Demo

```bash
$ python examples/demo_data_quality.py

================================================================================
Data Quality Validation Demo
================================================================================

1. Testing with HIGH QUALITY data (example_conversations.json):
--------------------------------------------------------------------------------
Total conversations: 8
Valid conversations: 8
Invalid conversations: 0
Quality score: 100.0%

2. Testing with LOW QUALITY data (bad_quality_data.json):
--------------------------------------------------------------------------------
Total conversations: 5
Valid conversations: 0
Invalid conversations: 5
Quality score: 0.0%

[Shows detailed comparison and recommendations]
```

## Best Practices

### DO ✓

- **Start with high-quality, curated conversations**
- **Review your data before training** - don't trust chat logs blindly
- **Filter out problematic conversations** from logged data
- **Use the quality report** to identify and fix issues
- **Aim for quality scores above 80%** for best results

### DON'T ✗

- **Don't train on raw chat logs** from a poorly performing model
- **Don't ignore quality warnings** - they're there for a reason
- **Don't use --force** unless you know what you're doing
- **Don't expect good results** from low-quality data
- **Don't retrain on bad outputs** - this creates a downward spiral

## Troubleshooting

### "My model produces repetitive text"

**Cause:** You trained on data with repetitive responses.

**Solution:**
1. Check your training data quality
2. Remove conversations with repetitive outputs
3. Retrain from scratch with clean data

### "My model echoes what I say"

**Cause:** Your training data contains echo responses.

**Solution:**
1. Filter out conversations where output echoes input
2. Use the quality validator to identify these
3. Retrain with filtered data

### "My model gives very short or empty responses"

**Cause:** Your training data has short/empty outputs.

**Solution:**
1. Remove conversations with empty or very short outputs
2. Ensure all training responses are complete, meaningful sentences
3. Retrain with better data

### "Training loss stays high (> 2.0)"

**Cause:** Often indicates poor data quality or mismatch.

**Solution:**
1. Check the quality report - likely shows low score
2. Review and improve your training data
3. Make sure data follows the expected format

## Programmatic Usage

You can also use the quality validation in your own scripts:

```python
from src.data_handler import ConversationDataHandler

# Load data
handler = ConversationDataHandler()
handler.load_from_json('my_data.json')

# Analyze quality
report = handler.analyze_dataset_quality()

print(f"Quality Score: {report['quality_score']:.1%}")
print(f"Valid: {report['valid_conversations']}")
print(f"Invalid: {report['invalid_conversations']}")

# Check individual conversations
for conv in handler.conversations:
    is_valid, issues = handler.validate_conversation_quality(conv)
    if not is_valid:
        print(f"Issues: {issues}")
```

## Summary

Data quality validation helps you:
- **Avoid wasting time** training on bad data
- **Prevent the "garbage in, garbage out" problem**
- **Identify specific issues** in your training data
- **Improve model performance** by using only high-quality conversations
- **Understand why** your model might be performing poorly

Always review the quality report and aim for high-quality training data!
