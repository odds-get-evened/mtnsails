# Scraped Content Processing Feature

This feature allows you to process content downloaded by [scrapyer](https://github.com/odds-get-evened/scrapyer) into conversational JSON format suitable for training with mtnsails.

## Overview

The `process_scraped_content.py` script converts plain text files (downloaded by scrapyer) into the conversational JSON format required by mtnsails for LLM training. It automatically:

1. Reads all `.txt` files from a directory
2. Extracts topics from filenames or content
3. Generates conversational input/output pairs
4. Removes duplicate titles to avoid echo issues
5. Saves the result in mtnsails-compatible JSON format

## Installation

No additional dependencies are required beyond the base mtnsails requirements.

## Usage

### Basic Usage

```bash
# Process scraped content from a directory
python process_scraped_content.py /path/to/scraped/files

# This creates chat_data.json in the current directory
```

### Custom Output File

```bash
# Specify a custom output filename
python process_scraped_content.py /path/to/scraped/files --output my_training_data.json
```

### Custom Prompt Template

```bash
# Use a custom prompt template
python process_scraped_content.py /path/to/scraped/files --prompt-template "Explain {topic}"
```

The `{topic}` placeholder will be replaced with the extracted topic from each file.

## Workflow with Scrapyer

### Step 1: Download Content with Scrapyer

First, use scrapyer to download and extract content from web pages:

```bash
# Install scrapyer
pip install git+https://github.com/odds-get-evened/scrapyer.git

# Scrape content from a URL
scrapyer "http://example.com/article" /path/to/output/directory/
```

Scrapyer will create plain text files in the specified directory.

### Step 2: Process into Conversational Format

Use the mtnsails processor to convert the text files:

```bash
# Process the scraped content
python process_scraped_content.py /path/to/output/directory/ --output chat_data.json
```

### Step 3: Validate and Train

```bash
# Validate the data quality
python main.py validate --data-file chat_data.json

# Train your model
python main.py train --data-file chat_data.json --epochs 3

# Convert to ONNX
python main.py convert --model-path ./trained_model --onnx-output ./onnx_model

# Chat with your model
python main.py chat --model-path ./onnx_model
```

## Output Format

The script generates JSON files in the following format:

```json
[
  {
    "input": "Tell me about Python Programming",
    "output": "Python is a high-level, interpreted programming language..."
  },
  {
    "input": "Tell me about Machine Learning",
    "output": "Machine learning is a subset of artificial intelligence..."
  }
]
```

This format is directly compatible with mtnsails' training pipeline.

## Features

### Automatic Topic Extraction

The script intelligently extracts topics from:
1. **Content headers**: First line or heading in the text
2. **Filenames**: If no suitable header is found, uses the filename
3. **Fallback**: Uses generic prompts if neither works

### Content Cleaning

The script automatically:
- Removes duplicate titles/headers
- Cleans excessive whitespace
- Removes common web artifacts (cookie notices, etc.)
- Handles various text encodings

### Quality Checks

- Skips files with content shorter than 50 characters
- Reports processing statistics
- Provides clear error messages for failed files

## Command-Line Options

```
usage: process_scraped_content.py [-h] [--output OUTPUT] [--prompt-template PROMPT_TEMPLATE] input_directory

positional arguments:
  input_directory       Directory containing plain text files from scrapyer

optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT, -o OUTPUT
                        Output JSON file path (default: chat_data.json)
  --prompt-template PROMPT_TEMPLATE, -p PROMPT_TEMPLATE
                        Template for input prompts. Use {topic} as placeholder
                        (default: "Tell me about {topic}")
```

## Examples

### Example 1: Process Technical Documentation

```bash
# Scrape Python documentation
scrapyer "https://docs.python.org/3/tutorial/" /tmp/python_docs/

# Process into training data
python process_scraped_content.py /tmp/python_docs/ --output python_training.json

# Train a Python expert model
python main.py train --data-file python_training.json --epochs 5
```

### Example 2: Build a Domain-Specific Assistant

```bash
# Scrape multiple related articles
scrapyer "https://example.com/article1" /tmp/articles/
scrapyer "https://example.com/article2" /tmp/articles/
scrapyer "https://example.com/article3" /tmp/articles/

# Process all articles
python process_scraped_content.py /tmp/articles/ --output domain_data.json \
  --prompt-template "What can you tell me about {topic}?"

# Validate and train
python main.py validate --data-file domain_data.json
python main.py train --data-file domain_data.json
```

### Example 3: Build FAQ Bot from Documentation

```bash
# Scrape FAQ pages
scrapyer "https://mycompany.com/faq" /tmp/faq/

# Process with question-style prompts
python process_scraped_content.py /tmp/faq/ --output faq_bot.json \
  --prompt-template "Can you help me with {topic}?"

# Train FAQ bot
python main.py train --data-file faq_bot.json --epochs 3
python main.py convert --model-path ./trained_model --onnx-output ./faq_bot
```

## Tips and Best Practices

1. **Use Descriptive Filenames**: Scrapyer often uses URL-based names. Rename files to have descriptive names if possible.

2. **Quality Over Quantity**: A few high-quality, well-written articles are better than many poorly formatted ones.

3. **Always Validate**: Run `python main.py validate --data-file <file>` before training to check data quality.

4. **Customize Prompts**: Different prompt templates work better for different types of content:
   - Technical docs: `"Explain {topic}"`
   - FAQs: `"What is {topic}?"`
   - Tutorials: `"How do I {topic}?"`

5. **Filter Low Quality**: If validation shows quality issues, use the `--filter` flag to remove bad conversations.

## Troubleshooting

### No Files Processed

- **Issue**: "No .txt files found in directory"
- **Solution**: Ensure scrapyer created `.txt` files (not `.html` or other formats)

### Content Too Short

- **Issue**: Files are being skipped as "content too short"
- **Solution**: Check that the text files contain substantial content (>50 characters)

### Echo Output Warnings

- **Issue**: Validation reports "Output echoes the input"
- **Solution**: This is usually a false positive when the topic appears in both input and output. It's generally fine unless the entire output is just repeating the input.

### Poor Quality Responses

- **Issue**: Trained model gives poor responses
- **Solution**: 
  - Check the source content quality
  - Use more diverse training data
  - Increase training epochs
  - Validate data before training

## Integration with Existing Workflows

This feature seamlessly integrates with the existing mtnsails workflow:

1. **Data Collection**: Use scrapyer to gather content
2. **Processing**: Use `process_scraped_content.py` to format data
3. **Validation**: Use `main.py validate` to check quality
4. **Training**: Use `main.py train` as normal
5. **Deployment**: Use `main.py convert` and `main.py chat` as normal

## License

This feature is part of mtnsails and is licensed under the GNU General Public License v3.0.

## Contributing

Contributions are welcome! Please ensure:
- The script maintains compatibility with mtnsails' JSON format
- Code follows the project's OOP principles
- Changes include appropriate error handling
- Documentation is updated
