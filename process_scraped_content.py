#!/usr/bin/env python3
"""
Feature script to process scraped content from scrapyer into conversational JSON format.

This script reads plain text files downloaded by scrapyer and converts them into
the conversational JSON format (input/output pairs) required by mtnsails for training.

Features:
- Randomized prompt templates for natural conversation variety
- Multiple placeholder options: {topic}, {source}, {content_type}, {category}, {filename}
- Automatic metadata extraction from content and filenames
- Content classification and categorization

Usage:
    python process_scraped_content.py <input_directory> [--output chat_data.json] [--prompt-template TEMPLATE] [--no-randomize]

Examples:
    # Use randomized prompts (default behavior)
    python process_scraped_content.py /path/to/scraped/files --output chat_data.json
    
    # Use custom template with multiple placeholders
    python process_scraped_content.py /path/to/scraped/files --prompt-template "What is {topic} from {source}?"
    
    # Disable randomization for consistent prompts
    python process_scraped_content.py /path/to/scraped/files --no-randomize
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
import re
import random
from urllib.parse import urlparse


class ScrapedContentProcessor:
    """Processes scraped content into conversational JSON format."""
    
    # Default prompt templates with various phrasings
    DEFAULT_PROMPT_TEMPLATES = [
        "Tell me about {topic}",
        "What is {topic}?",
        "Explain {topic}",
        "Can you describe {topic}?",
        "What can you tell me about {topic}?",
        "I'd like to know about {topic}",
        "Please explain {topic}",
        "Give me information about {topic}",
        "What do you know about {topic}?",
        "Provide details on {topic}",
        "Help me understand {topic}",
        "Tell me more about {topic}",
        "What should I know about {topic}?",
        "Describe {topic} for me",
        "I'm interested in {topic}",
    ]
    
    def __init__(self, prompt_template: Optional[str] = None, randomize_prompts: bool = True):
        """
        Initialize the processor.
        
        Args:
            prompt_template: Template for generating input prompts.
                           If None, uses randomized templates from DEFAULT_PROMPT_TEMPLATES.
                           If provided, uses this specific template for all conversations.
            randomize_prompts: If True and prompt_template is None, randomly selects from
                             DEFAULT_PROMPT_TEMPLATES for each conversation. If False, uses
                             first template from the list.
        """
        self.prompt_template = prompt_template
        self.randomize_prompts = randomize_prompts
        self.conversations = []
    
    def extract_topic_from_filename(self, filename: str) -> str:
        """
        Extract a topic from a filename.
        
        Args:
            filename: The filename to extract topic from
            
        Returns:
            A cleaned topic string
        """
        # Remove file extension
        topic = Path(filename).stem
        
        # Replace common separators with spaces
        topic = topic.replace('_', ' ').replace('-', ' ')
        
        # Remove URL-like patterns
        topic = re.sub(r'https?://', '', topic)
        topic = re.sub(r'www\.', '', topic)
        
        # Clean up multiple spaces
        topic = re.sub(r'\s+', ' ', topic).strip()
        
        # Capitalize first letter
        if topic:
            topic = topic[0].upper() + topic[1:]
        
        return topic
    
    def extract_topic_from_content(self, content: str) -> str:
        """
        Extract a topic from the content (e.g., first heading or first line).
        
        Args:
            content: The text content
            
        Returns:
            A topic string extracted from content
        """
        lines = content.strip().split('\n')
        
        # Try to find a heading (lines that look like titles)
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line and not line.startswith('#'):
                # If line is short and looks like a title
                if len(line) < 100 and len(line.split()) < 15:
                    return line
        
        # Fallback: use first non-empty line
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:
                # Truncate if too long
                if len(line) > 100:
                    line = line[:100] + "..."
                return line
        
        return "this topic"
    
    def extract_metadata(self, file_path: Path, content: str, topic: str) -> Dict[str, str]:
        """
        Extract metadata that can be used as placeholders in prompt templates.
        
        Args:
            file_path: Path to the file
            content: Content of the file
            topic: Extracted topic
            
        Returns:
            Dictionary with placeholder values
        """
        # Extract potential domain/source from filename
        filename = file_path.stem
        
        # Try to extract domain or source identifier from URL-like filenames
        source = "unknown source"
        
        # First, try to find domain after protocol in underscore/dash separated format
        # e.g., "https_example_com_article" -> "example"
        protocol_pattern = r'https?[_\-](?:www[_\-.])?([a-zA-Z0-9]+)'
        match = re.search(protocol_pattern, filename.lower())
        
        if match:
            source = match.group(1)
        else:
            # Try standard URL pattern with word boundaries
            url_pattern = r'\b([a-zA-Z0-9-]+)(?:[_.][a-zA-Z]+)\b'
            match = re.search(url_pattern, filename)
            
            if match and match.group(1).lower() not in ['http', 'https', 'www']:
                source = match.group(1)
            elif filename:
                # Use first meaningful part of filename if no URL found
                parts = re.split(r'[_\-\s]+', filename)
                if parts and parts[0].lower() not in ['http', 'https', 'www']:
                    source = parts[0]
        
        # Determine content type based on length and structure
        word_count = len(content.split())
        if word_count < 200:
            content_type = "brief explanation"
        elif word_count < 500:
            content_type = "article"
        elif word_count < 1000:
            content_type = "detailed guide"
        else:
            content_type = "comprehensive resource"
        
        # Try to infer category from content keywords
        content_lower = content.lower()
        category = "general information"
        
        # Simple keyword-based categorization
        if any(word in content_lower for word in ['tutorial', 'how to', 'step', 'guide']):
            category = "tutorial"
        elif any(word in content_lower for word in ['documentation', 'reference', 'api']):
            category = "documentation"
        elif any(word in content_lower for word in ['concept', 'theory', 'principle', 'introduction']):
            category = "concept"
        elif any(word in content_lower for word in ['example', 'sample', 'demo']):
            category = "example"
        
        return {
            'topic': topic,
            'source': source,
            'content_type': content_type,
            'category': category,
            'filename': file_path.name,
        }
    
    def clean_content(self, content: str, remove_first_line: bool = False) -> str:
        """
        Clean and normalize content.
        
        Args:
            content: Raw text content
            remove_first_line: If True, removes the first line (useful if it's a title)
            
        Returns:
            Cleaned content
        """
        # Optionally remove first line if it looks like a title
        if remove_first_line:
            lines = content.split('\n')
            if len(lines) > 1:
                # Remove first line if it's relatively short (likely a title)
                first_line = lines[0].strip()
                if len(first_line) < 150 and not first_line.endswith('.'):
                    content = '\n'.join(lines[1:])
        
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        
        # Remove common web artifacts
        content = re.sub(r'Cookie Policy.*?Accept', '', content, flags=re.IGNORECASE)
        content = re.sub(r'Subscribe to newsletter', '', content, flags=re.IGNORECASE)
        
        # Trim
        content = content.strip()
        
        return content
    
    def format_prompt(self, template: str, metadata: Dict[str, str], filename: str) -> str:
        """
        Format a prompt template with metadata placeholders.
        
        Args:
            template: The template string with placeholders
            metadata: Dictionary of placeholder values
            filename: Filename for warning messages
            
        Returns:
            Formatted prompt string
        """
        try:
            return template.format(**metadata)
        except KeyError as e:
            # Warn user about unsupported placeholders and use fallback
            missing_key = str(e).strip("'")
            print(f"  Warning for {filename}: Template uses unsupported placeholder '{missing_key}'. Using fallback with {{topic}} only.")
            # Fallback to using only the topic placeholder
            try:
                return template.format(topic=metadata['topic'])
            except KeyError:
                # If even topic isn't in template, just use the topic value directly
                return f"Tell me about {metadata['topic']}"
    
    def process_file(self, file_path: Path) -> Optional[Dict[str, str]]:
        """
        Process a single text file into a conversation.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            A conversation dictionary or None if processing fails
        """
        try:
            # Read content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Skip if content is too short
            if len(content.strip()) < 50:
                print(f"  Skipping {file_path.name}: content too short")
                return None
            
            # Extract topic first (before cleaning to get the title)
            topic = self.extract_topic_from_content(content)
            if topic == "this topic" or len(topic) < 5:
                topic = self.extract_topic_from_filename(file_path.name)
            
            # Clean content and remove first line if it looks like a title
            # This prevents echo issues where the output starts with the same text as the input
            content = self.clean_content(content, remove_first_line=True)
            
            # Extract metadata for placeholders
            metadata = self.extract_metadata(file_path, content, topic)
            
            # Select prompt template
            if self.prompt_template:
                # Use user-provided template
                template = self.prompt_template
            elif self.randomize_prompts:
                # Randomly select from available templates
                template = random.choice(self.DEFAULT_PROMPT_TEMPLATES)
            else:
                # Use first template as default
                template = self.DEFAULT_PROMPT_TEMPLATES[0]
            
            # Generate input prompt with available placeholders
            input_text = self.format_prompt(template, metadata, file_path.name)
            
            # Create conversation
            conversation = {
                "input": input_text,
                "output": content
            }
            
            return conversation
            
        except Exception as e:
            print(f"  Error processing {file_path.name}: {e}")
            return None
    
    def process_directory(self, directory: Path) -> int:
        """
        Process all text files in a directory.
        
        Args:
            directory: Path to directory containing text files
            
        Returns:
            Number of files processed successfully
        """        
        # Find all text files
        text_files = list(directory.glob('*.txt'))
        
        if not text_files:
            print(f"Warning: No .txt files found in {directory}")
            return 0
        
        print(f"Found {len(text_files)} text files to process")
        
        # Process each file
        processed_count = 0
        for file_path in sorted(text_files):
            print(f"Processing: {file_path.name}")
            conversation = self.process_file(file_path)
            
            if conversation:
                self.conversations.append(conversation)
                processed_count += 1
        
        print(f"\nSuccessfully processed {processed_count}/{len(text_files)} files")
        return processed_count
    
    def save_conversations(self, output_path: Path) -> None:
        """
        Save conversations to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.conversations, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved {len(self.conversations)} conversations to {output_path}")
    
    def get_conversations(self) -> List[Dict[str, str]]:
        """
        Get the processed conversations.
        
        Returns:
            List of conversation dictionaries
        """
        return self.conversations


def main():
    """
    Main entry point for the script.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description='Process scraped content into conversational JSON format for mtnsails training.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process files from scrapyer output directory (with randomized prompts)
  python process_scraped_content.py /path/to/scraped/files

  # Specify custom output file
  python process_scraped_content.py /path/to/scraped/files --output my_chat_data.json

  # Use custom prompt template (supports multiple placeholders)
  python process_scraped_content.py /path/to/scraped/files --prompt-template "Explain {topic}"
  python process_scraped_content.py /path/to/scraped/files --prompt-template "What is {topic} from {source}?"
  python process_scraped_content.py /path/to/scraped/files --prompt-template "Tell me about this {category}"

  # Disable prompt randomization (uses first default template)
  python process_scraped_content.py /path/to/scraped/files --no-randomize

Available Placeholders:
  {topic}         - The main topic/title extracted from content or filename
  {source}        - Source identifier extracted from filename
  {content_type}  - Type of content (e.g., "article", "guide", "brief explanation")
  {category}      - Inferred category (e.g., "tutorial", "documentation", "concept")
  {filename}      - Original filename
        """
    )
    
    parser.add_argument(
        'input_directory',
        type=str,
        help='Directory containing plain text files from scrapyer'
    )
    
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='chat_data.json',
        help='Output JSON file path (default: chat_data.json)'
    )
    
    parser.add_argument(
        '--prompt-template',
        '-p',
        type=str,
        default=None,
        help='Template for input prompts. Supports placeholders: {topic}, {source}, {content_type}, {category}, {filename}. If not provided, randomly selects from built-in templates.'
    )
    
    parser.add_argument(
        '--no-randomize',
        action='store_true',
        help='Disable prompt randomization. Uses a single consistent template instead of varying prompts.'
    )
    
    args = parser.parse_args()
    
    # Convert paths
    input_dir = Path(args.input_directory)
    output_file = Path(args.output)
    
    # Validate input directory first
    if not input_dir.exists():
        print(f"❌ Error: Input directory not found: {input_dir}")
        print(f"\nPlease provide a valid directory containing text files from scrapyer.")
        return 1
    
    if not input_dir.is_dir():
        print(f"❌ Error: Not a directory: {input_dir}")
        print(f"\nPlease provide a directory, not a file.")
        return 1
    
    print("=" * 60)
    print("MTN Sails - Scraped Content Processor")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    
    if args.prompt_template:
        print(f"Prompt template: {args.prompt_template}")
    else:
        if args.no_randomize:
            print(f"Prompt template: Using default template (non-randomized)")
        else:
            print(f"Prompt templates: Randomized from {len(ScrapedContentProcessor.DEFAULT_PROMPT_TEMPLATES)} variations")
    
    print("=" * 60)
    print()
    
    try:
        # Create processor
        processor = ScrapedContentProcessor(
            prompt_template=args.prompt_template,
            randomize_prompts=not args.no_randomize
        )
        
        # Process directory
        processed_count = processor.process_directory(input_dir)
        
        if processed_count == 0:
            print("\nNo files were processed successfully.")
            return 1
        
        # Save conversations
        processor.save_conversations(output_file)
        
        print("\n" + "=" * 60)
        print("✅ Processing complete!")
        print("=" * 60)
        print(f"\nNext steps:")
        print(f"1. Validate data quality: python main.py validate --data-file {output_file}")
        print(f"2. Train your model: python main.py train --data-file {output_file}")
        print()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
