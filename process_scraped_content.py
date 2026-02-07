#!/usr/bin/env python3
"""
Feature script to process scraped content from scrapyer into conversational JSON format.

This script reads plain text files downloaded by scrapyer and converts them into
the conversational JSON format (input/output pairs) required by mtnsails for training.

Usage:
    python process_scraped_content.py <input_directory> [--output chat_data.json] [--prompt-template TEMPLATE]

Example:
    python process_scraped_content.py /path/to/scraped/files --output chat_data.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
import re


class ScrapedContentProcessor:
    """Processes scraped content into conversational JSON format."""
    
    def __init__(self, prompt_template: Optional[str] = None):
        """
        Initialize the processor.
        
        Args:
            prompt_template: Template for generating input prompts.
                           Default: "Tell me about {topic}"
        """
        self.prompt_template = prompt_template or "Tell me about {topic}"
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
            
            # Generate input prompt
            input_text = self.prompt_template.format(topic=topic)
            
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
  # Process files from scrapyer output directory
  python process_scraped_content.py /path/to/scraped/files

  # Specify custom output file
  python process_scraped_content.py /path/to/scraped/files --output my_chat_data.json

  # Use custom prompt template
  python process_scraped_content.py /path/to/scraped/files --prompt-template "Explain {topic}"
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
        default='Tell me about {topic}',
        help='Template for input prompts. Use {topic} as placeholder (default: "Tell me about {topic}")'
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
    print(f"Prompt template: {args.prompt_template}")
    print("=" * 60)
    print()
    
    try:
        # Create processor
        processor = ScrapedContentProcessor(prompt_template=args.prompt_template)
        
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
