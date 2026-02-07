#!/usr/bin/env python3
"""
Main application for LLM training and ONNX conversion.
"""

import argparse
import sys
import warnings
from pathlib import Path

# Suppress common warnings from transformers and torch libraries
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')
warnings.filterwarnings('ignore', category=FutureWarning, module='optimum')
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

# Suppress specific torch warnings
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('optimum').setLevel(logging.ERROR)


def train_model(args):
    """Train a model on conversation data."""
    from src.data_handler import ConversationDataHandler
    from src.trainer import LLMTrainer
    
    print("=== Training Model ===")
    
    # Load data
    data_handler = ConversationDataHandler()
    if args.data_file:
        data_handler.load_from_json(args.data_file)
    else:
        print("No data file provided. Creating example data...")
        # Add example conversations
        example_conversations = [
            {"input": "What is Python?", "output": "Python is a high-level programming language."},
            {"input": "How do I install Python?", "output": "You can download Python from python.org."},
            {"input": "What is machine learning?", "output": "Machine learning is a subset of AI."},
        ]
        data_handler.add_conversations(example_conversations)
    
    print(f"Loaded {len(data_handler)} conversations")
    
    # Format data for training
    train_texts = data_handler.format_for_training()
    
    # Initialize trainer
    trainer = LLMTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Train
    trainer.train(
        train_texts=train_texts,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Save model
    model_path = trainer.save_model()
    print(f"Model saved to: {model_path}")
    
    return model_path


def convert_to_onnx(args):
    """Convert a trained model to ONNX format."""
    from src.onnx_converter import ONNXConverter
    
    print("=== Converting to ONNX ===")
    
    converter = ONNXConverter(args.model_path)
    onnx_path = converter.convert_to_onnx(
        output_path=args.onnx_output,
        opset_version=args.opset_version
    )
    
    # Verify conversion
    if args.verify:
        converter.verify_onnx_model(onnx_path)
    
    print(f"ONNX model saved to: {onnx_path}")
    return onnx_path


def chat(args):
    """Start a chat session with an ONNX model."""
    from src.chat_interface import ChatInterface
    
    print("=== Chat Interface ===")
    
    chat_interface = ChatInterface(
        onnx_model_path=args.model_path,
        device=args.device,
        max_length=args.max_length,
        log_conversations=getattr(args, 'log_conversations', False),
        log_file=getattr(args, 'log_file', None) if getattr(args, 'log_conversations', False) else None
    )
    
    try:
        if args.prompt:
            # Single prompt mode
            response = chat_interface.generate_response(
                args.prompt,
                max_new_tokens=args.max_tokens
            )
            print(f"Assistant: {response}")
        else:
            # Interactive mode
            chat_interface.chat(interactive=True)
    finally:
        # Ensure logs are flushed on exit
        if chat_interface.log_conversations:
            chat_interface._flush_logs()


def full_pipeline(args):
    """Run the full pipeline: train, convert, and test."""
    print("=== Full Pipeline ===")
    
    # Step 1: Train
    print("\nStep 1: Training model...")
    model_path = train_model(args)
    
    # Step 2: Convert to ONNX
    print("\nStep 2: Converting to ONNX...")
    args.model_path = model_path
    onnx_path = convert_to_onnx(args)
    
    # Step 3: Test chat
    print("\nStep 3: Testing chat interface...")
    args.model_path = onnx_path
    args.prompt = "Hello, how are you?"
    chat(args)
    
    print("\n=== Pipeline Complete ===")
    print(f"Trained model: {model_path}")
    print(f"ONNX model: {onnx_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Training and ONNX Conversion System"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--data-file', type=str, help='Path to conversation data JSON')
    train_parser.add_argument('--model-name', type=str, default='distilgpt2', help='Base model name')
    train_parser.add_argument('--output-dir', type=str, default='./trained_model', help='Output directory')
    train_parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    train_parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert model to ONNX')
    convert_parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    convert_parser.add_argument('--onnx-output', type=str, default='./onnx_model', help='ONNX output path')
    convert_parser.add_argument('--opset-version', type=int, default=14, help='ONNX opset version')
    convert_parser.add_argument('--verify', action='store_true', help='Verify ONNX model')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Chat with ONNX model')
    chat_parser.add_argument('--model-path', type=str, required=True, help='Path to ONNX model')
    chat_parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    chat_parser.add_argument('--max-length', type=int, default=256, help='Max input length')
    chat_parser.add_argument('--max-tokens', type=int, default=50, help='Max tokens to generate')
    chat_parser.add_argument('--prompt', type=str, help='Single prompt (non-interactive)')
    chat_parser.add_argument('--log-conversations', action='store_true',
                            help='Enable conversation logging for retraining')
    chat_parser.add_argument('--log-file', type=str, default='./chat_history.json',
                            help='Path to save conversation logs')
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full pipeline')
    pipeline_parser.add_argument('--data-file', type=str, help='Path to conversation data JSON')
    pipeline_parser.add_argument('--model-name', type=str, default='distilgpt2', help='Base model name')
    pipeline_parser.add_argument('--output-dir', type=str, default='./trained_model', help='Output directory')
    pipeline_parser.add_argument('--onnx-output', type=str, default='./onnx_model', help='ONNX output path')
    pipeline_parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    pipeline_parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    pipeline_parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    pipeline_parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate')
    pipeline_parser.add_argument('--opset-version', type=int, default=14, help='ONNX opset version')
    pipeline_parser.add_argument('--verify', action='store_true', help='Verify ONNX model')
    pipeline_parser.add_argument('--max-length', type=int, default=256, help='Max input length')
    pipeline_parser.add_argument('--max-tokens', type=int, default=50, help='Max tokens to generate')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'train':
            train_model(args)
        elif args.command == 'convert':
            convert_to_onnx(args)
        elif args.command == 'chat':
            chat(args)
        elif args.command == 'pipeline':
            full_pipeline(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
