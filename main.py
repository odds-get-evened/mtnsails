#!/usr/bin/env python3
"""
Main application for LLM training and ONNX conversion.
"""

import argparse
import json
import shutil
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


def check_model_exists(model_path: str) -> bool:
    """Check if a trained model exists at the given path."""
    model_path = Path(model_path)
    
    # Check for required files
    has_config = (model_path / "config.json").exists()
    has_safetensors = (model_path / "model.safetensors").exists()
    has_pytorch = (model_path / "pytorch_model.bin").exists()
    
    return has_config and (has_safetensors or has_pytorch)


def validate_data(args):
    """
    Validate training data quality without training.
    
    Args:
        args: Argument namespace with data_file attribute
        
    Returns:
        Does not return - exits with code 0 if quality is good (≥50%),
        exits with code 1 if quality is critical (<50%)
    """
    from src.data_handler import ConversationDataHandler
    
    print("=== Data Quality Validation ===")
    
    # Load data
    data_handler = ConversationDataHandler()
    if not args.data_file:
        print("Error: --data-file is required")
        sys.exit(1)
    
    try:
        data_handler.load_from_json(args.data_file)
    except FileNotFoundError:
        print(f"Error: File '{args.data_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: File '{args.data_file}' is not valid JSON")
        sys.exit(1)
    
    print(f"Loaded {len(data_handler)} conversations from {args.data_file}")
    print()
    
    # Analyze data quality
    quality_report = data_handler.analyze_dataset_quality()
    
    # Note: quality_score is a decimal (0.0 to 1.0), formatted as percentage
    print(f"Total conversations: {quality_report['total_conversations']}")
    print(f"Valid conversations: {quality_report['valid_conversations']}")
    print(f"Invalid conversations: {quality_report['invalid_conversations']}")
    print(f"Quality score: {quality_report['quality_score']:.1%}")
    
    if quality_report['issue_summary']:
        issues_found = sum(quality_report['issue_summary'].values())
        if issues_found > 0:
            print("\nIssues detected:")
            for issue_type, count in quality_report['issue_summary'].items():
                if count > 0:
                    print(f"  - {issue_type.replace('_', ' ').title()}: {count}")
    
    if quality_report['problematic_examples']:
        print("\nExample problematic conversations:")
        for example in quality_report['problematic_examples'][:5]:
            print(f"\n  Conversation #{example['index']}:")
            print(f"    Input: {example['input']}")
            print(f"    Output: {example['output']}")
            print(f"    Issues: {', '.join(example['issues'])}")
    
    print("\nRecommendations:")
    for rec in quality_report['recommendations']:
        print(f"  {rec}")
    
    # Handle filtering if requested
    if args.filter:
        print("\n" + "="*70)
        print("=== Filtering Data ===")
        print("="*70)
        
        # Get valid conversations
        valid_conversations = data_handler.filter_valid_conversations()
        
        # Determine output file
        if args.output:
            output_file = args.output
        else:
            # Use os.path.splitext to handle file extensions properly
            from pathlib import Path
            file_path = Path(args.data_file)
            output_file = str(file_path.parent / f"{file_path.stem}_filtered{file_path.suffix}")
        
        # Save filtered data
        filtered_handler = ConversationDataHandler()
        filtered_handler.add_conversations(valid_conversations)
        filtered_handler.save_to_json(output_file)
        
        print(f"\n✅ Filtered data saved to: {output_file}")
        print(f"   Original: {quality_report['total_conversations']} conversations")
        print(f"   Filtered: {len(valid_conversations)} conversations")
        print(f"   Removed: {quality_report['invalid_conversations']} conversations")
        print("="*70)
    
    # Summary
    print("\n" + "="*70)
    if quality_report['quality_score'] >= 0.7:
        print("✅ DATA QUALITY: GOOD")
        print("="*70)
        print("Your data quality is good. You can proceed with training.")
    elif quality_report['quality_score'] >= 0.5:
        print("⚠️  DATA QUALITY: ACCEPTABLE (with warnings)")
        print("="*70)
        print("Your data has some quality issues. Consider reviewing problematic")
        print("conversations, but training may still produce reasonable results.")
        if not args.filter:
            print("\nTIP: Use --filter to automatically remove bad conversations.")
    else:
        print("❌ DATA QUALITY: CRITICAL")
        print("="*70)
        print("Your data quality is very low. Training on this data will likely")
        print("produce a model that generates poor quality responses.")
        print("\nRECOMMENDATIONS:")
        print("  1. Filter out conversations with empty or nonsensical responses")
        print("  2. Remove repetitive conversations")
        print("  3. Review the problematic examples above")
        print("  4. Use only high-quality, meaningful conversations")
        if not args.filter:
            print("\nTIP: Use --filter to automatically remove bad conversations.")
    print("="*70)
    
    # Exit with appropriate code
    if quality_report['quality_score'] < 0.5:
        sys.exit(1)
    else:
        sys.exit(0)


def train_model(args):
    """Train a model on conversation data."""
    from src.data_handler import ConversationDataHandler
    from src.trainer import LLMTrainer
    
    print("=== Training Model ===")
    
    # Check if we should continue training from an existing model
    model_to_use = args.model_name
    learning_rate_to_use = args.learning_rate
    is_retraining = False
    
    if check_model_exists(args.output_dir):
        print(f"🔄 Found existing trained model at '{args.output_dir}'")
        print("🔄 Continuing training from this checkpoint...")
        model_to_use = args.output_dir
        is_retraining = True
        # Use lower learning rate for fine-tuning to avoid catastrophic forgetting
        learning_rate_to_use = 1e-5
        print(f"📚 Using lower learning rate ({learning_rate_to_use}) for fine-tuning to preserve existing knowledge")
    else:
        print(f"🆕 Training new model from base '{args.model_name}'")
        print(f"📚 Using standard learning rate ({learning_rate_to_use}) for initial training")
    
    print()
    
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
    
    # Analyze data quality
    print("\n=== Data Quality Analysis ===")
    quality_report = data_handler.analyze_dataset_quality()
    
    print(f"Total conversations: {quality_report['total_conversations']}")
    print(f"Valid conversations: {quality_report['valid_conversations']}")
    print(f"Invalid conversations: {quality_report['invalid_conversations']}")
    print(f"Quality score: {quality_report['quality_score']:.1%}")
    
    if quality_report['issue_summary']:
        print("\nIssues detected:")
        for issue_type, count in quality_report['issue_summary'].items():
            if count > 0:
                print(f"  - {issue_type.replace('_', ' ').title()}: {count}")
    
    if quality_report['problematic_examples']:
        print("\nExample problematic conversations:")
        for example in quality_report['problematic_examples'][:3]:
            print(f"\n  Conversation #{example['index']}:")
            print(f"    Input: {example['input']}")
            print(f"    Output: {example['output']}")
            print(f"    Issues: {', '.join(example['issues'])}")
    
    print("\nRecommendations:")
    for rec in quality_report['recommendations']:
        print(f"  {rec}")
    
    # Warn if quality is low
    if quality_report['quality_score'] < 0.5:
        print("\n" + "="*70)
        print("⚠️  CRITICAL WARNING: DATA QUALITY IS VERY LOW")
        print("="*70)
        print("Training on this data will likely result in a model that produces:")
        print("  - Nonsense or gibberish responses")
        print("  - Repetitive text")
        print("  - Echoes of user input")
        print("\nThis is a 'garbage in, garbage out' situation.")
        print("\nRECOMMENDATIONS:")
        print("  1. Filter out low-quality conversations")
        print("  2. Use only conversations with meaningful, coherent responses")
        print("  3. Avoid training on chat logs with nonsense outputs")
        print("  4. Start with high-quality example conversations")
        print("="*70)
        
        # Ask for confirmation
        if not args.force:
            response = input("\nDo you want to continue anyway? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("Training cancelled. Please improve your data quality first.")
                sys.exit(0)
    elif quality_report['quality_score'] < 0.7:
        print("\n⚠️  WARNING: Consider filtering out problematic conversations")
        print("Your model's performance may be affected by low-quality data.\n")
    
    # Format data for training
    train_texts = data_handler.format_for_training()
    
    # Initialize trainer
    trainer = LLMTrainer(
        model_name=model_to_use,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Train
    trainer.train(
        train_texts=train_texts,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=learning_rate_to_use
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
            chat_interface.flush_logs()


def full_pipeline(args):
    """Run the full pipeline: train, convert, and test."""
    print("=== Full Pipeline ===")
    print()
    
    # Check if ONNX model exists and if we should retrain from existing model
    onnx_path = Path(args.onnx_output)
    trained_model_path = Path(args.output_dir)
    
    # Check if we have an existing ONNX model and corresponding trained model
    if onnx_path.exists() and check_model_exists(str(trained_model_path)):
        print(f"🔄 Found existing ONNX model at {args.onnx_output}")
        print(f"🔄 Found existing trained model at {args.output_dir}")
        print("🔄 Continuing training from previous checkpoint...")
        print()
    elif onnx_path.exists():
        # ONNX exists but source model was deleted
        print(f"🔄 Found existing ONNX model at {args.onnx_output}")
        print(f"⚠️  Source trained model not found at {args.output_dir}")
        print(f"🆕 Training from base model '{args.model_name}'")
        print()
    elif check_model_exists(str(trained_model_path)):
        # Trained model exists but no ONNX yet
        print(f"🔄 Found existing trained model at {args.output_dir}")
        print("🔄 Continuing training from this checkpoint...")
        print()
    else:
        print(f"🆕 No existing model found. Training from base model '{args.model_name}'")
        print()
    
    # Step 1: Train
    print("Step 1: Training model...")
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


def baseline_model(args):
    """
    Create baseline ONNX model from base DistilGPT-2 without training.
    This gives you a fresh, untouched model to test against.
    
    Args:
        args: Argument namespace with model_name, baseline_output, and test attributes
    """
    from optimum.onnxruntime import ORTModelForCausalLM
    from transformers import AutoTokenizer
    
    print("=== Creating Baseline ONNX Model ===")
    print()
    print(f"📦 Base model: {args.model_name}")
    print(f"📂 Output directory: {args.baseline_output}")
    print()
    print("Note: No training will be performed - this is a clean export")
    print()
    
    output_path = Path(args.baseline_output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load and convert the base model directly to ONNX
        print(f"Loading {args.model_name} from Hugging Face...")
        model = ORTModelForCausalLM.from_pretrained(
            args.model_name,
            export=True,
            use_io_binding=True
        )
        
        print(f"Saving ONNX model to {args.baseline_output}...")
        model.save_pretrained(str(output_path))
        
        # Also save the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.save_pretrained(str(output_path))
        
        print()
        print(f"✅ Baseline ONNX model saved to: {args.baseline_output}")
        print(f"This is the untrained {args.model_name} in ONNX format")
        
        # Optional: Test the model
        if args.test:
            print()
            print("=== Testing Baseline Model ===")
            test_prompt = "Hello, I am a"
            print(f"Test prompt: '{test_prompt}'")
            
            inputs = tokenizer(test_prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_length=20)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"Generated text: {result}")
            print()
            print("✅ Test successful!")
        
        print()
        print("Next steps:")
        print(f"  - Test: python main.py chat --model-path {args.baseline_output} --prompt \"Your prompt here\"")
        print(f"  - Compare with trained model using same prompts")
        
    except Exception as e:
        print(f"❌ Error creating baseline model: {e}")
        sys.exit(1)


def reset_model(args):
    """
    Reset model to original pretrained state by removing fine-tuned models.
    
    Args:
        args: Argument namespace with output_dir and onnx_output attributes
    """
    print("=== Reset Model ===")
    print()
    
    # Paths to delete
    trained_model_path = Path(args.output_dir)
    onnx_model_path = Path(args.onnx_output)
    
    # Show what will be deleted
    print("This will delete the following directories:")
    print(f"  - {args.output_dir} (fine-tuned model)")
    print(f"  - {args.onnx_output} (ONNX converted model)")
    print()
    
    # Confirmation prompt unless --force is provided
    if not args.force:
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Reset cancelled.")
            sys.exit(0)
        print()
    
    # Helper function to delete a directory
    def delete_directory(path, name):
        """Delete a directory if it exists."""
        if path.exists():
            try:
                print(f"🗑️  Deleting {path}...")
                shutil.rmtree(path)
                print(f"✅ Deleted {name}")
            except PermissionError:
                print(f"❌ Error deleting {path}: Permission denied")
                print(f"   Check that you have write permissions and the directory is not in use")
                sys.exit(1)
            except Exception as e:
                print(f"❌ Error deleting {path}: {e}")
                print(f"   Check that the directory is not in use and you have the necessary permissions")
                sys.exit(1)
        else:
            print(f"ℹ️  {path} not found (already clean)")
    
    # Delete trained model directory
    delete_directory(trained_model_path, "fine-tuned model")
    print()
    
    # Delete ONNX model directory
    delete_directory(onnx_model_path, "ONNX model")
    
    print()
    print("✅ Reset complete! Model directories cleared and ready for fresh training.")
    print()
    print("Next steps:")
    print("  - Run: python main.py train --data-file your_data.json")
    print("  - Or: python main.py pipeline --data-file your_data.json")


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
    train_parser.add_argument('--force', action='store_true', help='Skip data quality warnings')
    
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
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate training data quality')
    validate_parser.add_argument('--data-file', type=str, required=True, 
                                help='Path to conversation data JSON to validate')
    validate_parser.add_argument('--filter', action='store_true',
                                help='Filter out bad data and save only valid conversations')
    validate_parser.add_argument('--output', type=str,
                                help='Output file for filtered data (default: <input>_filtered.json)')
    
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
    
    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Reset model to original pretrained state')
    reset_parser.add_argument('--output-dir', type=str, default='./trained_model', 
                             help='Path to trained model directory')
    reset_parser.add_argument('--onnx-output', type=str, default='./onnx_model', 
                             help='Path to ONNX model directory')
    reset_parser.add_argument('--force', action='store_true', 
                             help='Skip confirmation prompt')
    
    # Baseline command
    baseline_parser = subparsers.add_parser('baseline', help='Export base model to ONNX without training')
    baseline_parser.add_argument('--model-name', type=str, default='distilgpt2',
                                help='Base model to export (default: distilgpt2)')
    baseline_parser.add_argument('--baseline-output', type=str, default='./baseline_onnx',
                                help='Output directory for baseline ONNX model (default: ./baseline_onnx)')
    baseline_parser.add_argument('--test', action='store_true',
                                help='Test the model after export with a sample prompt')
    
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
        elif args.command == 'validate':
            validate_data(args)
        elif args.command == 'pipeline':
            full_pipeline(args)
        elif args.command == 'reset':
            reset_model(args)
        elif args.command == 'baseline':
            baseline_model(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
