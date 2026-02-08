#!/usr/bin/env python3
"""
Validation script to verify the code structure and basic functionality.
This script doesn't require heavy ML dependencies.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import directly to avoid loading ML dependencies
from src.data_handler import ConversationDataHandler


def test_data_handler():
    """Test ConversationDataHandler without ML dependencies."""
    print("\n=== Testing ConversationDataHandler ===")
    
    # Test 1: Initialization
    print("✓ Test 1: Initialization")
    handler = ConversationDataHandler()
    assert len(handler) == 0, "Handler should be empty initially"
    
    # Test 2: Add single conversation
    print("✓ Test 2: Add single conversation")
    handler.add_conversation({
        "input": "What is Python?",
        "output": "Python is a programming language."
    })
    assert len(handler) == 1, "Handler should have 1 conversation"
    
    # Test 3: Add multiple conversations
    print("✓ Test 3: Add multiple conversations")
    handler.add_conversations([
        {"input": "Q1", "output": "A1"},
        {"input": "Q2", "output": "A2"},
    ])
    assert len(handler) == 3, "Handler should have 3 conversations"
    
    # Test 4: Format for training
    print("✓ Test 4: Format for training")
    formatted = handler.format_for_training()
    assert len(formatted) == 3, "Should have 3 formatted texts"
    assert "User:" in formatted[0], "Formatted text should contain 'User:'"
    assert "Assistant:" in formatted[0], "Formatted text should contain 'Assistant:'"
    
    # Test 5: Get batch
    print("✓ Test 5: Get batch")
    batch = handler.get_batch(batch_size=2, start_idx=1)
    assert len(batch) == 2, "Batch should have 2 items"
    
    # Test 6: Save and load JSON
    print("✓ Test 6: Save and load JSON")
    test_file = "/tmp/test_conversations.json"
    handler.save_to_json(test_file)
    
    new_handler = ConversationDataHandler()
    new_handler.load_from_json(test_file)
    assert len(new_handler) == 3, "Loaded handler should have 3 conversations"
    
    # Clean up
    Path(test_file).unlink(missing_ok=True)
    
    print("✓ All ConversationDataHandler tests passed!")
    return True


def test_structure():
    """Test that all files are present."""
    print("\n=== Testing Project Structure ===")
    
    required_files = [
        "README.md",
        "requirements.txt",
        "setup.py",
        "main.py",
        "examples/example.py",
        "examples/example_conversations.json",
        ".gitignore",
        "src/__init__.py",
        "src/data_handler.py",
        "src/trainer.py",
        "src/onnx_converter.py",
        "src/chat_interface.py",
        "tests/test_data_handler.py",
    ]
    
    base_path = Path(__file__).parent
    for file in required_files:
        file_path = base_path / file
        assert file_path.exists(), f"Missing required file: {file}"
        print(f"✓ Found: {file}")
    
    print("✓ All required files present!")
    return True


def test_example_data():
    """Test that example data is valid."""
    print("\n=== Testing Example Data ===")
    
    data_file = Path(__file__).parent / "examples" / "example_conversations.json"
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    assert isinstance(data, list), "Data should be a list"
    assert len(data) > 0, "Data should not be empty"
    
    for idx, conv in enumerate(data):
        assert "input" in conv, f"Conversation {idx} missing 'input'"
        assert "output" in conv, f"Conversation {idx} missing 'output'"
        assert isinstance(conv["input"], str), f"Conversation {idx} input should be string"
        assert isinstance(conv["output"], str), f"Conversation {idx} output should be string"
    
    print(f"✓ Validated {len(data)} example conversations")
    return True


def validate_imports():
    """Validate that core modules can be imported."""
    print("\n=== Validating Module Imports ===")
    
    # Test data_handler (no heavy dependencies)
    print("✓ Imported: src.data_handler")
    
    # Note: We can't import trainer, onnx_converter, and chat_interface
    # without ML dependencies, but we've validated they exist
    print("⚠ Skipping ML module imports (require dependencies)")
    
    return True


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("MTN Sails - System Validation")
    print("=" * 60)
    
    try:
        # Run tests
        test_structure()
        test_example_data()
        validate_imports()
        test_data_handler()
        
        print("\n" + "=" * 60)
        print("✅ All validation tests PASSED!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run example: python examples/example.py")
        print("3. Or use CLI: python main.py pipeline")
        print("\n")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ Validation FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
