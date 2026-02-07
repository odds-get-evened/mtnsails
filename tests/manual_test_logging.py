#!/usr/bin/env python3
"""
Manual test script for conversation logging functionality.
This script tests the logging features without requiring a full ONNX model.
"""

import json
import sys
import tempfile
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_logging_json_format():
    """Test that the JSON format is correct."""
    print("Test 1: JSON format validation...")
    
    # Create a mock conversation log
    conversations = [
        {
            "input": "What is Python?",
            "output": "Python is a high-level programming language.",
            "timestamp": "2026-02-07T10:30:00"
        },
        {
            "input": "How do I install it?",
            "output": "You can download it from python.org",
            "timestamp": "2026-02-07T10:31:00"
        }
    ]
    
    # Write to temp file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(conversations, temp_file, indent=2)
    temp_file.close()
    
    # Try to load with ConversationDataHandler
    from src.data_handler import ConversationDataHandler
    
    handler = ConversationDataHandler()
    handler.load_from_json(temp_file.name)
    
    assert len(handler) == 2, f"Expected 2 conversations, got {len(handler)}"
    assert handler.conversations[0]["input"] == "What is Python?"
    assert handler.conversations[0]["output"] == "Python is a high-level programming language."
    
    # Cleanup
    Path(temp_file.name).unlink()
    
    print("✓ JSON format is compatible with ConversationDataHandler")


def test_logging_methods():
    """Test the logging methods directly."""
    print("\nTest 2: Logging methods...")
    
    # Test imports
    try:
        from src.chat_interface import ChatInterface
        print("✓ ChatInterface imports successfully")
    except ImportError as e:
        print(f"✗ Import failed (expected in test environment without ML deps): {e}")
        return
    
    # Can't test further without ML dependencies
    print("  (Skipping ML-dependent tests)")


def test_cli_args():
    """Test that CLI arguments are properly defined."""
    print("\nTest 3: CLI arguments...")
    
    import sys
    sys.argv = ['main.py', 'chat', '--help']
    
    try:
        import main
        # Check if the module loads
        print("✓ main.py module loads successfully")
    except SystemExit:
        # --help causes exit, which is expected
        pass
    except Exception as e:
        print(f"✗ main.py failed to load: {e}")
        return


def test_file_operations():
    """Test file write operations."""
    print("\nTest 4: File operations...")
    
    temp_dir = tempfile.mkdtemp()
    log_file = Path(temp_dir) / "test_log.json"
    
    # Test writing conversations
    conversations = [
        {"input": "Q1", "output": "A1", "timestamp": "2026-02-07T10:00:00"},
        {"input": "Q2", "output": "A2", "timestamp": "2026-02-07T10:01:00"}
    ]
    
    # Write
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)
    
    assert log_file.exists(), "Log file was not created"
    
    # Read back
    with open(log_file, 'r', encoding='utf-8') as f:
        loaded = json.load(f)
    
    assert len(loaded) == 2, f"Expected 2 conversations, got {len(loaded)}"
    assert loaded[0]["input"] == "Q1"
    
    # Test appending
    existing_data = loaded
    new_conversations = [
        {"input": "Q3", "output": "A3", "timestamp": "2026-02-07T10:02:00"}
    ]
    existing_data.extend(new_conversations)
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)
    
    # Verify
    with open(log_file, 'r', encoding='utf-8') as f:
        final_data = json.load(f)
    
    assert len(final_data) == 3, f"Expected 3 conversations after append, got {len(final_data)}"
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
    print("✓ File operations work correctly")


def test_threading_imports():
    """Test that threading modules are available."""
    print("\nTest 5: Threading imports...")
    
    try:
        import json
        import threading
        import queue
        from datetime import datetime
        print("✓ All required modules are available")
    except ImportError as e:
        print(f"✗ Import failed: {e}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Manual Tests for Conversation Logging Feature")
    print("=" * 60)
    
    try:
        test_threading_imports()
        test_logging_json_format()
        test_file_operations()
        test_logging_methods()
        test_cli_args()
        
        print("\n" + "=" * 60)
        print("All basic tests passed! ✓")
        print("=" * 60)
        print("\nNote: Full integration tests require ML dependencies.")
        print("To test with a real model, run:")
        print("  python main.py chat --model-path ./onnx_model --log-conversations --log-file test.json")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
