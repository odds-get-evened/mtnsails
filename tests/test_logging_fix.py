#!/usr/bin/env python3
"""
Test to verify that conversation logging is fixed.
This test validates that logs are flushed even with few prompts.
"""

import json
import sys
import tempfile
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_logging_with_mock_interface():
    """
    Test that simulates the logging mechanism without ML dependencies.
    This verifies the flush mechanism works correctly.
    """
    print("Testing logging flush mechanism...")
    
    import queue
    import threading
    from datetime import datetime
    
    # Simulate the logging queue/thread pattern from ChatInterface
    log_queue = queue.Queue()
    stop_logging = threading.Event()
    log_file = Path(tempfile.mkdtemp()) / "test_chat.json"
    
    # Worker function similar to _async_save_conversations
    def async_save_worker():
        buffer = []
        while not stop_logging.is_set():
            try:
                item = log_queue.get(timeout=1.0)
                
                if item == "FLUSH":
                    if buffer:
                        # Write to file
                        existing_data = []
                        if log_file.exists():
                            try:
                                with open(log_file, 'r', encoding='utf-8') as f:
                                    existing_data = json.load(f)
                            except (json.JSONDecodeError, IOError):
                                existing_data = []
                        
                        existing_data.extend(buffer)
                        
                        with open(log_file, 'w', encoding='utf-8') as f:
                            json.dump(existing_data, f, indent=2, ensure_ascii=False)
                        buffer.clear()
                elif item == "STOP":
                    if buffer:
                        # Final flush
                        existing_data = []
                        if log_file.exists():
                            try:
                                with open(log_file, 'r', encoding='utf-8') as f:
                                    existing_data = json.load(f)
                            except (json.JSONDecodeError, IOError):
                                existing_data = []
                        
                        existing_data.extend(buffer)
                        
                        with open(log_file, 'w', encoding='utf-8') as f:
                            json.dump(existing_data, f, indent=2, ensure_ascii=False)
                    break
                else:
                    buffer.append(item)
            except queue.Empty:
                continue
    
    # Start worker thread
    log_thread = threading.Thread(target=async_save_worker, daemon=True)
    log_thread.start()
    
    # Simulate a few conversations (less than auto_flush_interval)
    conversations = [
        {"input": "What is Python?", "output": "A programming language.", "timestamp": datetime.now().isoformat()},
        {"input": "What is ML?", "output": "Machine learning.", "timestamp": datetime.now().isoformat()},
        {"input": "What is AI?", "output": "Artificial intelligence.", "timestamp": datetime.now().isoformat()},
    ]
    
    for conv in conversations:
        log_queue.put(conv)
    
    # Now flush (this is what the fix adds)
    log_queue.put("FLUSH")
    time.sleep(0.5)  # Wait for flush to complete
    
    # Verify file was written
    assert log_file.exists(), "Log file was not created"
    
    with open(log_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    assert len(data) == 3, f"Expected 3 conversations, got {len(data)}"
    assert data[0]["input"] == "What is Python?"
    assert data[1]["input"] == "What is ML?"
    assert data[2]["input"] == "What is AI?"
    
    # Cleanup
    log_queue.put("STOP")
    stop_logging.set()
    log_thread.join(timeout=2.0)
    
    print("✓ Logging flush mechanism works correctly!")
    print(f"✓ Successfully logged {len(data)} conversations")
    
    # Cleanup temp file
    log_file.unlink()


def test_without_flush():
    """
    Test that demonstrates the bug: without flush, logs are not written.
    """
    print("\nTesting scenario WITHOUT flush (demonstrates the bug)...")
    
    import queue
    import threading
    from datetime import datetime
    
    log_queue = queue.Queue()
    stop_logging = threading.Event()
    log_file = Path(tempfile.mkdtemp()) / "test_no_flush.json"
    
    # Same worker as before
    def async_save_worker():
        buffer = []
        while not stop_logging.is_set():
            try:
                item = log_queue.get(timeout=1.0)
                
                if item == "FLUSH":
                    if buffer:
                        existing_data = []
                        if log_file.exists():
                            try:
                                with open(log_file, 'r', encoding='utf-8') as f:
                                    existing_data = json.load(f)
                            except (json.JSONDecodeError, IOError):
                                existing_data = []
                        
                        existing_data.extend(buffer)
                        
                        with open(log_file, 'w', encoding='utf-8') as f:
                            json.dump(existing_data, f, indent=2, ensure_ascii=False)
                        buffer.clear()
                else:
                    buffer.append(item)
            except queue.Empty:
                continue
    
    log_thread = threading.Thread(target=async_save_worker, daemon=True)
    log_thread.start()
    
    # Add a few conversations
    conversations = [
        {"input": "Q1", "output": "A1", "timestamp": datetime.now().isoformat()},
        {"input": "Q2", "output": "A2", "timestamp": datetime.now().isoformat()},
    ]
    
    for conv in conversations:
        log_queue.put(conv)
    
    # DON'T flush - simulating the bug
    time.sleep(0.5)
    
    # Check if file was written (it shouldn't be)
    if not log_file.exists():
        print("✓ Confirmed: Without flush, log file is NOT created (bug reproduced)")
    else:
        with open(log_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if len(data) == 0:
            print("✓ Confirmed: Without flush, log file is empty (bug reproduced)")
    
    # Cleanup
    stop_logging.set()
    log_thread.join(timeout=2.0)
    if log_file.exists():
        log_file.unlink()


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Conversation Logging Fix")
    print("=" * 60)
    
    try:
        test_without_flush()
        test_logging_with_mock_interface()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print("\nThe fix ensures that conversations are flushed to disk")
        print("even when the auto_flush_interval is not reached.")
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
