#!/usr/bin/env python3
"""
Example: Chat with conversation logging enabled for retraining.
"""

from src.chat_interface import ChatInterface
from src.data_handler import ConversationDataHandler
from src.trainer import LLMTrainer


def main():
    """Demonstrate chat with logging and retraining workflow."""
    print("=== Chat with Logging Example ===\n")
    
    # Step 1: Chat with logging enabled
    print("Step 1: Chatting with logging enabled...")
    print("Note: This example requires an existing ONNX model at ./onnx_model\n")
    
    # Check if model exists
    import os
    if not os.path.exists("./onnx_model"):
        print("Error: ONNX model not found at ./onnx_model")
        print("Please run 'python main.py pipeline' first to create a model.\n")
        return
    
    chat = ChatInterface(
        "./onnx_model",
        log_conversations=True,
        log_file="collected_chats.json"
    )
    
    # Simulate some conversations
    test_prompts = [
        "What is machine learning?",
        "How does neural network work?",
        "Explain Python decorators"
    ]
    
    for prompt in test_prompts:
        response = chat.generate_response(prompt)
        print(f"Q: {prompt}")
        print(f"A: {response}\n")
    
    # Manually save the log
    print("Saving conversation log...")
    chat.save_conversation_log()
    print(f"Conversations saved to: collected_chats.json")
    print(f"Total conversations logged: {len(chat.get_conversation_log())}\n")
    
    # Step 2: Use logged conversations for retraining
    print("Step 2: Loading logged conversations for potential retraining...")
    data_handler = ConversationDataHandler()
    data_handler.load_from_json("collected_chats.json")
    
    print(f"Loaded {len(data_handler)} conversations for retraining")
    print("These conversations are now ready to be used for training!\n")
    
    # Show the format
    print("Sample logged conversation:")
    if len(chat.get_conversation_log()) > 0:
        import json
        print(json.dumps(chat.get_conversation_log()[0], indent=2))
    
    print("\n=== Example Complete ===")
    print("\nTo retrain with this data, run:")
    print("python main.py train --data-file collected_chats.json --epochs 3")


if __name__ == "__main__":
    main()
