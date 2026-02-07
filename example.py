"""
Example script demonstrating the full workflow.
"""


def main():
    """Run a complete example workflow."""
    from src.data_handler import ConversationDataHandler
    from src.trainer import LLMTrainer
    from src.onnx_converter import ONNXConverter
    from src.chat_interface import ChatInterface
    
    print("=== MTN Sails - LLM Training and ONNX Conversion Example ===\n")
    
    # Step 1: Prepare Data
    print("Step 1: Preparing conversation data...")
    data_handler = ConversationDataHandler()
    
    # Add example conversations
    example_conversations = [
        {"input": "What is machine learning?", 
         "output": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."},
        {"input": "What is Python?", 
         "output": "Python is a high-level, interpreted programming language known for its simplicity and versatility."},
        {"input": "How do neural networks work?", 
         "output": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process and transmit information."},
        {"input": "What is deep learning?", 
         "output": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to progressively extract higher-level features from raw input."},
        {"input": "What is natural language processing?", 
         "output": "Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language."},
    ]
    
    data_handler.add_conversations(example_conversations)
    print(f"Added {len(data_handler)} conversations\n")
    
    # Step 2: Train Model
    print("Step 2: Training the model...")
    print("Note: This uses a small model (distilgpt2) suitable for CPU training\n")
    
    trainer = LLMTrainer(
        model_name="distilgpt2",
        output_dir="./example_trained_model",
        device="cpu"
    )
    
    train_texts = data_handler.format_for_training()
    trainer.train(
        train_texts=train_texts,
        num_epochs=2,  # Short training for example
        batch_size=2,
        learning_rate=5e-5
    )
    
    model_path = trainer.save_model()
    print(f"Model saved to: {model_path}\n")
    
    # Step 3: Convert to ONNX
    print("Step 3: Converting model to ONNX format...")
    converter = ONNXConverter(model_path)
    onnx_path = converter.convert_to_onnx(
        output_path="./example_onnx_model",
        opset_version=14
    )
    print(f"ONNX model saved to: {onnx_path}\n")
    
    # Verify the conversion
    print("Verifying ONNX model...")
    if converter.verify_onnx_model(onnx_path, "What is AI?"):
        print("ONNX model verified successfully!\n")
    
    # Step 4: Use the ONNX model for chat
    print("Step 4: Testing chat with ONNX model...")
    chat_interface = ChatInterface(
        onnx_model_path=onnx_path,
        device="cpu"
    )
    
    # Test with sample prompts
    test_prompts = [
        "What is machine learning?",
        "Tell me about Python",
    ]
    
    for prompt in test_prompts:
        response = chat_interface.generate_response(
            prompt,
            max_new_tokens=30,
            temperature=0.7
        )
        print(f"User: {prompt}")
        print(f"Assistant: {response}\n")
    
    print("=== Example Complete ===")
    print(f"\nYou can now use the ONNX model at: {onnx_path}")
    print("Run 'python main.py chat --model-path ./example_onnx_model' for interactive chat")


if __name__ == "__main__":
    main()
