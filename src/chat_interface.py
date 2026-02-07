"""
Chat Interface for interacting with ONNX models.
"""

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer
from pathlib import Path
from typing import Optional, List


class ChatInterface:
    """Interface for chatting with an ONNX-converted LLM model."""
    
    def __init__(
        self,
        onnx_model_path: str,
        device: str = "cpu",
        max_length: int = 256
    ):
        """
        Initialize the chat interface.
        
        Args:
            onnx_model_path: Path to the ONNX model
            device: Device to use for inference ('cpu' or 'cuda')
            max_length: Maximum generation length
        """
        self.model_path = Path(onnx_model_path)
        self.device = device
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the ONNX model and tokenizer."""
        if not self.model_path.exists():
            raise ValueError(f"Model path does not exist: {self.model_path}")
        
        print(f"Loading ONNX model from {self.model_path}")
        
        self.model = ORTModelForCausalLM.from_pretrained(str(self.model_path))
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Model loaded successfully!")
    
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: Input prompt/question
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            
        Returns:
            Generated response text
        """
        # Format the prompt
        formatted_prompt = f"User: {prompt}\nAssistant:"
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        )
        
        # Generate
        with self.tokenizer.as_target_tokenizer():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "Assistant:" in full_text:
            response = full_text.split("Assistant:")[-1].strip()
        else:
            response = full_text[len(formatted_prompt):].strip()
        
        return response
    
    def chat(self, interactive: bool = True) -> None:
        """
        Start an interactive chat session.
        
        Args:
            interactive: Whether to run in interactive mode
        """
        if not interactive:
            return
        
        print("\n=== Chat Interface ===")
        print("Type 'quit' or 'exit' to end the conversation.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                response = self.generate_response(user_input)
                print(f"Assistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 50,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum number of tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated responses
        """
        responses = []
        for prompt in prompts:
            response = self.generate_response(
                prompt,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
            responses.append(response)
        
        return responses
