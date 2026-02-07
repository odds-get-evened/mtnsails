"""
Chat Interface for interacting with ONNX models.
"""

import json
import threading
import queue
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer


# Constants
LOGGING_THREAD_SHUTDOWN_TIMEOUT = 5.0  # seconds


class ChatInterface:
    """Interface for chatting with an ONNX-converted LLM model."""
    
    def __init__(
        self,
        onnx_model_path: str,
        device: str = "cpu",
        max_length: int = 256,
        log_conversations: bool = False,
        log_file: Optional[str] = None,
        auto_flush_interval: int = 10
    ):
        """
        Initialize the chat interface.
        
        Args:
            onnx_model_path: Path to the ONNX model
            device: Device to use for inference ('cpu' or 'cuda')
            max_length: Maximum generation length
            log_conversations: Enable/disable conversation logging (default: False)
            log_file: Path to save conversation logs (e.g., "chat_history.json")
            auto_flush_interval: Number of conversations before auto-saving to disk
        """
        self.model_path = Path(onnx_model_path)
        self.device = device
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        
        # Logging configuration
        self.log_conversations = log_conversations
        self.log_file = Path(log_file) if log_file else None
        self.conversation_log: List[Dict[str, str]] = []
        self.log_queue: queue.Queue = queue.Queue()
        self.log_thread: Optional[threading.Thread] = None
        self.stop_logging = threading.Event()
        self.auto_flush_interval = auto_flush_interval
        self._conversation_count = 0
        
        self._load_model()
        
        # Start logging thread if enabled
        if self.log_conversations and self.log_file:
            self._start_logging_thread()
    
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
    
    def _start_logging_thread(self) -> None:
        """Start background thread for async file writes."""
        self.log_thread = threading.Thread(
            target=self._async_save_conversations,
            daemon=True
        )
        self.log_thread.start()
    
    def _log_conversation(self, prompt: str, response: str) -> None:
        """
        Log a single conversation.
        
        Args:
            prompt: User input
            response: Assistant response
        """
        conversation = {
            "input": prompt,
            "output": response,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to in-memory buffer
        self.conversation_log.append(conversation)
        self._conversation_count += 1
        
        # Queue for async file write
        self.log_queue.put(conversation)
        
        # Auto-flush every N conversations
        if self._conversation_count % self.auto_flush_interval == 0:
            self.log_queue.put("FLUSH")
    
    def _async_save_conversations(self) -> None:
        """Background worker that saves to disk."""
        buffer = []
        
        while not self.stop_logging.is_set():
            try:
                # Get item from queue with timeout
                item = self.log_queue.get(timeout=1.0)
                
                if item == "FLUSH":
                    # Save all buffered conversations
                    if buffer:
                        self._write_to_file(buffer)
                        buffer.clear()
                elif item == "STOP":
                    # Final save and exit
                    if buffer:
                        self._write_to_file(buffer)
                    break
                else:
                    # Accumulate conversations
                    buffer.append(item)
                    
            except queue.Empty:
                # Periodic check for stop signal
                continue
            except Exception as e:
                print(f"Logging error: {e}")
    
    def _write_to_file(self, conversations: List[Dict[str, str]]) -> None:
        """
        Write conversations to file.
        
        Args:
            conversations: List of conversation dicts to write
        """
        try:
            # Create parent directories if needed
            if self.log_file:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Load existing data if file exists
                existing_data = []
                if self.log_file.exists():
                    try:
                        with open(self.log_file, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                            if not isinstance(existing_data, list):
                                existing_data = [existing_data]
                    except (json.JSONDecodeError, IOError):
                        # If file is corrupted, start fresh
                        existing_data = []
                
                # Append new conversations
                existing_data.extend(conversations)
                
                # Write to file
                with open(self.log_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error writing to log file: {e}")
    
    def _flush_logs(self) -> None:
        """
        Force flush any pending logs to disk.
        
        Note: This uses a short sleep after queuing the FLUSH command to allow
        the background thread time to process it. This is a simple approach that
        works for most cases, though it's not guaranteed to be immediate.
        """
        if self.log_conversations and self.log_thread and self.log_thread.is_alive():
            self.log_queue.put("FLUSH")
            # Give the thread time to process the flush
            time.sleep(0.5)  # Wait for flush to complete
    
    def flush_logs(self) -> None:
        """
        Public method to force flush any pending logs to disk.
        Call this method to ensure all queued conversations are written to file.
        """
        self._flush_logs()
    
    def get_conversation_log(self) -> List[Dict[str, str]]:
        """
        Return logged conversations.
        
        Returns:
            Copy of conversation_log
        """
        return self.conversation_log.copy()
    
    def clear_conversation_log(self) -> None:
        """Clear the in-memory log."""
        self.conversation_log.clear()
        self._conversation_count = 0
    
    def save_conversation_log(self, file_path: Optional[str] = None) -> None:
        """
        Manually save to file.
        
        Args:
            file_path: Optional path to save to (uses log_file if not provided)
        """
        save_path = Path(file_path) if file_path else self.log_file
        
        if not save_path:
            raise ValueError("No log file path specified")
        
        # Create parent directories if needed
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write conversations
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_log, f, indent=2, ensure_ascii=False)
    
    def __del__(self) -> None:
        """Cleanup on exit."""
        if self.log_conversations and self.log_thread and self.log_thread.is_alive():
            # Signal stop and send final flush
            self.log_queue.put("STOP")
            self.stop_logging.set()
            
            # Wait for thread to finish (with timeout)
            self.log_thread.join(timeout=LOGGING_THREAD_SHUTDOWN_TIMEOUT)

    
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
        
        # Log conversation if enabled
        if self.log_conversations:
            self._log_conversation(prompt, response)
        
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
        if self.log_conversations:
            print(f"[Conversation logging enabled: {self.log_file}]")
        print("Type 'quit' or 'exit' to end the conversation.\n")
        
        try:
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
        finally:
            # Ensure logs are flushed when chat session ends
            self._flush_logs()
    
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
