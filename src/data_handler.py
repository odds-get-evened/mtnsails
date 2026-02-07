"""
Data Handler for processing conversation batches.
"""

import json
from typing import List, Dict, Optional
from pathlib import Path


class ConversationDataHandler:
    """Handles loading and preprocessing conversation data for training."""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the data handler.
        
        Args:
            data_path: Path to conversation data file (JSON format)
        """
        self.data_path = data_path
        self.conversations = []
    
    def load_from_json(self, file_path: str) -> None:
        """
        Load conversations from a JSON file.
        
        Args:
            file_path: Path to JSON file containing conversations
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.conversations = data if isinstance(data, list) else [data]
    
    def add_conversation(self, conversation: Dict[str, str]) -> None:
        """
        Add a single conversation to the dataset.
        
        Args:
            conversation: Dictionary with 'input' and 'output' keys
        """
        if 'input' in conversation and 'output' in conversation:
            self.conversations.append(conversation)
        else:
            raise ValueError("Conversation must have 'input' and 'output' keys")
    
    def add_conversations(self, conversations: List[Dict[str, str]]) -> None:
        """
        Add multiple conversations to the dataset.
        
        Args:
            conversations: List of conversation dictionaries
        """
        for conv in conversations:
            self.add_conversation(conv)
    
    def format_for_training(self) -> List[str]:
        """
        Format conversations for training.
        
        Returns:
            List of formatted training texts
        """
        formatted_texts = []
        for conv in self.conversations:
            text = f"User: {conv['input']}\nAssistant: {conv['output']}"
            formatted_texts.append(text)
        return formatted_texts
    
    def save_to_json(self, file_path: str) -> None:
        """
        Save conversations to a JSON file.
        
        Args:
            file_path: Path to save the JSON file
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.conversations, f, indent=2, ensure_ascii=False)
    
    def get_batch(self, batch_size: int, start_idx: int = 0) -> List[Dict[str, str]]:
        """
        Get a batch of conversations.
        
        Args:
            batch_size: Number of conversations per batch
            start_idx: Starting index for the batch
            
        Returns:
            List of conversation dictionaries
        """
        return self.conversations[start_idx:start_idx + batch_size]
    
    def __len__(self) -> int:
        """Return the number of conversations."""
        return len(self.conversations)
