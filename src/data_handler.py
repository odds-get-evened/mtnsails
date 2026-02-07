"""
Data Handler for processing conversation batches.
"""

import json
import re
from typing import List, Dict, Optional, Tuple, Any
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
    
    def _calculate_repetition_score(self, text: str) -> float:
        """
        Calculate how repetitive a text is.
        
        Args:
            text: Text to analyze
            
        Returns:
            Repetition score (0.0 = no repetition, 1.0 = highly repetitive)
        """
        if not text or len(text) < 10:
            return 0.0
        
        # Split into words
        words = text.lower().split()
        if len(words) < 3:
            return 0.0
        
        # Count repeated sequences
        repetition_count = 0
        total_sequences = 0
        
        # Check for repeated 2-word sequences
        for i in range(len(words) - 3):
            sequence = ' '.join(words[i:i+2])
            rest = ' '.join(words[i+2:])
            if sequence in rest:
                repetition_count += 1
            total_sequences += 1
        
        # Check for repeated 3-word sequences
        for i in range(len(words) - 5):
            sequence = ' '.join(words[i:i+3])
            rest = ' '.join(words[i+3:])
            if sequence in rest:
                repetition_count += 2  # Weight longer sequences more
            total_sequences += 1
        
        if total_sequences == 0:
            return 0.0
        
        return min(1.0, repetition_count / total_sequences)
    
    def _is_mostly_repetitive(self, text: str, threshold: float = 0.3) -> bool:
        """
        Check if text is mostly repetitive.
        
        Args:
            text: Text to check
            threshold: Repetition threshold (0.0-1.0)
            
        Returns:
            True if text is highly repetitive
        """
        return self._calculate_repetition_score(text) > threshold
    
    def _is_too_short(self, text: str, min_words: int = 3) -> bool:
        """
        Check if text is too short to be meaningful.
        
        Args:
            text: Text to check
            min_words: Minimum number of words
            
        Returns:
            True if text is too short
        """
        return len(text.split()) < min_words
    
    def _is_gibberish(self, text: str) -> bool:
        """
        Check if text appears to be gibberish.
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears to be gibberish
        """
        if not text or len(text.strip()) == 0:
            return True
        
        # Check for URLs only or incomplete URLs
        if text.strip() in ['http://', 'https://', '']:
            return True
        
        # Check for excessive special characters or incomplete text
        alpha_chars = sum(c.isalpha() for c in text)
        total_chars = len(text.replace(' ', ''))
        
        if total_chars == 0:
            return True
        
        # If less than 30% alphabetic characters, likely gibberish
        if alpha_chars / total_chars < 0.3:
            return True
        
        return False
    
    def _echos_input(self, input_text: str, output_text: str) -> bool:
        """
        Check if output simply echoes/repeats the input.
        
        Args:
            input_text: User input
            output_text: Model output
            
        Returns:
            True if output echoes input
        """
        if not input_text or not output_text:
            return False
        
        # Normalize texts
        input_lower = input_text.lower().strip()
        output_lower = output_text.lower().strip()
        
        # Check if output starts with or contains most of the input
        if len(input_lower) > 10:
            # Check if output contains 80% or more of the input
            if input_lower in output_lower:
                return True
            
            # Check if they have significant overlap
            input_words = set(input_lower.split())
            output_words = set(output_lower.split())
            
            if len(input_words) > 3:
                overlap = len(input_words & output_words) / len(input_words)
                if overlap > 0.8:
                    return True
        
        return False
    
    def validate_conversation_quality(
        self, 
        conversation: Dict[str, str],
        check_repetition: bool = True,
        check_length: bool = True,
        check_gibberish: bool = True,
        check_echo: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Validate the quality of a single conversation.
        
        Args:
            conversation: Conversation dictionary with 'input' and 'output'
            check_repetition: Check for repetitive text
            check_length: Check for minimum length
            check_gibberish: Check for gibberish/nonsense
            check_echo: Check if output echoes input
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        input_text = conversation.get('input', '')
        output_text = conversation.get('output', '')
        
        # Check if output is empty
        if not output_text or len(output_text.strip()) == 0:
            issues.append("Output is empty")
        
        # Check for repetition
        if check_repetition and output_text:
            if self._is_mostly_repetitive(output_text):
                issues.append("Output is highly repetitive")
        
        # Check for minimum length
        if check_length and output_text:
            if self._is_too_short(output_text):
                issues.append("Output is too short")
        
        # Check for gibberish
        if check_gibberish and output_text:
            if self._is_gibberish(output_text):
                issues.append("Output appears to be gibberish")
        
        # Check for echo
        if check_echo and input_text and output_text:
            if self._echos_input(input_text, output_text):
                issues.append("Output echoes the input")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def analyze_dataset_quality(self) -> Dict[str, Any]:
        """
        Analyze the quality of the entire dataset.
        
        Returns:
            Dictionary with quality metrics and issues
        """
        if len(self.conversations) == 0:
            return {
                'total_conversations': 0,
                'valid_conversations': 0,
                'invalid_conversations': 0,
                'quality_score': 0.0,
                'issues_found': [],
                'recommendations': ['No conversations to analyze']
            }
        
        valid_count = 0
        issue_summary = {
            'empty_outputs': 0,
            'repetitive_outputs': 0,
            'short_outputs': 0,
            'gibberish_outputs': 0,
            'echo_outputs': 0
        }
        
        problematic_conversations = []
        
        for idx, conv in enumerate(self.conversations):
            is_valid, issues = self.validate_conversation_quality(conv)
            
            if is_valid:
                valid_count += 1
            else:
                # Track issue types
                for issue in issues:
                    if 'empty' in issue.lower():
                        issue_summary['empty_outputs'] += 1
                    elif 'repetitive' in issue.lower():
                        issue_summary['repetitive_outputs'] += 1
                    elif 'short' in issue.lower():
                        issue_summary['short_outputs'] += 1
                    elif 'gibberish' in issue.lower():
                        issue_summary['gibberish_outputs'] += 1
                    elif 'echo' in issue.lower():
                        issue_summary['echo_outputs'] += 1
                
                # Store first few problematic examples
                if len(problematic_conversations) < 5:
                    problematic_conversations.append({
                        'index': idx,
                        'input': conv.get('input', '')[:100],
                        'output': conv.get('output', '')[:100],
                        'issues': issues
                    })
        
        total = len(self.conversations)
        quality_score = valid_count / total if total > 0 else 0.0
        
        # Generate recommendations
        recommendations = []
        if quality_score < 0.5:
            recommendations.append("⚠️  CRITICAL: Less than 50% of conversations are high quality")
            recommendations.append("Training on this data will likely produce a poorly performing model")
            recommendations.append("Consider filtering or regenerating your training data")
        elif quality_score < 0.7:
            recommendations.append("⚠️  WARNING: Less than 70% of conversations are high quality")
            recommendations.append("Consider reviewing and filtering problematic conversations")
        else:
            recommendations.append("✓ Dataset quality is acceptable")
        
        if issue_summary['repetitive_outputs'] > total * 0.2:
            recommendations.append(f"High repetition detected in {issue_summary['repetitive_outputs']} conversations")
        
        if issue_summary['empty_outputs'] > 0:
            recommendations.append(f"{issue_summary['empty_outputs']} conversations have empty outputs")
        
        if issue_summary['gibberish_outputs'] > total * 0.1:
            recommendations.append(f"{issue_summary['gibberish_outputs']} conversations contain gibberish")
        
        if issue_summary['echo_outputs'] > total * 0.1:
            recommendations.append(f"{issue_summary['echo_outputs']} conversations echo the input")
        
        return {
            'total_conversations': total,
            'valid_conversations': valid_count,
            'invalid_conversations': total - valid_count,
            'quality_score': quality_score,
            'issue_summary': issue_summary,
            'problematic_examples': problematic_conversations,
            'recommendations': recommendations
        }
