"""
LLM Trainer for fine-tuning language models on conversation data.
"""

import warnings
import torch
import gc
import shutil
import tempfile
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from typing import List, Optional
from pathlib import Path

# Suppress specific warnings from transformers and torch
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')
# Suppress loss_type deprecation warning - this parameter is deprecated in transformers
# and should not be used. The model automatically uses the correct loss function (cross-entropy)
# for causal language modeling. See docs/LOSS_CONFIGURATION.md for details.
warnings.filterwarnings('ignore', message='.*loss_type.*')


class LLMTrainer:
    """Handles training and fine-tuning of language models."""
    
    def __init__(
        self,
        model_name: str = "distilgpt2",
        output_dir: str = "./trained_model",
        device: str = "cpu"
    ):
        """
        Initialize the LLM trainer.
        
        Args:
            model_name: Name of the pre-trained model to use
            output_dir: Directory to save the trained model
            device: Device to use for training ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.device = device
        self.model = None
        self.tokenizer = None
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the pre-trained model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float32  # Use float32 for training stability
        )
        self.model.to(self.device)
    
    def prepare_dataset(self, texts: List[str], max_length: int = 256) -> Dataset:
        """
        Prepare dataset for training.
        
        Args:
            texts: List of text strings to train on
            max_length: Maximum sequence length
            
        Returns:
            Prepared dataset
        """
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors=None
            )
        
        dataset = Dataset.from_dict({'text': texts})
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def train(
        self,
        train_texts: List[str],
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        save_steps: int = 100,
        max_length: int = 256
    ) -> None:
        """
        Train the model on provided texts.
        
        Args:
            train_texts: List of training texts
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
            save_steps: Save checkpoint every N steps
            max_length: Maximum sequence length
        """
        print(f"Preparing dataset with {len(train_texts)} samples...")
        train_dataset = self.prepare_dataset(train_texts, max_length)
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            save_steps=save_steps,
            save_total_limit=2,
            logging_steps=10,
            logging_dir=str(self.output_dir / "logs"),
            report_to="none",  # Disable reporting to external services
            use_cpu=(self.device == "cpu")
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator
        )
        
        print("Starting training...")
        trainer.train()
        print("Training completed!")
    
    def save_model(self, save_path: Optional[str] = None) -> str:
        """
        Save the trained model and tokenizer.
        
        Args:
            save_path: Path to save the model (defaults to output_dir)
            
        Returns:
            Path where model was saved
        """
        save_path = save_path or str(self.output_dir)
        save_path = Path(save_path)
        
        print(f"Saving model to {save_path}")
        
        # On Windows, safetensors may keep files memory-mapped which prevents overwriting.
        # To work around this, we save to a temporary directory first, then move the files.
        # This ensures clean file handles and prevents "os error 1224" on Windows.
        
        # Store the original dtype to preserve it after reload
        original_dtype = self.model.dtype if hasattr(self.model, 'dtype') else torch.float32
        
        try:
            # Create a temporary directory for saving
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Save model and tokenizer to temporary location
                self.model.save_pretrained(temp_path, safe_serialization=True)
                self.tokenizer.save_pretrained(temp_path)
                
                # Explicitly free model and tokenizer references to release file handles
                # This is crucial on Windows to avoid memory-mapped file issues
                del self.model
                del self.tokenizer
                gc.collect()
                
                # Now move files from temp to final destination
                save_path.mkdir(parents=True, exist_ok=True)
                
                # Copy all files from temp directory to final destination
                for item in temp_path.iterdir():
                    dest = save_path / item.name
                    if item.is_file():
                        # Remove existing file if it exists to avoid conflicts
                        if dest.exists():
                            dest.unlink()
                        shutil.copy2(item, dest)
                    elif item.is_dir():
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.copytree(item, dest)
            
            return str(save_path)
            
        except Exception as e:
            print(f"Error while saving model: {e}")
            # Re-raise with more context
            raise RuntimeError(f"Failed to save model to {save_path}: {e}") from e
        finally:
            # Always reload the model and tokenizer to keep trainer in a usable state
            # even if an error occurred during saving
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(str(save_path))
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(save_path),
                    dtype=original_dtype  # Preserve original dtype
                )
                self.model.to(self.device)
            except Exception as reload_error:
                print(f"Warning: Could not reload model after save: {reload_error}")
                # If reload fails, at least try to restore from the original model_name
                # This ensures the trainer is not left in an unusable state
                try:
                    self._load_model()
                except Exception as fallback_error:
                    print(f"Critical: Could not restore model: {fallback_error}")
    
    def load_trained_model(self, model_path: str) -> None:
        """
        Load a previously trained model.
        
        Args:
            model_path: Path to the trained model
        """
        print(f"Loading trained model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float32
        )
        self.model.to(self.device)
