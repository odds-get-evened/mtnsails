"""
Qwen2.5-1.5B model builder for fine-tuning on conversation data.

Downloads safetensors weights directly from Hugging Face.
CPU-friendly defaults: batch_size=1, max_length=256, gradient_accumulation=4.
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

warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')
warnings.filterwarnings('ignore', message='.*loss_type.*')

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B"


class QwenBuilder:
    """
    Loads, fine-tunes, and saves Qwen/Qwen2.5-1.5B models.

    Mirrors the LLMTrainer API so it can be swapped in anywhere that class is
    used, but ships with CPU-friendly defaults appropriate for a 1.5 B model:
      - batch_size=1 (with gradient_accumulation_steps=4 to stay stable)
      - max_length=256 tokens
      - learning_rate=2e-5
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_ID,
        output_dir: str = "./qwen_trained_model",
        device: str = "cpu"
    ):
        """
        Initialize the Qwen builder.

        Args:
            model_name: Hugging Face model ID (default: Qwen/Qwen2.5-1.5B)
            output_dir: Directory to save checkpoints and the final model
            device: Compute device ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.device = device
        self.model = None
        self.tokenizer = None

        self._load_model()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Download (on first use) and load the model + tokenizer."""
        print(f"Loading model: {self.model_name}")
        print("Note: Qwen2.5-1.5B weights are ~3 GB; first run will download them.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=False  # Qwen2.5 is natively supported in transformers >=4.37
        )

        # Qwen2.5 tokenizer uses <|endoftext|> for EOS; mirror it to PAD so
        # the DataCollator can build attention masks correctly.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # float32 for stable CPU training
            trust_remote_code=False
        )
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Public API (matches LLMTrainer)
    # ------------------------------------------------------------------

    def prepare_dataset(self, texts: List[str], max_length: int = 256) -> Dataset:
        """
        Tokenise a list of training texts into a Hugging Face Dataset.

        Args:
            texts: Formatted training strings
            max_length: Token sequence length (256 recommended for CPU)

        Returns:
            Tokenised Dataset ready for the Trainer
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
        batch_size: int = 1,
        learning_rate: float = 2e-5,
        save_steps: int = 100,
        max_length: int = 256,
        max_steps: int = -1
    ) -> None:
        """
        Fine-tune the model on conversation texts.

        Args:
            train_texts: List of formatted training strings
            num_epochs: Training epochs (overridden if max_steps > 0)
            batch_size: Per-device batch size (1 is safe on CPU for 1.5 B)
            learning_rate: Peak learning rate (2e-5 suits instruction fine-tuning)
            save_steps: Checkpoint frequency in gradient steps
            max_length: Maximum sequence length in tokens
            max_steps: Hard cap on gradient steps (-1 = epoch-based)
        """
        print(f"Preparing dataset with {len(train_texts)} samples...")
        print("Note: Training Qwen2.5-1.5B on CPU is slow. Use a focused dataset.")

        train_dataset = self.prepare_dataset(train_texts, max_length)

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            max_steps=max_steps,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,   # effective batch = batch_size * 4
            learning_rate=learning_rate,
            save_steps=save_steps,
            save_total_limit=2,
            logging_steps=10,
            logging_dir=str(self.output_dir / "logs"),
            report_to="none",
            use_cpu=(self.device == "cpu")
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # causal LM
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
        Save the model and tokenizer as safetensors.

        Uses a write-to-temp-then-move strategy to avoid memory-mapped file
        conflicts on Windows (same approach as LLMTrainer).

        Args:
            save_path: Destination directory (defaults to output_dir)

        Returns:
            Absolute path where the model was saved
        """
        save_path = save_path or str(self.output_dir)
        save_path = Path(save_path)

        print(f"Saving model to {save_path}")

        original_dtype = self.model.dtype if hasattr(self.model, 'dtype') else torch.float32

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Write safetensors to temp first
                self.model.save_pretrained(temp_path, safe_serialization=True)
                self.tokenizer.save_pretrained(temp_path)

                # Release file handles before moving
                del self.model
                del self.tokenizer
                gc.collect()

                save_path.mkdir(parents=True, exist_ok=True)

                for item in temp_path.iterdir():
                    dest = save_path / item.name
                    if item.is_file():
                        if dest.exists():
                            dest.unlink()
                        shutil.copy2(item, dest)
                    elif item.is_dir():
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.copytree(item, dest)

            return str(save_path)

        except Exception as e:
            raise RuntimeError(f"Failed to save model to {save_path}: {e}") from e

        finally:
            # Always restore the trainer to a usable state
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(str(save_path))
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(save_path),
                    torch_dtype=original_dtype
                )
                self.model.to(self.device)
            except Exception as reload_error:
                print(f"Warning: Could not reload model after save: {reload_error}")
                try:
                    self._load_model()
                except Exception as fallback_error:
                    print(f"Critical: Could not restore model: {fallback_error}")

    def load_trained_model(self, model_path: str) -> None:
        """
        Load a previously saved Qwen2.5 checkpoint for continued training.

        Args:
            model_path: Path to a directory containing config.json + safetensors
        """
        print(f"Loading trained model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32
        )
        self.model.to(self.device)
