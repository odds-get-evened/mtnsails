"""
Taber Enviro LSTM - LLM Trainer

Trains and retrains the MTN Sails LLM on IoT sensor query data so that it
can parse natural language queries and route them to the correct Taber Enviro
LSTM analysis function (forecast / anomaly / gradient).

Usage::

    from llm_interface import TaberEnviroTrainer

    trainer = TaberEnviroTrainer(output_dir="./mtnsails_model")

    # Initial training
    model_path = trainer.train(synthetic_count=500, num_epochs=3)

    # Fine-tune on new data later
    model_path = trainer.retrain(synthetic_count=200, num_epochs=1)
"""

import sys
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional

# Ensure the project root is on the path so we can import src.trainer
_parent_dir = str(Path(__file__).parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Load training_data_generator directly to avoid pulling in the full
# llm_interface package (which imports mtnsails_bridge and numpy).
_tdg_spec = importlib.util.spec_from_file_location(
    "training_data_generator",
    Path(__file__).parent / "training_data_generator.py",
)
_tdg_module = importlib.util.module_from_spec(_tdg_spec)
_tdg_spec.loader.exec_module(_tdg_module)
create_iot_analyst_training_data = _tdg_module.create_iot_analyst_training_data
expand_training_data = _tdg_module.expand_training_data

# ---------------------------------------------------------------------------
# Intent response template
# ---------------------------------------------------------------------------
# This mirrors the structured format that MTNSailsLSTMBridge._llm_parse_query
# expects the LLM to produce.  Training the model on this format teaches it
# to extract intent fields (query type, metric, duration) from free-form NL
# queries.
_INTENT_TEMPLATE = (
    "Query Type: {query_type}\n"
    "Metric: {metric}\n"
    "Duration: {duration} minutes"
)

# Map training-data categories → bridge query types
_CATEGORY_TO_TYPE: Dict[str, str] = {
    "forecast": "forecast",
    "anomaly_detection": "anomaly",
    "gradient_analysis": "gradient",
    "correlation": "forecast",  # correlation queries fall back to forecast
}


def _format_training_text(item: Dict) -> str:
    """
    Format a single training-data item as a conversation text.

    The output format mirrors the prompt structure used by
    ``MTNSailsLSTMBridge._llm_parse_query`` so that the trained model
    learns to produce correctly structured intent responses.

    Args:
        item: A training-data dictionary produced by training_data_generator.

    Returns:
        A formatted conversation string, or an empty string if the item
        has no ``query`` field.
    """
    query = item.get("query", "")
    if not query:
        return ""

    category = item.get("category", "forecast")
    query_type = _CATEGORY_TO_TYPE.get(category, "forecast")

    # Prefer explicit single metric; fall back to first element of metrics list
    if "metric" in item:
        metric = item["metric"]
    elif "metrics" in item and item["metrics"]:
        metric = item["metrics"][0]
    else:
        metric = "temp"

    # Use stored time_range (in minutes) or default to 60 minutes
    duration = item.get("time_range", 60)

    intent_text = _INTENT_TEMPLATE.format(
        query_type=query_type,
        metric=metric,
        duration=duration,
    )
    return f"User: {query}\nAssistant: {intent_text}"


class TaberEnviroTrainer:
    """
    Trains and retrains the MTN Sails LLM on Taber Enviro IoT training data.

    This class wires together the IoT training-data generator with the
    ``LLMTrainer`` (from ``src.trainer``) so that the language model learns
    to parse sensor-related natural-language queries and route them to the
    appropriate Taber Enviro LSTM analysis function.

    After training, point ``MTNSailsLSTMBridge`` at the saved model directory
    to enable LLM-powered query parsing and result explanation.
    """

    def __init__(
        self,
        model_name: str = "distilgpt2",
        output_dir: str = "./mtnsails_model",
        device: str = "cpu",
    ):
        """
        Initialize the trainer.

        Args:
            model_name: HuggingFace model name used for *initial* training.
                        Ignored when retraining from an existing checkpoint.
            output_dir: Directory to save / load the trained model.
            device:     Compute device (``'cpu'`` or ``'cuda'``).
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.device = device

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def is_trained(self) -> bool:
        """Return ``True`` if a trained model already exists at *output_dir*."""
        has_config = (self.output_dir / "config.json").exists()
        has_weights = (
            (self.output_dir / "model.safetensors").exists()
            or (self.output_dir / "pytorch_model.bin").exists()
        )
        return has_config and has_weights

    def generate_training_texts(
        self,
        augment_factor: int = 2,
        synthetic_count: int = 100,
        diversity_mode: bool = False,
        seed: Optional[int] = None,
    ) -> List[str]:
        """
        Generate formatted training texts for LLM fine-tuning.

        Each text is a single conversation turn::

            User: <natural-language sensor query>
            Assistant: Query Type: <type>
            Metric: <metric>
            Duration: <N> minutes

        Args:
            augment_factor:  Number of paraphrase variations per base example.
            synthetic_count: Number of additional synthetic examples to generate.
            diversity_mode:  When ``True``, include extended metric types
                             (CO2, noise, UV index, etc.).
            seed:            Random seed for reproducible generation.

        Returns:
            List of non-empty formatted training strings.
        """
        base_data = create_iot_analyst_training_data()
        expanded = expand_training_data(
            base_data,
            augment_factor=augment_factor,
            synthetic_count=synthetic_count,
            diversity_mode=diversity_mode,
            seed=seed,
        )
        texts = [_format_training_text(item) for item in expanded]
        # Drop empty strings (items that had no 'query' field)
        return [t for t in texts if t]

    # ------------------------------------------------------------------
    # Training entry points
    # ------------------------------------------------------------------

    def train(
        self,
        augment_factor: int = 2,
        synthetic_count: int = 100,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        diversity_mode: bool = False,
        seed: Optional[int] = None,
    ) -> str:
        """
        Train the MTN Sails LLM from scratch on IoT sensor query data.

        Args:
            augment_factor:  Paraphrase variations per base example.
            synthetic_count: Synthetic examples to generate.
            num_epochs:      Training epochs.
            batch_size:      Training batch size.
            learning_rate:   Initial learning rate.
            diversity_mode:  Enable extended metric variety.
            seed:            Random seed.

        Returns:
            Path to the saved model directory (string).
        """
        from src.trainer import LLMTrainer

        train_texts = self.generate_training_texts(
            augment_factor=augment_factor,
            synthetic_count=synthetic_count,
            diversity_mode=diversity_mode,
            seed=seed,
        )
        print(f"Training on {len(train_texts)} IoT query examples...")

        llm_trainer = LLMTrainer(
            model_name=self.model_name,
            output_dir=str(self.output_dir),
            device=self.device,
        )
        llm_trainer.train(
            train_texts=train_texts,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        return llm_trainer.save_model()

    def retrain(
        self,
        augment_factor: int = 2,
        synthetic_count: int = 100,
        num_epochs: int = 1,
        batch_size: int = 4,
        learning_rate: float = 1e-5,
        diversity_mode: bool = False,
        seed: Optional[int] = None,
    ) -> str:
        """
        Fine-tune an existing trained model, or train from scratch if none exists.

        When a previously trained model is found at *output_dir*, it is loaded
        as the starting checkpoint and fine-tuned with a conservative learning
        rate to avoid catastrophic forgetting.  If no model is found, this
        method delegates to :meth:`train`.

        Args:
            augment_factor:  Paraphrase variations per base example.
            synthetic_count: Synthetic examples to generate.
            num_epochs:      Fine-tuning epochs (typically 1).
            batch_size:      Training batch size.
            learning_rate:   Fine-tuning learning rate (should be ≤ initial LR).
            diversity_mode:  Enable extended metric variety.
            seed:            Random seed.

        Returns:
            Path to the saved model directory (string).
        """
        if not self.is_trained():
            print("No existing model found – performing initial training.")
            return self.train(
                augment_factor=augment_factor,
                synthetic_count=synthetic_count,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                diversity_mode=diversity_mode,
                seed=seed,
            )

        from src.trainer import LLMTrainer

        train_texts = self.generate_training_texts(
            augment_factor=augment_factor,
            synthetic_count=synthetic_count,
            diversity_mode=diversity_mode,
            seed=seed,
        )
        print(
            f"Retraining existing model on {len(train_texts)} IoT query examples "
            f"(lr={learning_rate})..."
        )

        # Load from the existing checkpoint so fine-tuning preserves prior knowledge
        llm_trainer = LLMTrainer(
            model_name=str(self.output_dir),
            output_dir=str(self.output_dir),
            device=self.device,
        )
        llm_trainer.train(
            train_texts=train_texts,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        return llm_trainer.save_model()
