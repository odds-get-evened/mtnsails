"""
Unit tests for TaberEnviroTrainer.

Tests cover:
- Training text generation and format
- is_trained() filesystem check
- train() delegation to LLMTrainer (mocked)
- retrain() when no model exists (delegates to train)
- retrain() when model exists (fine-tunes from checkpoint)
"""

import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the module under test via importlib to avoid heavy ML imports
import importlib.util

spec = importlib.util.spec_from_file_location(
    "taber_enviro_trainer",
    Path(__file__).parent.parent / "llm_interface" / "taber_enviro_trainer.py",
)
taber_enviro_trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(taber_enviro_trainer)

TaberEnviroTrainer = taber_enviro_trainer.TaberEnviroTrainer
_format_training_text = taber_enviro_trainer._format_training_text


class TestFormatTrainingText(unittest.TestCase):
    """Unit tests for the _format_training_text helper."""

    def test_forecast_item(self):
        """Forecast item is formatted with correct query type and duration."""
        item = {
            "query": "What will temperature be in 2 hours?",
            "category": "forecast",
            "metric": "temperature",
            "time_range": 120,
        }
        text = _format_training_text(item)
        self.assertIn("User: What will temperature be in 2 hours?", text)
        self.assertIn("Query Type: forecast", text)
        self.assertIn("Metric: temperature", text)
        self.assertIn("Duration: 120 minutes", text)

    def test_anomaly_item(self):
        """Anomaly detection category maps to 'anomaly' query type."""
        item = {
            "query": "Are there any anomalies in humidity?",
            "category": "anomaly_detection",
            "metric": "humidity",
        }
        text = _format_training_text(item)
        self.assertIn("Query Type: anomaly", text)
        self.assertIn("Metric: humidity", text)

    def test_gradient_item(self):
        """Gradient analysis category maps to 'gradient' query type."""
        item = {
            "query": "What is the temperature gradient across sensors?",
            "category": "gradient_analysis",
            "metric": "temperature",
        }
        text = _format_training_text(item)
        self.assertIn("Query Type: gradient", text)

    def test_correlation_item_maps_to_forecast(self):
        """Correlation category falls back to 'forecast' query type."""
        item = {
            "query": "How does temperature correlate with humidity?",
            "category": "correlation",
            "metrics": ["temperature", "humidity"],
        }
        text = _format_training_text(item)
        self.assertIn("Query Type: forecast", text)
        # First metric in list should be used
        self.assertIn("Metric: temperature", text)

    def test_missing_query_returns_empty(self):
        """Items without a 'query' field should produce an empty string."""
        item = {"category": "forecast", "metric": "temp"}
        text = _format_training_text(item)
        self.assertEqual(text, "")

    def test_default_duration_when_no_time_range(self):
        """Items without 'time_range' use 60-minute default."""
        item = {"query": "Any anomalies?", "category": "anomaly_detection", "metric": "temp"}
        text = _format_training_text(item)
        self.assertIn("Duration: 60 minutes", text)

    def test_default_metric_when_missing(self):
        """Items without any metric field default to 'temp'."""
        item = {"query": "Show forecast", "category": "forecast"}
        text = _format_training_text(item)
        self.assertIn("Metric: temp", text)

    def test_format_starts_with_user_prefix(self):
        """Formatted text must start with 'User:' to match LLM prompt format."""
        item = {"query": "Predict humidity", "category": "forecast", "metric": "humidity"}
        text = _format_training_text(item)
        self.assertTrue(text.startswith("User:"))

    def test_assistant_section_present(self):
        """Formatted text must contain 'Assistant:' section."""
        item = {"query": "Forecast light", "category": "forecast", "metric": "light"}
        text = _format_training_text(item)
        self.assertIn("Assistant:", text)


class TestTaberEnviroTrainerIsTrained(unittest.TestCase):
    """Tests for the is_trained() filesystem check."""

    def test_is_trained_false_when_empty_dir(self):
        """is_trained() returns False when output directory is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = TaberEnviroTrainer(output_dir=tmpdir)
            self.assertFalse(trainer.is_trained())

    def test_is_trained_false_when_dir_does_not_exist(self):
        """is_trained() returns False when output directory does not exist."""
        trainer = TaberEnviroTrainer(output_dir="/nonexistent/path/xyz")
        self.assertFalse(trainer.is_trained())

    def test_is_trained_true_with_safetensors(self):
        """is_trained() returns True when config.json + model.safetensors exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            (p / "config.json").write_text("{}")
            (p / "model.safetensors").write_bytes(b"")
            trainer = TaberEnviroTrainer(output_dir=tmpdir)
            self.assertTrue(trainer.is_trained())

    def test_is_trained_true_with_pytorch_bin(self):
        """is_trained() returns True when config.json + pytorch_model.bin exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            (p / "config.json").write_text("{}")
            (p / "pytorch_model.bin").write_bytes(b"")
            trainer = TaberEnviroTrainer(output_dir=tmpdir)
            self.assertTrue(trainer.is_trained())

    def test_is_trained_false_when_only_config(self):
        """is_trained() requires both config.json and a weights file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "config.json").write_text("{}")
            trainer = TaberEnviroTrainer(output_dir=tmpdir)
            self.assertFalse(trainer.is_trained())


class TestGenerateTrainingTexts(unittest.TestCase):
    """Tests for generate_training_texts()."""

    def setUp(self):
        self.trainer = TaberEnviroTrainer(output_dir="/tmp/mtnsails_test_model")

    def test_returns_nonempty_list(self):
        """generate_training_texts() should return a non-empty list."""
        texts = self.trainer.generate_training_texts(augment_factor=1, synthetic_count=0)
        self.assertIsInstance(texts, list)
        self.assertGreater(len(texts), 0)

    def test_texts_contain_user_prefix(self):
        """All returned texts should start with 'User:'."""
        texts = self.trainer.generate_training_texts(augment_factor=1, synthetic_count=0)
        for text in texts:
            self.assertTrue(text.startswith("User:"), f"Bad format: {text[:60]!r}")

    def test_texts_contain_assistant_section(self):
        """All returned texts should contain 'Assistant:'."""
        texts = self.trainer.generate_training_texts(augment_factor=1, synthetic_count=0)
        for text in texts:
            self.assertIn("Assistant:", text, f"Missing Assistant: in {text[:60]!r}")

    def test_synthetic_count_increases_output(self):
        """Providing synthetic_count > 0 should produce more texts."""
        base = self.trainer.generate_training_texts(augment_factor=1, synthetic_count=0)
        with_synthetic = self.trainer.generate_training_texts(
            augment_factor=1, synthetic_count=50, seed=42
        )
        self.assertGreater(len(with_synthetic), len(base))

    def test_seed_produces_reproducible_results(self):
        """Same seed should produce identical text lists."""
        texts1 = self.trainer.generate_training_texts(
            augment_factor=2, synthetic_count=30, seed=7
        )
        texts2 = self.trainer.generate_training_texts(
            augment_factor=2, synthetic_count=30, seed=7
        )
        self.assertEqual(texts1, texts2)

    def test_no_empty_strings(self):
        """generate_training_texts() must not return empty strings."""
        texts = self.trainer.generate_training_texts(augment_factor=1, synthetic_count=10)
        for text in texts:
            self.assertNotEqual(text.strip(), "", "Found empty string in training texts")


class TestTaberEnviroTrainerTrain(unittest.TestCase):
    """Tests for train() – LLMTrainer is mocked to avoid heavy ML deps."""

    def _make_src_trainer_mock(self, save_path: str):
        """Return a fake src.trainer module whose LLMTrainer is a mock."""
        mock_instance = MagicMock()
        mock_instance.save_model.return_value = save_path
        mock_class = MagicMock(return_value=mock_instance)

        mock_src_trainer = MagicMock()
        mock_src_trainer.LLMTrainer = mock_class

        return mock_src_trainer, mock_class, mock_instance

    def test_train_calls_llm_trainer(self):
        """train() should instantiate LLMTrainer, call train() then save_model()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = TaberEnviroTrainer(output_dir=tmpdir)
            texts = ["User: test\nAssistant: Query Type: forecast\nMetric: temp\nDuration: 60 minutes"]
            mock_src_trainer, mock_class, mock_instance = self._make_src_trainer_mock(tmpdir)

            with patch.object(trainer, "generate_training_texts", return_value=texts):
                with patch.dict(sys.modules, {"src.trainer": mock_src_trainer}):
                    result = trainer.train(num_epochs=1, batch_size=1)

            mock_class.assert_called_once()
            mock_instance.train.assert_called_once()
            mock_instance.save_model.assert_called_once()
            self.assertEqual(result, tmpdir)

    def test_train_passes_learning_rate(self):
        """train() must forward the learning_rate argument to LLMTrainer.train()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = TaberEnviroTrainer(output_dir=tmpdir)
            texts = ["User: q\nAssistant: Query Type: forecast\nMetric: temp\nDuration: 60 minutes"]
            mock_src_trainer, mock_class, mock_instance = self._make_src_trainer_mock(tmpdir)

            with patch.object(trainer, "generate_training_texts", return_value=texts):
                with patch.dict(sys.modules, {"src.trainer": mock_src_trainer}):
                    trainer.train(learning_rate=1e-4, num_epochs=2)

            call_kwargs = mock_instance.train.call_args
            self.assertEqual(call_kwargs.kwargs.get("learning_rate"), 1e-4)


class TestTaberEnviroTrainerRetrain(unittest.TestCase):
    """Tests for retrain() – LLMTrainer is mocked."""

    def test_retrain_delegates_to_train_when_not_trained(self):
        """retrain() should call train() when no model exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = TaberEnviroTrainer(output_dir=tmpdir)

            with patch.object(trainer, "train", return_value="/tmp/model") as mock_train:
                result = trainer.retrain(num_epochs=1)

            mock_train.assert_called_once()
            self.assertEqual(result, "/tmp/model")

    def test_retrain_uses_existing_checkpoint_when_trained(self):
        """retrain() should load from output_dir when a model already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Plant a fake trained model
            p = Path(tmpdir)
            (p / "config.json").write_text("{}")
            (p / "model.safetensors").write_bytes(b"")

            trainer = TaberEnviroTrainer(output_dir=tmpdir)

            mock_instance = MagicMock()
            mock_instance.save_model.return_value = tmpdir
            mock_class = MagicMock(return_value=mock_instance)
            mock_src_trainer = MagicMock()
            mock_src_trainer.LLMTrainer = mock_class

            texts = ["User: q\nAssistant: Query Type: forecast\nMetric: temp\nDuration: 60 minutes"]
            with patch.object(trainer, "generate_training_texts", return_value=texts):
                with patch.dict(sys.modules, {"src.trainer": mock_src_trainer}):
                    result = trainer.retrain(learning_rate=1e-5, num_epochs=1)

            # The model_name passed to LLMTrainer should be the existing output_dir
            ctor_kwargs = mock_class.call_args
            self.assertEqual(ctor_kwargs.kwargs.get("model_name"), tmpdir)

            mock_instance.train.assert_called_once()
            mock_instance.save_model.assert_called_once()
            self.assertEqual(result, tmpdir)

    def test_retrain_uses_lower_learning_rate_by_default(self):
        """retrain() default LR (1e-5) should be lower than train() default (5e-5)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            (p / "config.json").write_text("{}")
            (p / "model.safetensors").write_bytes(b"")

            trainer = TaberEnviroTrainer(output_dir=tmpdir)

            mock_instance = MagicMock()
            mock_instance.save_model.return_value = tmpdir
            mock_class = MagicMock(return_value=mock_instance)
            mock_src_trainer = MagicMock()
            mock_src_trainer.LLMTrainer = mock_class

            texts = ["User: q\nAssistant: Query Type: anomaly\nMetric: temp\nDuration: 60 minutes"]
            with patch.object(trainer, "generate_training_texts", return_value=texts):
                with patch.dict(sys.modules, {"src.trainer": mock_src_trainer}):
                    trainer.retrain()

            call_kwargs = mock_instance.train.call_args
            lr = call_kwargs.kwargs.get("learning_rate")
            self.assertIsNotNone(lr)
            self.assertLess(lr, 5e-5)


class TestTaberEnviroTrainerInit(unittest.TestCase):
    """Tests for __init__ defaults."""

    def test_default_model_name(self):
        trainer = TaberEnviroTrainer()
        self.assertEqual(trainer.model_name, "distilgpt2")

    def test_default_device(self):
        trainer = TaberEnviroTrainer()
        self.assertEqual(trainer.device, "cpu")

    def test_custom_output_dir(self):
        trainer = TaberEnviroTrainer(output_dir="/custom/path")
        self.assertEqual(trainer.output_dir, Path("/custom/path"))


if __name__ == "__main__":
    unittest.main()
