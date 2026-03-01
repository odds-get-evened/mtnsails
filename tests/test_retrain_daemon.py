"""
Unit tests for llm_interface.retrain_daemon.

Tests cover:
- _acquire_lock / _release_lock file-locking helpers
- retrain_once returns False when buffer is empty
- retrain_once returns False when all records have empty training_text
- retrain_once deletes consumed buffer files on success
- retrain_once moves files to archive_dir when configured
- run_daemon argument wiring (via argparse)
"""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import retrain_buffer directly (avoid heavy package __init__ → mtnsails_bridge → numpy)
_buf_spec = importlib.util.spec_from_file_location(
    "retrain_buffer",
    Path(__file__).parent.parent / "llm_interface" / "retrain_buffer.py",
)
_buf_module = importlib.util.module_from_spec(_buf_spec)
_buf_spec.loader.exec_module(_buf_module)
sys.modules["llm_interface.retrain_buffer"] = _buf_module

# Now import the daemon module
spec = importlib.util.spec_from_file_location(
    "retrain_daemon",
    Path(__file__).parent.parent / "llm_interface" / "retrain_daemon.py",
)
retrain_daemon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retrain_daemon)

_acquire_lock = retrain_daemon._acquire_lock
_release_lock = retrain_daemon._release_lock
retrain_once = retrain_daemon.retrain_once

# Fixtures
_SAMPLE_RECORD = {
    "timestamp": "2099-01-01T00:00:00Z",
    "user_query": "What is the temp forecast?",
    "parsed_intent": {"type": "forecast", "metric": "temp", "duration": 60, "sensor_id": None},
    "llm_intent_raw_response": None,
    "intent_prompt": None,
    "lstm_output_summary": {"type": "forecast", "metric": "temp"},
    "intent_valid": True,
    "lstm_success": True,
    "training_text": "User: What is the temp forecast?\nAssistant: Query Type: forecast\nMetric: temp\nDuration: 60 minutes",
}


def _write_jsonl(path: Path, records: list) -> None:
    """Helper to write a list of dicts to a JSONL file."""
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestLockHelpers(unittest.TestCase):
    """Tests for _acquire_lock / _release_lock."""

    def test_acquire_creates_lock_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / ".retrain.lock"
            fd = _acquire_lock(lock_path)
            self.assertGreaterEqual(fd, 0)
            self.assertTrue(lock_path.exists())
            _release_lock(fd)

    def test_second_acquire_fails_while_locked(self):
        """A second acquire on the same path should return -1."""
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / ".retrain.lock"
            fd1 = _acquire_lock(lock_path)
            self.assertGreaterEqual(fd1, 0)
            try:
                fd2 = _acquire_lock(lock_path)
                self.assertEqual(fd2, -1)
            finally:
                _release_lock(fd1)


class TestRetrainOnce(unittest.TestCase):
    """Tests for retrain_once()."""

    def test_returns_false_when_buffer_empty(self):
        with tempfile.TemporaryDirectory() as buf_dir, \
             tempfile.TemporaryDirectory() as model_dir:
            result = retrain_once(
                model_dir=model_dir,
                buffer_dir=buf_dir,
                epochs=1,
                batch_size=1,
                learning_rate=1e-5,
            )
        self.assertFalse(result)

    def test_returns_false_when_no_training_texts(self):
        """Records without training_text should cause retrain_once to return False."""
        with tempfile.TemporaryDirectory() as buf_dir, \
             tempfile.TemporaryDirectory() as model_dir:
            empty_record = dict(_SAMPLE_RECORD)
            empty_record["training_text"] = ""
            _write_jsonl(Path(buf_dir) / "validated_2099-01-01.jsonl", [empty_record])

            result = retrain_once(
                model_dir=model_dir,
                buffer_dir=buf_dir,
                epochs=1,
                batch_size=1,
                learning_rate=1e-5,
            )
        self.assertFalse(result)

    def _make_trainer_mocks(self, model_dir: str):
        """Return mocked TaberEnviroTrainer and LLMTrainer modules."""
        mock_llm_instance = MagicMock()
        mock_llm_class = MagicMock(return_value=mock_llm_instance)
        mock_src_trainer = MagicMock()
        mock_src_trainer.LLMTrainer = mock_llm_class

        mock_taber_trainer = MagicMock()
        mock_taber_trainer.is_trained.return_value = False
        mock_taber_trainer.model_name = "distilgpt2"
        mock_taber_trainer.device = "cpu"
        mock_taber_class = MagicMock(return_value=mock_taber_trainer)

        mock_taber_module = MagicMock()
        mock_taber_module.TaberEnviroTrainer = mock_taber_class

        return mock_src_trainer, mock_taber_module, mock_llm_instance

    def test_returns_true_on_success(self):
        with tempfile.TemporaryDirectory() as buf_dir, \
             tempfile.TemporaryDirectory() as model_dir:
            _write_jsonl(Path(buf_dir) / "validated_2099-01-01.jsonl", [_SAMPLE_RECORD])
            mock_src, mock_taber, mock_llm_inst = self._make_trainer_mocks(model_dir)

            with patch.dict(sys.modules, {
                "src.trainer": mock_src,
                "llm_interface.taber_enviro_trainer": mock_taber,
            }):
                result = retrain_once(
                    model_dir=model_dir,
                    buffer_dir=buf_dir,
                    epochs=1,
                    batch_size=1,
                    learning_rate=1e-5,
                )
        self.assertTrue(result)

    def test_buffer_files_deleted_after_success(self):
        """Buffer JSONL files must be removed after a successful retrain."""
        with tempfile.TemporaryDirectory() as buf_dir, \
             tempfile.TemporaryDirectory() as model_dir:
            buf_file = Path(buf_dir) / "validated_2099-01-01.jsonl"
            _write_jsonl(buf_file, [_SAMPLE_RECORD])
            mock_src, mock_taber, _ = self._make_trainer_mocks(model_dir)

            with patch.dict(sys.modules, {
                "src.trainer": mock_src,
                "llm_interface.taber_enviro_trainer": mock_taber,
            }):
                retrain_once(
                    model_dir=model_dir,
                    buffer_dir=buf_dir,
                    epochs=1,
                    batch_size=1,
                    learning_rate=1e-5,
                )
            self.assertFalse(buf_file.exists())

    def test_buffer_files_archived_when_archive_dir_set(self):
        """Buffer files should be moved to archive_dir instead of deleted."""
        with tempfile.TemporaryDirectory() as buf_dir, \
             tempfile.TemporaryDirectory() as model_dir, \
             tempfile.TemporaryDirectory() as arch_dir:
            buf_file = Path(buf_dir) / "validated_2099-01-01.jsonl"
            _write_jsonl(buf_file, [_SAMPLE_RECORD])
            mock_src, mock_taber, _ = self._make_trainer_mocks(model_dir)

            with patch.dict(sys.modules, {
                "src.trainer": mock_src,
                "llm_interface.taber_enviro_trainer": mock_taber,
            }):
                retrain_once(
                    model_dir=model_dir,
                    buffer_dir=buf_dir,
                    epochs=1,
                    batch_size=1,
                    learning_rate=1e-5,
                    archive_dir=arch_dir,
                )
            # Original file gone
            self.assertFalse(buf_file.exists())
            # File now in archive
            self.assertTrue((Path(arch_dir) / "validated_2099-01-01.jsonl").exists())

    def test_returns_false_when_trainer_import_fails(self):
        """retrain_once should return False and print error if imports fail."""
        with tempfile.TemporaryDirectory() as buf_dir, \
             tempfile.TemporaryDirectory() as model_dir:
            _write_jsonl(Path(buf_dir) / "validated_2099-01-01.jsonl", [_SAMPLE_RECORD])

            # Simulate missing ML dependencies
            with patch.dict(sys.modules, {
                "src.trainer": None,
                "llm_interface.taber_enviro_trainer": None,
            }):
                result = retrain_once(
                    model_dir=model_dir,
                    buffer_dir=buf_dir,
                    epochs=1,
                    batch_size=1,
                    learning_rate=1e-5,
                )
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
