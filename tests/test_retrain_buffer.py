"""
Unit tests for llm_interface.retrain_buffer.

Tests cover:
- append_validated_interaction writes a valid JSONL record
- load_buffer_records reads records back correctly
- count_buffer_records returns accurate totals
- Multiple records accumulate in the same daily file
- load_buffer_records on a missing directory returns []
- Record fields (intent_valid, lstm_success, training_text) are preserved
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
import importlib.util

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import retrain_buffer directly to avoid triggering the full package __init__
# (which pulls in mtnsails_bridge → numpy, an optional heavy dependency)
_buf_spec = importlib.util.spec_from_file_location(
    "retrain_buffer",
    Path(__file__).parent.parent / "llm_interface" / "retrain_buffer.py",
)
_buf_module = importlib.util.module_from_spec(_buf_spec)
_buf_spec.loader.exec_module(_buf_module)

append_validated_interaction = _buf_module.append_validated_interaction
load_buffer_records = _buf_module.load_buffer_records
count_buffer_records = _buf_module.count_buffer_records

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_INTENT = {"type": "forecast", "metric": "temp", "duration": 60, "sensor_id": None}
_SAMPLE_LSTM = {"type": "forecast", "metric": "temp", "predictions": [23.1, 23.5]}
_SAMPLE_TRAINING_TEXT = "User: What is the temp forecast?\nAssistant: Query Type: forecast\nMetric: temp\nDuration: 60 minutes"


class TestAppendValidatedInteraction(unittest.TestCase):
    """Tests for append_validated_interaction."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.buf_dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_creates_jsonl_file(self):
        """A JSONL file should be created on first append."""
        append_validated_interaction(
            buffer_dir=self.buf_dir,
            user_query="What is the temp forecast?",
            parsed_intent=_SAMPLE_INTENT,
            lstm_output=_SAMPLE_LSTM,
            training_text=_SAMPLE_TRAINING_TEXT,
        )
        jsonl_files = list(Path(self.buf_dir).glob("validated_*.jsonl"))
        self.assertEqual(len(jsonl_files), 1)

    def test_record_is_valid_json(self):
        """Each appended line should be valid JSON."""
        append_validated_interaction(
            buffer_dir=self.buf_dir,
            user_query="Any anomalies in humidity?",
            parsed_intent={"type": "anomaly", "metric": "humidity", "duration": 60, "sensor_id": None},
            lstm_output={"type": "anomaly", "metric": "humidity", "anomaly_score": 0.1},
            training_text="User: Any anomalies in humidity?\nAssistant: Query Type: anomaly\nMetric: humidity\nDuration: 60 minutes",
        )
        jsonl_file = next(Path(self.buf_dir).glob("validated_*.jsonl"))
        with open(jsonl_file) as f:
            lines = [l.strip() for l in f if l.strip()]
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertIsInstance(record, dict)

    def test_record_fields_present(self):
        """Record must contain all required fields."""
        append_validated_interaction(
            buffer_dir=self.buf_dir,
            user_query="Forecast light for 30 minutes",
            parsed_intent={"type": "forecast", "metric": "light", "duration": 30, "sensor_id": None},
            lstm_output={"type": "forecast", "metric": "light", "predictions": [100.0]},
            training_text="User: Forecast light for 30 minutes\nAssistant: Query Type: forecast\nMetric: light\nDuration: 30 minutes",
            llm_intent_raw_response="Query Type: forecast\nMetric: light\nDuration: 30 minutes",
            intent_prompt="Analyze this IoT sensor query...",
        )
        record = load_buffer_records(self.buf_dir)[0]
        for field in (
            "timestamp", "user_query", "parsed_intent",
            "llm_intent_raw_response", "intent_prompt",
            "lstm_output_summary", "intent_valid", "lstm_success", "training_text",
        ):
            self.assertIn(field, record, f"Missing field: {field}")

    def test_intent_valid_and_lstm_success_are_true(self):
        """intent_valid and lstm_success must always be True in the record."""
        append_validated_interaction(
            buffer_dir=self.buf_dir,
            user_query="q",
            parsed_intent=_SAMPLE_INTENT,
            lstm_output=_SAMPLE_LSTM,
            training_text=_SAMPLE_TRAINING_TEXT,
        )
        record = load_buffer_records(self.buf_dir)[0]
        self.assertTrue(record["intent_valid"])
        self.assertTrue(record["lstm_success"])

    def test_training_text_preserved(self):
        """The training_text in the record must match the input."""
        append_validated_interaction(
            buffer_dir=self.buf_dir,
            user_query="q",
            parsed_intent=_SAMPLE_INTENT,
            lstm_output=_SAMPLE_LSTM,
            training_text=_SAMPLE_TRAINING_TEXT,
        )
        record = load_buffer_records(self.buf_dir)[0]
        self.assertEqual(record["training_text"], _SAMPLE_TRAINING_TEXT)

    def test_multiple_records_accumulate(self):
        """Multiple calls should append multiple lines to the same daily file."""
        for i in range(5):
            append_validated_interaction(
                buffer_dir=self.buf_dir,
                user_query=f"query {i}",
                parsed_intent=_SAMPLE_INTENT,
                lstm_output=_SAMPLE_LSTM,
                training_text=_SAMPLE_TRAINING_TEXT,
            )
        self.assertEqual(count_buffer_records(self.buf_dir), 5)

    def test_returns_path_to_buffer_file(self):
        """append_validated_interaction should return a Path object."""
        result = append_validated_interaction(
            buffer_dir=self.buf_dir,
            user_query="q",
            parsed_intent=_SAMPLE_INTENT,
            lstm_output=_SAMPLE_LSTM,
            training_text=_SAMPLE_TRAINING_TEXT,
        )
        self.assertIsInstance(result, Path)
        self.assertTrue(result.exists())

    def test_creates_buffer_dir_if_absent(self):
        """Buffer directory should be created automatically when it does not exist."""
        new_dir = Path(self.buf_dir) / "nested" / "buffer"
        append_validated_interaction(
            buffer_dir=str(new_dir),
            user_query="q",
            parsed_intent=_SAMPLE_INTENT,
            lstm_output=_SAMPLE_LSTM,
            training_text=_SAMPLE_TRAINING_TEXT,
        )
        self.assertTrue(new_dir.is_dir())
        self.assertEqual(count_buffer_records(str(new_dir)), 1)


class TestLoadBufferRecords(unittest.TestCase):
    """Tests for load_buffer_records."""

    def test_returns_empty_list_for_missing_dir(self):
        """load_buffer_records on a non-existent directory returns []."""
        result = load_buffer_records("/nonexistent/path/xyz_buffer")
        self.assertEqual(result, [])

    def test_skips_malformed_lines(self):
        """Malformed JSON lines should be silently skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            buf_file = Path(tmp) / "validated_2099-01-01.jsonl"
            buf_file.write_text('{"id": 1, "valid": true}\nnot-json\n{"id": 2, "valid": true}\n')
            records = load_buffer_records(tmp)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["id"], 1)
        self.assertEqual(records[1]["id"], 2)

    def test_returns_records_in_file_order(self):
        """Records should be returned in the order they appear (ascending date)."""
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(3):
                append_validated_interaction(
                    buffer_dir=tmp,
                    user_query=f"q{i}",
                    parsed_intent=_SAMPLE_INTENT,
                    lstm_output=_SAMPLE_LSTM,
                    training_text=f"text {i}",
                )
            records = load_buffer_records(tmp)
        queries = [r["user_query"] for r in records]
        self.assertEqual(queries, ["q0", "q1", "q2"])


class TestCountBufferRecords(unittest.TestCase):
    """Tests for count_buffer_records."""

    def test_zero_for_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(count_buffer_records(tmp), 0)

    def test_zero_for_missing_dir(self):
        self.assertEqual(count_buffer_records("/nonexistent/abc"), 0)

    def test_counts_all_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            for _ in range(7):
                append_validated_interaction(
                    buffer_dir=tmp,
                    user_query="q",
                    parsed_intent=_SAMPLE_INTENT,
                    lstm_output=_SAMPLE_LSTM,
                    training_text=_SAMPLE_TRAINING_TEXT,
                )
            self.assertEqual(count_buffer_records(tmp), 7)


if __name__ == "__main__":
    unittest.main()
