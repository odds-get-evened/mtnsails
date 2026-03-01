"""
Tests for new JSONL support in ConversationDataHandler and daemon state helpers.
"""

import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_handler import ConversationDataHandler
from src.daemon import load_daemon_state, save_daemon_state, count_jsonl_lines


class TestJSONLSupport(unittest.TestCase):
    """Tests for JSONL read/write methods on ConversationDataHandler."""

    def _write_jsonl(self, path: Path, records: list) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            for rec in records:
                f.write(json.dumps(rec) + '\n')

    def test_append_to_jsonl_creates_file(self):
        """append_to_jsonl should create the file if it does not exist."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pairs.jsonl"
            handler = ConversationDataHandler()
            handler.append_to_jsonl({"input": "hi", "output": "hello"}, str(out))
            self.assertTrue(out.exists())
            with open(out) as f:
                record = json.loads(f.readline())
            self.assertEqual(record['input'], 'hi')
            self.assertEqual(record['output'], 'hello')

    def test_append_to_jsonl_accumulates(self):
        """Multiple appends should produce multiple JSONL lines."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pairs.jsonl"
            handler = ConversationDataHandler()
            handler.append_to_jsonl({"input": "q1", "output": "a1"}, str(out))
            handler.append_to_jsonl({"input": "q2", "output": "a2"}, str(out))
            lines = out.read_text().strip().split('\n')
            self.assertEqual(len(lines), 2)

    def test_append_to_jsonl_invalid_raises(self):
        """append_to_jsonl should raise ValueError for records missing required keys."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pairs.jsonl"
            handler = ConversationDataHandler()
            with self.assertRaises(ValueError):
                handler.append_to_jsonl({"only_key": "value"}, str(out))

    def test_load_from_jsonl_basic(self):
        """load_from_jsonl should load all valid records."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            records = [
                {"input": "Q1", "output": "A1"},
                {"input": "Q2", "output": "A2", "timestamp": "2024-01-01", "accepted": True},
            ]
            self._write_jsonl(path, records)

            handler = ConversationDataHandler()
            total = handler.load_from_jsonl(str(path))
            self.assertEqual(len(handler), 2)
            self.assertEqual(total, 2)

    def test_load_from_jsonl_with_offset(self):
        """load_from_jsonl with start_line should skip already-processed lines."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            records = [
                {"input": f"Q{i}", "output": f"A{i}"}
                for i in range(5)
            ]
            self._write_jsonl(path, records)

            handler = ConversationDataHandler()
            total = handler.load_from_jsonl(str(path), start_line=3)
            # Lines 4 and 5 (1-indexed) should be loaded
            self.assertEqual(len(handler), 2)
            self.assertEqual(total, 5)

    def test_load_from_jsonl_skips_malformed_lines(self):
        """load_from_jsonl should silently skip lines that are not valid JSON."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            with open(path, 'w') as f:
                f.write('{"input": "good", "output": "line"}\n')
                f.write('NOT VALID JSON\n')
                f.write('{"input": "also good", "output": "line2"}\n')

            handler = ConversationDataHandler()
            handler.load_from_jsonl(str(path))
            self.assertEqual(len(handler), 2)

    def test_load_from_jsonl_skips_records_without_required_keys(self):
        """load_from_jsonl should skip records that lack 'input' or 'output'."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            with open(path, 'w') as f:
                f.write('{"input": "q", "output": "a"}\n')
                f.write('{"only": "metadata"}\n')

            handler = ConversationDataHandler()
            handler.load_from_jsonl(str(path))
            self.assertEqual(len(handler), 1)

    def test_load_from_jsonl_missing_file_raises(self):
        """load_from_jsonl should raise FileNotFoundError for a missing file."""
        handler = ConversationDataHandler()
        with self.assertRaises(FileNotFoundError):
            handler.load_from_jsonl("/nonexistent/path/data.jsonl")

    def test_format_for_training_from_jsonl(self):
        """Records loaded from JSONL should be formattable for training."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            with open(path, 'w') as f:
                f.write('{"input": "What is Python?", "output": "A programming language."}\n')

            handler = ConversationDataHandler()
            handler.load_from_jsonl(str(path))
            texts = handler.format_for_training()
            self.assertEqual(len(texts), 1)
            self.assertIn("User:", texts[0])
            self.assertIn("Assistant:", texts[0])


class TestDaemonStateHelpers(unittest.TestCase):
    """Tests for daemon state persistence helpers."""

    def test_load_daemon_state_missing_file(self):
        """load_daemon_state returns default state when file is absent."""
        state = load_daemon_state("/nonexistent/path/state.json")
        self.assertEqual(state['lines_consumed'], 0)

    def test_save_and_load_daemon_state(self):
        """State round-trips correctly through save/load."""
        with tempfile.TemporaryDirectory() as tmp:
            state_file = str(Path(tmp) / "state.json")
            state = {'lines_consumed': 42}
            save_daemon_state(state_file, state)
            loaded = load_daemon_state(state_file)
            self.assertEqual(loaded['lines_consumed'], 42)

    def test_load_daemon_state_malformed_json(self):
        """load_daemon_state returns default state when the file is corrupt."""
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            state_file.write_text("not valid json")
            state = load_daemon_state(str(state_file))
            self.assertEqual(state['lines_consumed'], 0)

    def test_count_jsonl_lines_empty_file(self):
        """count_jsonl_lines returns 0 for a file that does not exist."""
        self.assertEqual(count_jsonl_lines("/nonexistent.jsonl"), 0)

    def test_count_jsonl_lines(self):
        """count_jsonl_lines counts non-empty lines."""
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "data.jsonl")
            with open(path, 'w') as f:
                f.write('{"input": "a", "output": "b"}\n')
                f.write('\n')  # blank line — should not be counted
                f.write('{"input": "c", "output": "d"}\n')
            self.assertEqual(count_jsonl_lines(path), 2)


if __name__ == '__main__':
    unittest.main()
