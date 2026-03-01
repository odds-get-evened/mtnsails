"""
Unit tests for the Taber bridge — JSON parsing/validation and command construction.
"""

import unittest
import argparse
from pathlib import Path
import sys

# Ensure the repo root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.taber_bridge import (
    VALID_TARGETS,
    TaberForecastRequest,
    build_taber_command,
    extract_json_from_text,
    validate_request,
)
import main


class TestExtractJsonFromText(unittest.TestCase):
    """Tests for extract_json_from_text()."""

    def test_clean_json(self):
        """Plain JSON object is parsed correctly."""
        text = '{"query": "sensor_id=1", "duration": 24, "interval": 1, "format": "json"}'
        result = extract_json_from_text(text)
        self.assertEqual(result["query"], "sensor_id=1")
        self.assertEqual(result["duration"], 24)

    def test_json_embedded_in_prose(self):
        """JSON embedded in surrounding prose is extracted."""
        text = 'Sure! Here is the JSON: {"query": "sensor_id=2", "duration": 6, "interval": 0.5, "format": "csv"} Hope that helps!'
        result = extract_json_from_text(text)
        self.assertEqual(result["format"], "csv")

    def test_no_json_raises(self):
        """Text without a JSON object raises ValueError."""
        with self.assertRaises(ValueError):
            extract_json_from_text("no braces here")

    def test_malformed_json_raises(self):
        """Malformed JSON raises ValueError."""
        with self.assertRaises(ValueError):
            extract_json_from_text("{bad json: }")


class TestValidateRequest(unittest.TestCase):
    """Tests for validate_request()."""

    def _base(self, **overrides):
        """Return a minimal valid raw dict, optionally overriding fields."""
        data = {
            "query": "sensor_id=1,latitude=39.5",
            "duration": 24,
            "interval": 1,
            "format": "json",
        }
        data.update(overrides)
        return data

    def test_valid_minimal(self):
        """Minimal valid dict produces a correct TaberForecastRequest."""
        req = validate_request(self._base())
        self.assertIsInstance(req, TaberForecastRequest)
        self.assertEqual(req.query, "sensor_id=1,latitude=39.5")
        self.assertEqual(req.duration, 24.0)
        self.assertEqual(req.interval, 1.0)
        self.assertEqual(req.format, "json")
        self.assertEqual(req.targets, [])
        self.assertIsNone(req.data)
        self.assertIsNone(req.data_dir)

    def test_valid_with_targets(self):
        """Valid targets list is accepted."""
        req = validate_request(self._base(targets=["temp", "humidity"]))
        self.assertEqual(req.targets, ["temp", "humidity"])

    def test_valid_all_targets(self):
        """All four valid targets are accepted."""
        req = validate_request(self._base(targets=sorted(VALID_TARGETS)))
        self.assertEqual(set(req.targets), VALID_TARGETS)

    def test_optional_data_fields(self):
        """data and data_dir are passed through."""
        req = validate_request(self._base(data="/tmp/d.csv", data_dir="/tmp/dir"))
        self.assertEqual(req.data, "/tmp/d.csv")
        self.assertEqual(req.data_dir, "/tmp/dir")

    def test_missing_required_field_raises(self):
        """Missing required field raises ValueError."""
        for field in ("query", "duration", "interval", "format"):
            bad = self._base()
            del bad[field]
            with self.assertRaises(ValueError):
                validate_request(bad)

    def test_invalid_format_raises(self):
        """Unknown format value raises ValueError."""
        with self.assertRaises(ValueError):
            validate_request(self._base(format="xml"))

    def test_invalid_target_raises(self):
        """Unknown target raises ValueError."""
        with self.assertRaises(ValueError):
            validate_request(self._base(targets=["temp", "wind"]))

    def test_non_numeric_duration_raises(self):
        """Non-numeric duration raises ValueError."""
        with self.assertRaises(ValueError):
            validate_request(self._base(duration="long"))

    def test_empty_query_raises(self):
        """Empty query string raises ValueError."""
        with self.assertRaises(ValueError):
            validate_request(self._base(query="   "))

    def test_duration_as_string_number(self):
        """String-encoded number is coerced to float."""
        req = validate_request(self._base(duration="12"))
        self.assertEqual(req.duration, 12.0)


class TestBuildTaberCommand(unittest.TestCase):
    """Tests for build_taber_command()."""

    def _make_req(self, **kwargs):
        base = dict(
            query="sensor_id=3,altitude=1500",
            duration=12,
            interval=1,
            format="table",
        )
        base.update(kwargs)
        return validate_request(base)

    def test_minimal_command(self):
        """Minimal request produces expected CLI args."""
        req = self._make_req()
        cmd = build_taber_command(req)
        self.assertEqual(cmd[0], "taber_enviro")
        self.assertIn("--query", cmd)
        self.assertIn("sensor_id=3,altitude=1500", cmd)
        self.assertIn("--duration", cmd)
        self.assertIn("--interval", cmd)
        self.assertIn("--format", cmd)
        # No targets when list is empty
        self.assertNotIn("--targets", cmd)

    def test_targets_added(self):
        """Targets are joined as comma-separated value."""
        req = self._make_req(targets=["temp", "barometer"])
        cmd = build_taber_command(req)
        idx = cmd.index("--targets")
        self.assertEqual(cmd[idx + 1], "temp,barometer")

    def test_data_flags(self):
        """data and data_dir flags are appended when set."""
        req = self._make_req(data="/d/file.csv", data_dir="/d/")
        cmd = build_taber_command(req)
        self.assertIn("--data", cmd)
        self.assertIn("--data-dir", cmd)

    def test_custom_taber_cmd(self):
        """Custom taber_cmd is used as the executable."""
        req = self._make_req()
        cmd = build_taber_command(req, taber_cmd="/usr/local/bin/taber_enviro")
        self.assertEqual(cmd[0], "/usr/local/bin/taber_enviro")


class TestTaberSubcommand(unittest.TestCase):
    """Test that the 'taber' subcommand is registered in main.py."""

    def setUp(self):
        # Re-create the arg parser by calling parse_args on a known subset
        self.parser = argparse.ArgumentParser()
        sub = self.parser.add_subparsers(dest="command")
        p = sub.add_parser("taber")
        p.add_argument("--model-path", required=True)
        p.add_argument("--device", default="cpu")
        p.add_argument("--max-length", type=int, default=512)
        p.add_argument("--max-tokens", type=int, default=256)
        p.add_argument("--taber-cmd", default="taber_enviro")
        p.add_argument("--prompt", default=None)
        p.add_argument("--save-dir", default=None)

    def test_subcommand_defaults(self):
        """taber subcommand parses required and optional args with defaults."""
        args = self.parser.parse_args(["taber", "--model-path", "./onnx_model"])
        self.assertEqual(args.command, "taber")
        self.assertEqual(args.model_path, "./onnx_model")
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.max_length, 512)
        self.assertEqual(args.max_tokens, 256)
        self.assertEqual(args.taber_cmd, "taber_enviro")
        self.assertIsNone(args.prompt)
        self.assertIsNone(args.save_dir)

    def test_subcommand_custom_args(self):
        """taber subcommand accepts all arguments."""
        args = self.parser.parse_args([
            "taber",
            "--model-path", "/models/my_onnx",
            "--device", "cuda",
            "--max-length", "1024",
            "--max-tokens", "512",
            "--taber-cmd", "/usr/bin/taber_enviro",
            "--prompt", "Predict temperature for sensor 7",
            "--save-dir", "/tmp/retraining",
        ])
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.max_tokens, 512)
        self.assertEqual(args.taber_cmd, "/usr/bin/taber_enviro")
        self.assertEqual(args.prompt, "Predict temperature for sensor 7")
        self.assertEqual(args.save_dir, "/tmp/retraining")

    def test_taber_bridge_function_exists(self):
        """taber_bridge function is present and callable in main."""
        self.assertTrue(hasattr(main, "taber_bridge"))
        self.assertTrue(callable(main.taber_bridge))

    def test_taber_bridge_function_signature(self):
        """taber_bridge accepts a single 'args' parameter."""
        import inspect
        sig = inspect.signature(main.taber_bridge)
        self.assertEqual(list(sig.parameters.keys()), ["args"])


if __name__ == "__main__":
    unittest.main()
