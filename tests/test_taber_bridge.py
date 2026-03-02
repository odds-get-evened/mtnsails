"""
Unit tests for the Taber bridge — JSON parsing/validation and command construction.
"""

import unittest
import argparse
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

# Ensure the repo root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.taber_bridge import (
    VALID_TARGETS,
    TaberForecastRequest,
    _format_predictions,
    build_taber_command,
    extract_json_from_text,
    parse_query_string,
    run_taber_python,
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

    def test_single_quoted_json(self):
        """Single-quoted Python-style dict is parsed correctly."""
        text = "{'query': 'sensor_id=1', 'duration': 24, 'interval': 1, 'format': 'table'}"
        result = extract_json_from_text(text)
        self.assertEqual(result["query"], "sensor_id=1")
        self.assertEqual(result["duration"], 24)
        self.assertEqual(result["format"], "table")

    def test_single_quoted_with_list(self):
        """Single-quoted dict containing a list is parsed correctly."""
        text = "{'query': 'sensor_id=1', 'duration': 24, 'interval': 1, 'format': 'json', 'targets': ['temp', 'humidity']}"
        result = extract_json_from_text(text)
        self.assertEqual(result["targets"], ["temp", "humidity"])

    def test_python_none_literal(self):
        """Python None is handled via ast.literal_eval fallback."""
        text = "{'query': 'sensor_id=1', 'duration': 24, 'interval': 1, 'format': 'json', 'data': None}"
        result = extract_json_from_text(text)
        self.assertIsNone(result["data"])

    def test_trailing_comma(self):
        """Trailing comma before closing brace is repaired."""
        text = '{"query": "sensor_id=1", "duration": 24, "interval": 1, "format": "json",}'
        result = extract_json_from_text(text)
        self.assertEqual(result["query"], "sensor_id=1")

    def test_single_quoted_embedded_in_prose(self):
        """Single-quoted dict embedded in LLM prose is extracted."""
        text = "Here is my answer: {'query': 'sensor_id=5', 'duration': 12, 'interval': 1, 'format': 'csv'} Done."
        result = extract_json_from_text(text)
        self.assertEqual(result["query"], "sensor_id=5")
        self.assertEqual(result["format"], "csv")


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
        for field in ("duration", "interval", "format"):
            bad = self._base()
            del bad[field]
            with self.assertRaises(ValueError):
                validate_request(bad)

    def test_missing_query_defaults_to_sensor_id_1(self):
        """Missing 'query' field uses 'sensor_id=1' as the default."""
        raw = self._base()
        del raw["query"]
        req = validate_request(raw)
        self.assertEqual(req.query, "sensor_id=1")

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

    def test_taber_model_dir_argument(self):
        """taber subcommand accepts --taber-model-dir."""
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        p = sub.add_parser("taber")
        p.add_argument("--model-path", required=True)
        p.add_argument("--device", default="cpu")
        p.add_argument("--max-length", type=int, default=512)
        p.add_argument("--max-tokens", type=int, default=256)
        p.add_argument("--taber-model-dir", default=None)
        p.add_argument("--taber-cmd", default="taber_enviro")
        p.add_argument("--prompt", default=None)
        p.add_argument("--save-dir", default=None)

        args = parser.parse_args([
            "taber",
            "--model-path", "./onnx_model",
            "--taber-model-dir", "/models/taber",
        ])
        self.assertEqual(args.taber_model_dir, "/models/taber")
        self.assertIsNone(args.prompt)

    def test_taber_model_dir_default_is_none(self):
        """--taber-model-dir defaults to None (subprocess fallback)."""
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        p = sub.add_parser("taber")
        p.add_argument("--model-path", required=True)
        p.add_argument("--taber-model-dir", default=None)
        p.add_argument("--taber-cmd", default="taber_enviro")

        args = parser.parse_args(["taber", "--model-path", "./onnx"])
        self.assertIsNone(args.taber_model_dir)
        self.assertEqual(args.taber_cmd, "taber_enviro")


class TestParseQueryString(unittest.TestCase):
    """Tests for parse_query_string()."""

    def test_numeric_values(self):
        """Numeric values are coerced to float."""
        result = parse_query_string("sensor_id=1,latitude=40.0,longitude=-105.5")
        self.assertEqual(result["sensor_id"], 1.0)
        self.assertEqual(result["latitude"], 40.0)
        self.assertEqual(result["longitude"], -105.5)

    def test_string_value_kept(self):
        """Non-numeric values remain as strings."""
        result = parse_query_string("sensor_id=ABC,latitude=39.5")
        self.assertEqual(result["sensor_id"], "ABC")
        self.assertEqual(result["latitude"], 39.5)

    def test_single_pair(self):
        """Single key=value pair is parsed correctly."""
        result = parse_query_string("sensor_id=7")
        self.assertEqual(result, {"sensor_id": 7.0})

    def test_invalid_pair_raises(self):
        """Pair without '=' raises ValueError."""
        with self.assertRaises(ValueError):
            parse_query_string("sensor_id=1,bad_pair,latitude=40.0")

    def test_altitude(self):
        """Altitude is parsed as float."""
        result = parse_query_string("sensor_id=3,altitude=2800")
        self.assertEqual(result["altitude"], 2800.0)


class TestRunTaberPython(unittest.TestCase):
    """Tests for run_taber_python() — mocks ONNXPredictor to avoid real models."""

    def _make_req(self, fmt="json", targets=None):
        return validate_request({
            "query": "sensor_id=1,latitude=40.0",
            "duration": 1,
            "interval": 0.5,
            "format": fmt,
            "targets": targets or [],
        })

    def _fake_predictions(self):
        return [
            {"sensor_id": 1, "timestamp": 1000, "datetime": "2024-01-01T00:00:00", "temp": 20.5, "humidity": 55.0},
            {"sensor_id": 1, "timestamp": 2800, "datetime": "2024-01-01T00:30:00", "temp": 21.0, "humidity": 54.5},
        ]

    def test_uses_onnx_predictor_in_process(self):
        """run_taber_python() creates ONNXPredictor and calls forecast()."""
        mock_predictor = MagicMock()
        mock_predictor.forecast.return_value = self._fake_predictions()
        mock_onnx_cls = MagicMock(return_value=mock_predictor)
        mock_module = MagicMock()
        mock_module.ONNXPredictor = mock_onnx_cls

        req = self._make_req(fmt="json")
        with patch.dict("sys.modules", {"pipeline": MagicMock(), "pipeline.predictor": mock_module}):
            result = run_taber_python(req, "/fake/model/dir")

        # ONNXPredictor was instantiated with the model dir
        mock_onnx_cls.assert_called_once_with("/fake/model/dir", data_dir=None)
        # load_data was called (no explicit data file)
        mock_predictor.load_data.assert_called_once_with(data_file=None)
        # forecast was called with converted duration/interval
        mock_predictor.forecast.assert_called_once_with(
            query={"sensor_id": 1.0, "latitude": 40.0},
            duration_minutes=60,   # 1 hour × 60
            interval_seconds=1800, # 0.5 hours × 3600
            targets=None,
        )
        # Result is valid JSON
        import json
        parsed = json.loads(result)
        self.assertEqual(len(parsed), 2)

    def test_passes_data_file_to_load_data(self):
        """data and data_dir from the request are forwarded to ONNXPredictor."""
        mock_predictor = MagicMock()
        mock_predictor.forecast.return_value = self._fake_predictions()
        mock_onnx_cls = MagicMock(return_value=mock_predictor)
        mock_module = MagicMock()
        mock_module.ONNXPredictor = mock_onnx_cls

        req = validate_request({
            "query": "sensor_id=2",
            "duration": 2,
            "interval": 1,
            "format": "json",
            "data": "/tmp/data.jsonl",
            "data_dir": "/tmp/datadir",
        })
        with patch.dict("sys.modules", {"pipeline": MagicMock(), "pipeline.predictor": mock_module}):
            run_taber_python(req, "/model/dir")

        mock_onnx_cls.assert_called_once_with("/model/dir", data_dir="/tmp/datadir")
        mock_predictor.load_data.assert_called_once_with(data_file="/tmp/data.jsonl")

    def test_missing_package_raises_runtime_error(self):
        """RuntimeError is raised when taber_enviro package is not installed."""
        import sys
        # Remove pipeline from sys.modules to simulate it not being installed
        original = sys.modules.pop("pipeline", None)
        original_pred = sys.modules.pop("pipeline.predictor", None)
        try:
            req = self._make_req()
            with self.assertRaises(RuntimeError) as ctx:
                run_taber_python(req, "/model/dir")
            self.assertIn("taber_enviro", str(ctx.exception))
        finally:
            if original is not None:
                sys.modules["pipeline"] = original
            if original_pred is not None:
                sys.modules["pipeline.predictor"] = original_pred


class TestFormatPredictions(unittest.TestCase):
    """Tests for _format_predictions() helper."""

    def _preds(self):
        return [
            {"sensor_id": 1, "timestamp": 1000, "datetime": "2024-01-01T10:00:00", "temp": 20.5, "humidity": 55.0},
            {"sensor_id": 1, "timestamp": 2800, "datetime": "2024-01-01T10:30:00", "temp": 21.0, "humidity": 54.5},
        ]

    def test_json_format(self):
        """JSON format returns valid JSON list."""
        import json
        result = _format_predictions(self._preds(), "json", ["temp", "humidity"])
        parsed = json.loads(result)
        self.assertEqual(len(parsed), 2)
        self.assertAlmostEqual(parsed[0]["temp"], 20.5)

    def test_csv_format(self):
        """CSV format includes header and data rows."""
        result = _format_predictions(self._preds(), "csv", ["temp"])
        lines = result.strip().split("\n")
        self.assertGreater(len(lines), 1)
        self.assertIn("temp", lines[0])

    def test_table_format(self):
        """Table format includes a header separator line."""
        result = _format_predictions(self._preds(), "table", ["temp"])
        self.assertIn("-", result)
        self.assertIn("datetime", result)

    def test_empty_predictions_json(self):
        """Empty predictions return empty JSON list."""
        result = _format_predictions([], "json", None)
        self.assertEqual(result, "[]")

    def test_empty_predictions_csv(self):
        """Empty predictions return empty string for CSV."""
        result = _format_predictions([], "csv", None)
        self.assertEqual(result, "")

    def test_empty_predictions_table(self):
        """Empty predictions return empty string for table."""
        result = _format_predictions([], "table", None)
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
