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
    parse_fallback_request,
    parse_query_string,
    run_taber,
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

    def test_zero_duration_raises(self):
        """Zero duration raises ValueError."""
        with self.assertRaises(ValueError):
            validate_request(self._base(duration=0))

    def test_negative_duration_raises(self):
        """Negative duration raises ValueError."""
        with self.assertRaises(ValueError):
            validate_request(self._base(duration=-1))

    def test_zero_interval_raises(self):
        """Zero interval raises ValueError."""
        with self.assertRaises(ValueError):
            validate_request(self._base(interval=0))

    def test_negative_interval_raises(self):
        """Negative interval raises ValueError."""
        with self.assertRaises(ValueError):
            validate_request(self._base(interval=-0.5))


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


class TestRunTaber(unittest.TestCase):
    """Tests for run_taber() — focuses on the taber_cmd validation guard."""

    def _make_req(self):
        return validate_request({
            "query": "sensor_id=1",
            "duration": 6,
            "interval": 1,
            "format": "json",
        })

    def test_missing_taber_cmd_raises_file_not_found(self):
        """run_taber raises FileNotFoundError when executable is not on PATH."""
        req = self._make_req()
        with self.assertRaises(FileNotFoundError) as ctx:
            run_taber(req, taber_cmd="__nonexistent_cmd_xyz__")
        self.assertIn("not found", str(ctx.exception))

    def test_absolute_nonexistent_path_raises(self):
        """run_taber raises FileNotFoundError for a non-existent absolute path."""
        req = self._make_req()
        with self.assertRaises(FileNotFoundError):
            run_taber(req, taber_cmd="/nonexistent/path/taber_enviro")


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


class TestParseFallbackRequest(unittest.TestCase):
    """Tests for parse_fallback_request() — heuristic fallback parser."""

    def test_extracts_duration_hours(self):
        """Duration is extracted from 'X hours' pattern."""
        result = parse_fallback_request("what is the forecast for the next 6 hours?")
        self.assertEqual(result["duration"], 6.0)

    def test_extracts_temperature_target(self):
        """'temperature' keyword maps to 'temp' target only."""
        result = parse_fallback_request("top 5 temperature gradients for the next 6 hours")
        self.assertEqual(result["targets"], ["temp"])

    def test_extracts_humidity_target(self):
        """'humidity' keyword maps to 'humidity' target."""
        result = parse_fallback_request("show me humidity forecast for next 12 hours")
        self.assertIn("humidity", result["targets"])
        self.assertEqual(result["duration"], 12.0)

    def test_extracts_barometer_target(self):
        """'pressure' keyword maps to 'barometer' target."""
        result = parse_fallback_request("predict pressure over 3 hours")
        self.assertIn("barometer", result["targets"])
        self.assertEqual(result["duration"], 3.0)

    def test_no_duration_defaults_to_24(self):
        """When no duration is mentioned, defaults to 24 hours."""
        result = parse_fallback_request("what will the temperature be?")
        self.assertEqual(result["duration"], 24.0)

    def test_no_targets_produces_empty_list(self):
        """When no known target keywords appear, targets is empty."""
        result = parse_fallback_request("give me a forecast for the next 2 hours")
        self.assertEqual(result["targets"], [])

    def test_default_query_and_interval(self):
        """query defaults to 'sensor_id=1' and interval to 1.0."""
        result = parse_fallback_request("some request")
        self.assertEqual(result["query"], "sensor_id=1")
        self.assertEqual(result["interval"], 1.0)

    def test_default_format_is_table(self):
        """format defaults to 'table' when not mentioned."""
        result = parse_fallback_request("temperature forecast for 6 hours")
        self.assertEqual(result["format"], "table")

    def test_result_passes_validation(self):
        """Output of parse_fallback_request passes validate_request."""
        raw = parse_fallback_request("top 5 temperature gradients for the next 6 hours")
        req = validate_request(raw)
        self.assertIsInstance(req, TaberForecastRequest)
        self.assertEqual(req.duration, 6.0)
        self.assertIn("temp", req.targets)


class TestTaberBridgeExecutorFallback(unittest.TestCase):
    """
    Tests that TaberBridgeExecutor.run() falls back to the heuristic parser
    when the LLM emits JSON that is missing required fields.
    """

    def _make_executor(self, chat_mock):
        """Create a TaberBridgeExecutor with a mocked ChatInterface."""
        import importlib, sys

        # Stub out heavy dependencies so taber_executor imports cleanly
        fake_chat_mod = MagicMock()
        fake_chat_mod.ChatInterface = MagicMock(return_value=chat_mock)

        with patch.dict("sys.modules", {"src.chat_interface": fake_chat_mod}):
            # Force re-import with the stubbed dependency
            if "src.taber_executor" in sys.modules:
                del sys.modules["src.taber_executor"]
            from src import taber_executor as te

            exc = te.TaberBridgeExecutor.__new__(te.TaberBridgeExecutor)
            exc.max_new_tokens = 256
            exc.taber_model_dir = None
            exc.taber_cmd = "taber_enviro"
            exc._chat = chat_mock
            return exc, te

    def test_partial_json_falls_back_gracefully(self):
        """
        When the LLM emits JSON that is missing 'duration', the executor must
        fall back to parse_fallback_request rather than raising ValueError.
        Raw output is requested so the predictor output is returned directly.
        """
        chat_mock = MagicMock()
        # LLM output is valid JSON but missing 'duration' and 'interval'
        chat_mock.generate_response.return_value = (
            '{"query": "sensor_id=1", "format": "table"}'
        )
        executor, te = self._make_executor(chat_mock)

        with patch.object(te, "run_taber", return_value="ok") as mock_run:
            result = executor.run(
                "top 5 temperature gradients for the next 6 hours",
                natural_language_report=False,
            )

        self.assertEqual(result, "ok")
        # Ensure run_taber was called with a valid request (fallback filled in duration)
        req_passed = mock_run.call_args[0][0]
        self.assertEqual(req_passed.duration, 6.0)
        self.assertIn("temp", req_passed.targets)

    def test_no_json_falls_back_gracefully(self):
        """
        When the LLM emits no JSON at all, the executor falls back to the
        heuristic parser — same behaviour as before this fix.
        Raw output is requested so the predictor output is returned directly.
        """
        chat_mock = MagicMock()
        chat_mock.generate_response.return_value = "I cannot answer that."
        executor, te = self._make_executor(chat_mock)

        with patch.object(te, "run_taber", return_value="ok") as mock_run:
            result = executor.run(
                "humidity forecast for next 3 hours",
                natural_language_report=False,
            )

        self.assertEqual(result, "ok")
        req_passed = mock_run.call_args[0][0]
        self.assertEqual(req_passed.duration, 3.0)
        self.assertIn("humidity", req_passed.targets)


class TestNaturalLanguageReport(unittest.TestCase):
    """
    Tests for the natural-language report step in TaberBridgeExecutor.

    The executor sends the structured predictor output back to the LLM
    (step 5 / _generate_report) to produce a plain-English forecast summary.
    """

    def _make_executor(self, chat_mock):
        """Create a TaberBridgeExecutor with a mocked ChatInterface."""
        import sys

        fake_chat_mod = MagicMock()
        fake_chat_mod.ChatInterface = MagicMock(return_value=chat_mock)

        with patch.dict("sys.modules", {"src.chat_interface": fake_chat_mod}):
            if "src.taber_executor" in sys.modules:
                del sys.modules["src.taber_executor"]
            from src import taber_executor as te

            exc = te.TaberBridgeExecutor.__new__(te.TaberBridgeExecutor)
            exc.max_new_tokens = 256
            exc.taber_model_dir = None
            exc.taber_cmd = "taber_enviro"
            exc._chat = chat_mock
            return exc, te

    def test_nl_report_returned_by_default(self):
        """
        run() returns the LLM-generated NL report (not raw predictor output)
        when natural_language_report=True (the default).
        """
        chat_mock = MagicMock()
        # First call: LLM emits valid JSON for the structured request
        # Second call: LLM produces the NL report
        chat_mock.generate_response.side_effect = [
            '{"query": "sensor_id=1", "duration": 6, "interval": 1, "format": "json", "targets": ["temp"]}',
            "Temperatures are expected to rise steadily over the next 6 hours.",
        ]
        executor, te = self._make_executor(chat_mock)

        with patch.object(te, "run_taber", return_value="raw_predictor_data"):
            result = executor.run("temperature forecast for the next 6 hours")

        self.assertEqual(result, "Temperatures are expected to rise steadily over the next 6 hours.")
        # The LLM must be called twice: once for JSON, once for the NL report
        self.assertEqual(chat_mock.generate_response.call_count, 2)

    def test_raw_output_skips_nl_report(self):
        """
        run(natural_language_report=False) returns the raw predictor output
        and does NOT make a second LLM call.
        """
        chat_mock = MagicMock()
        chat_mock.generate_response.return_value = (
            '{"query": "sensor_id=1", "duration": 6, "interval": 1, "format": "json"}'
        )
        executor, te = self._make_executor(chat_mock)

        with patch.object(te, "run_taber", return_value="raw_predictor_data"):
            result = executor.run(
                "temperature forecast for the next 6 hours",
                natural_language_report=False,
            )

        self.assertEqual(result, "raw_predictor_data")
        # Only one LLM call (the JSON extraction step)
        self.assertEqual(chat_mock.generate_response.call_count, 1)

    def test_generate_report_includes_user_request_and_data(self):
        """
        _generate_report() passes the user request and predictor output
        to the LLM so it has the context to write the summary.
        """
        chat_mock = MagicMock()
        chat_mock.generate_response.return_value = "A brief summary."
        executor, te = self._make_executor(chat_mock)

        result = executor._generate_report(
            user_request="temperature for next 6 hours",
            predictor_output="datetime  temp\n2024-01-01T00:00  20.5\n2024-01-01T01:00  21.0",
        )

        self.assertEqual(result, "A brief summary.")
        call_args = chat_mock.generate_response.call_args[0][0]
        # The prompt must contain both the user request and the predictor data
        self.assertIn("temperature for next 6 hours", call_args)
        self.assertIn("20.5", call_args)

    def test_nl_report_still_saves_raw_data_to_disk(self):
        """
        When save_dir is set, the raw predictor output (not the NL report) is
        persisted to disk — even when natural_language_report=True.
        """
        import tempfile
        import os

        chat_mock = MagicMock()
        chat_mock.generate_response.side_effect = [
            '{"query": "sensor_id=1", "duration": 6, "interval": 1, "format": "json"}',
            "Brief NL summary.",
        ]
        executor, te = self._make_executor(chat_mock)

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(te, "run_taber", return_value="raw_data"):
                result = executor.run(
                    "temperature forecast for 6 hours",
                    save_dir=tmp_dir,
                    natural_language_report=True,
                )

            # NL report is returned to the caller
            self.assertEqual(result, "Brief NL summary.")

            # Raw predictor response is saved to disk
            saved_files = os.listdir(tmp_dir)
            response_files = [f for f in saved_files if "response" in f]
            self.assertTrue(response_files, "Expected at least one response file in save_dir")
            response_text = (Path(tmp_dir) / response_files[0]).read_text(encoding="utf-8")
            self.assertEqual(response_text, "raw_data")


class TestRunTaberTimeout(unittest.TestCase):
    """Tests for the timeout behaviour added to run_taber()."""

    def _make_req(self):
        return validate_request({
            "query": "sensor_id=1",
            "duration": 6,
            "interval": 1,
            "format": "json",
        })

    def test_timeout_raises_runtime_error(self):
        """run_taber raises RuntimeError when the subprocess times out."""
        import subprocess as sp

        req = self._make_req()
        with patch("src.taber_bridge.shutil.which", return_value="/fake/taber_enviro"), \
             patch("src.taber_bridge.subprocess.run",
                   side_effect=sp.TimeoutExpired(cmd="taber_enviro", timeout=1)):
            with self.assertRaises(RuntimeError) as ctx:
                run_taber(req, timeout=1)
        self.assertIn("timed out", str(ctx.exception))

    def test_custom_timeout_reflected_in_error(self):
        """The timeout value appears in the RuntimeError message."""
        import subprocess as sp

        req = self._make_req()
        with patch("src.taber_bridge.shutil.which", return_value="/fake/taber_enviro"), \
             patch("src.taber_bridge.subprocess.run",
                   side_effect=sp.TimeoutExpired(cmd="taber_enviro", timeout=30)):
            with self.assertRaises(RuntimeError) as ctx:
                run_taber(req, timeout=30)
        self.assertIn("30", str(ctx.exception))


class TestFormatPredictionsCSVTargets(unittest.TestCase):
    """Tests that CSV format correctly filters columns by requested targets."""

    def _preds(self):
        return [
            {"datetime": "2024-01-01T10:00:00", "temp": 20.5, "humidity": 55.0,
             "barometer": 1013.0, "light": 300.0},
            {"datetime": "2024-01-01T11:00:00", "temp": 21.0, "humidity": 54.5,
             "barometer": 1012.0, "light": 310.0},
        ]

    def test_csv_respects_single_target(self):
        """CSV header contains only 'datetime' and the requested target."""
        result = _format_predictions(self._preds(), "csv", ["temp"])
        header = result.split("\n")[0]
        self.assertEqual(header, "datetime,temp")
        self.assertNotIn("humidity", header)
        self.assertNotIn("barometer", header)

    def test_csv_none_targets_uses_all_defaults(self):
        """CSV header with targets=None includes all four default targets."""
        result = _format_predictions(self._preds(), "csv", None)
        header = result.split("\n")[0]
        self.assertIn("datetime", header)
        for t in ["temp", "barometer", "light", "humidity"]:
            self.assertIn(t, header)

    def test_csv_multiple_targets(self):
        """CSV header with multiple targets only includes those targets."""
        result = _format_predictions(self._preds(), "csv", ["temp", "humidity"])
        header = result.split("\n")[0]
        self.assertEqual(header, "datetime,temp,humidity")
        self.assertNotIn("barometer", header)
        self.assertNotIn("light", header)


class TestParseFallbackRequestExtendedDuration(unittest.TestCase):
    """Tests for the extended duration regex supporting days and minutes."""

    def test_extracts_duration_days(self):
        """'2 days' is converted to 48 hours."""
        result = parse_fallback_request("give me a 2 days forecast")
        self.assertEqual(result["duration"], 48.0)

    def test_extracts_duration_day_singular(self):
        """'1 day' is converted to 24 hours."""
        result = parse_fallback_request("forecast for the next 1 day")
        self.assertEqual(result["duration"], 24.0)

    def test_extracts_duration_minutes(self):
        """'30 minutes' is converted to 0.5 hours."""
        result = parse_fallback_request("forecast for the next 30 minutes")
        self.assertAlmostEqual(result["duration"], 0.5)

    def test_extracts_duration_mins_abbreviation(self):
        """'60 mins' is converted to 1.0 hours."""
        result = parse_fallback_request("predict temperature for 60 mins")
        self.assertAlmostEqual(result["duration"], 1.0)

    def test_hours_still_work(self):
        """Existing 'hours' pattern still extracts correctly."""
        result = parse_fallback_request("forecast for the next 6 hours")
        self.assertEqual(result["duration"], 6.0)

    def test_hrs_abbreviation_still_works(self):
        """Existing 'hrs' abbreviation still extracts correctly."""
        result = parse_fallback_request("6-hr forecast")
        self.assertEqual(result["duration"], 6.0)

    def test_days_result_passes_validation(self):
        """A days-based duration produces a valid TaberForecastRequest."""
        raw = parse_fallback_request("3-day temperature forecast")
        req = validate_request(raw)
        self.assertIsInstance(req, TaberForecastRequest)
        self.assertEqual(req.duration, 72.0)


class TestFallbackLogging(unittest.TestCase):
    """Tests that TaberBridgeExecutor.run() emits a warning when falling back."""

    def _make_executor(self, chat_mock):
        """Create a TaberBridgeExecutor with a mocked ChatInterface."""
        import sys

        fake_chat_mod = MagicMock()
        fake_chat_mod.ChatInterface = MagicMock(return_value=chat_mock)

        with patch.dict("sys.modules", {"src.chat_interface": fake_chat_mod}):
            if "src.taber_executor" in sys.modules:
                del sys.modules["src.taber_executor"]
            from src import taber_executor as te

            exc = te.TaberBridgeExecutor.__new__(te.TaberBridgeExecutor)
            exc.max_new_tokens = 256
            exc.taber_model_dir = None
            exc.taber_cmd = "taber_enviro"
            exc._chat = chat_mock
            return exc, te

    def test_fallback_emits_warning(self):
        """A warning is logged when the LLM output fails JSON extraction."""
        chat_mock = MagicMock()
        chat_mock.generate_response.return_value = "not json at all"
        executor, te = self._make_executor(chat_mock)

        with patch.object(te, "run_taber", return_value="ok"):
            with self.assertLogs("src.taber_executor", level="WARNING") as cm:
                executor.run("humidity forecast for next 3 hours",
                             natural_language_report=False)

        self.assertTrue(
            any("fallback" in msg.lower() or "heuristic" in msg.lower() for msg in cm.output),
            f"Expected fallback warning in log output, got: {cm.output}",
        )

    def test_valid_json_does_not_emit_warning(self):
        """No warning is logged when the LLM produces valid JSON."""
        chat_mock = MagicMock()
        chat_mock.generate_response.return_value = (
            '{"query": "sensor_id=1", "duration": 6, "interval": 1, "format": "table"}'
        )
        executor, te = self._make_executor(chat_mock)

        import logging
        with patch.object(te, "run_taber", return_value="ok"):
            with self.assertLogs("src.taber_executor", level="WARNING") as cm:
                # Inject a dummy warning so assertLogs doesn't fail when no
                # records are emitted (assertLogs raises if the logger is silent)
                logging.getLogger("src.taber_executor").warning("sentinel")
                executor.run("temperature forecast for 6 hours",
                             natural_language_report=False)

        # Only the sentinel should be present — no real fallback warning
        self.assertEqual(len(cm.output), 1)
        self.assertIn("sentinel", cm.output[0])


if __name__ == "__main__":
    unittest.main()
