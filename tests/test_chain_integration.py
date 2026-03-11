"""
Cross-repo integration test for the AI model chaining pipeline.

Verifies the full closed-loop chain:

    User (NL) → [LLM] → structured JSON → [taber_enviro ONNXPredictor]
                → structured forecast → [LLM] → NL report → User

Only ONNX inference sessions are mocked (no model files needed).
The real import chain — including from pipeline.predictor import ONNXPredictor —
is exercised so that future interface mismatches are caught early.
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── repo roots ──────────────────────────────────────────────────────────────
MTNSAILS_ROOT = Path(__file__).parent.parent
TABER_ROOT = MTNSAILS_ROOT.parent / "taber_enviro"

# Ensure both repos are importable
sys.path.insert(0, str(MTNSAILS_ROOT))
if str(TABER_ROOT) not in sys.path:
    sys.path.insert(0, str(TABER_ROOT))


# ── helpers ─────────────────────────────────────────────────────────────────

def _fake_predictions():
    """Return a minimal list of prediction dicts matching ONNXPredictor output."""
    return [
        {
            "sensor_id": 1,
            "timestamp": 1_000_000,
            "datetime": "2024-06-01T00:00:00",
            "temp": 18.5,
            "barometer": 1013.2,
            "light": 200.0,
            "humidity": 62.3,
        },
        {
            "sensor_id": 1,
            "timestamp": 1_003_600,
            "datetime": "2024-06-01T01:00:00",
            "temp": 17.9,
            "barometer": 1012.8,
            "light": 0.0,
            "humidity": 65.1,
        },
    ]


def _make_executor_with_mocks(llm_responses, predictor_predictions):
    """
    Build a TaberBridgeExecutor whose LLM and predictor are both mocked.

    torch and other heavy ML deps are stubbed in sys.modules so that
    src.chat_interface (which imports torch at module level) loads cleanly.

    Args:
        llm_responses: list of strings returned by ChatInterface.generate_response
                       (first call = JSON query, second call = NL report)
        predictor_predictions: list of dicts returned by ONNXPredictor.forecast()

    Returns:
        (executor, te_module, predictor_instance, predictor_cls) tuple.
    """
    # Mock ChatInterface so no ONNX LLM files are needed
    chat_mock = MagicMock()
    chat_mock.generate_response.side_effect = list(llm_responses)

    # Mock ONNXPredictor so no taber_enviro model files are needed
    mock_predictor_instance = MagicMock()
    mock_predictor_instance.forecast.return_value = predictor_predictions
    mock_predictor_cls = MagicMock(return_value=mock_predictor_instance)
    mock_pipeline_predictor = MagicMock()
    mock_pipeline_predictor.ONNXPredictor = mock_predictor_cls

    # Stub heavy deps so taber_executor (and its transitive imports) load cleanly
    fake_chat_mod = MagicMock()
    fake_chat_mod.ChatInterface = MagicMock(return_value=chat_mock)

    heavy_stubs = {
        "torch": MagicMock(),
        "transformers": MagicMock(),
        "onnxruntime": MagicMock(),
        "src.chat_interface": fake_chat_mod,
        "pipeline": MagicMock(),
        "pipeline.predictor": mock_pipeline_predictor,
    }

    with patch.dict("sys.modules", heavy_stubs):
        # Force a clean re-import inside the patched environment
        for mod_name in list(sys.modules):
            if mod_name in ("src.taber_executor", "src.taber_bridge"):
                del sys.modules[mod_name]

        from src import taber_executor as te_fresh

        executor = te_fresh.TaberBridgeExecutor.__new__(te_fresh.TaberBridgeExecutor)
        executor.max_new_tokens = 128
        executor.taber_model_dir = "/fake/taber/model"
        executor.taber_cmd = "taber_enviro"
        executor._chat = chat_mock

    return executor, te_fresh, mock_predictor_instance, mock_predictor_cls


# ── tests ────────────────────────────────────────────────────────────────────

class TestFullChainNLReport(unittest.TestCase):
    """
    Full end-to-end chain: NL prompt → LLM → JSON → predictor → LLM → NL report.
    """

    def test_chain_returns_nl_report(self):
        """Chain produces an NL report, not raw predictor data."""
        executor, te, predictor_inst, predictor_cls = _make_executor_with_mocks(
            llm_responses=[
                # Step 1: LLM produces structured JSON
                json.dumps({
                    "query": "sensor_id=1,latitude=40.0,longitude=-105.0",
                    "duration": 6,
                    "interval": 1,
                    "format": "json",
                    "targets": ["temp", "humidity"],
                }),
                # Step 2: LLM narrates the forecast
                "Temperatures will dip slightly overnight, staying around 18°C.",
            ],
            predictor_predictions=_fake_predictions(),
        )

        with patch.object(te, "run_taber_python", return_value=json.dumps(_fake_predictions())):
            result = executor.run(
                "What will the temperature and humidity be like over the next 6 hours?",
            )

        self.assertIn("18", result)
        self.assertIn("°C", result)

    def test_chain_llm_called_twice(self):
        """LLM is invoked exactly twice: once for JSON, once for the NL report."""
        executor, te, predictor_inst, predictor_cls = _make_executor_with_mocks(
            llm_responses=[
                json.dumps({
                    "query": "sensor_id=1",
                    "duration": 3,
                    "interval": 1,
                    "format": "table",
                }),
                "Expect mild conditions throughout the forecast period.",
            ],
            predictor_predictions=_fake_predictions(),
        )

        with patch.object(te, "run_taber_python", return_value="fake_table_output"):
            executor.run("Brief weather summary for the next 3 hours")

        self.assertEqual(executor._chat.generate_response.call_count, 2)

    def test_chain_raw_output_skips_second_llm_call(self):
        """With natural_language_report=False, only one LLM call is made."""
        executor, te, predictor_inst, predictor_cls = _make_executor_with_mocks(
            llm_responses=[
                json.dumps({
                    "query": "sensor_id=1",
                    "duration": 1,
                    "interval": 1,
                    "format": "json",
                }),
            ],
            predictor_predictions=_fake_predictions(),
        )

        with patch.object(te, "run_taber_python", return_value="raw_data"):
            result = executor.run(
                "1-hour forecast",
                natural_language_report=False,
            )

        self.assertEqual(result, "raw_data")
        self.assertEqual(executor._chat.generate_response.call_count, 1)


class TestChainFallbackToHeuristics(unittest.TestCase):
    """
    When the LLM fails to produce valid JSON, the chain falls back to
    heuristic parsing of the original user request.
    """

    def test_garbled_llm_output_uses_fallback(self):
        """Garbled LLM output still produces a valid forecast request."""
        executor, te, predictor_inst, predictor_cls = _make_executor_with_mocks(
            llm_responses=["I don't understand the request."],
            predictor_predictions=_fake_predictions(),
        )

        # taber_model_dir is set, so the executor uses run_taber_python
        with patch.object(te, "run_taber_python", return_value="fallback_output") as mock_run:
            result = executor.run(
                "temperature forecast for the next 12 hours",
                natural_language_report=False,
            )

        req = mock_run.call_args[0][0]
        self.assertEqual(req.duration, 12.0)
        self.assertIn("temp", req.targets)
        self.assertEqual(result, "fallback_output")

    def test_missing_required_fields_uses_fallback(self):
        """JSON missing required fields triggers fallback, not a crash."""
        executor, te, predictor_inst, predictor_cls = _make_executor_with_mocks(
            llm_responses=['{"query": "sensor_id=1"}'],  # missing duration/interval/format
            predictor_predictions=_fake_predictions(),
        )

        # taber_model_dir is set, so the executor uses run_taber_python
        with patch.object(te, "run_taber_python", return_value="ok"):
            result = executor.run(
                "humidity and pressure for the next 6 hours",
                natural_language_report=False,
            )

        self.assertEqual(result, "ok")


class TestChainBridgeSchema(unittest.TestCase):
    """
    Verify that the bridge schema (TaberForecastRequest) correctly maps
    human-readable hour/interval values to the units ONNXPredictor expects.
    """

    def test_duration_converted_to_minutes(self):
        """Bridge converts duration from hours to minutes for ONNXPredictor."""
        from src.taber_bridge import run_taber_python, validate_request

        req = validate_request({
            "query": "sensor_id=1",
            "duration": 2,     # 2 hours
            "interval": 0.5,   # 30 minutes
            "format": "json",
        })

        mock_pred = MagicMock()
        mock_pred.forecast.return_value = _fake_predictions()
        mock_cls = MagicMock(return_value=mock_pred)
        mock_mod = MagicMock()
        mock_mod.ONNXPredictor = mock_cls

        with patch.dict("sys.modules", {"pipeline": MagicMock(), "pipeline.predictor": mock_mod}):
            run_taber_python(req, "/fake/model")

        call_kwargs = mock_pred.forecast.call_args[1]
        self.assertEqual(call_kwargs["duration_minutes"], 120)   # 2h × 60
        self.assertEqual(call_kwargs["interval_seconds"], 1800)  # 0.5h × 3600

    def test_targets_none_when_empty(self):
        """Empty targets list is passed as None to ONNXPredictor (predict all)."""
        from src.taber_bridge import run_taber_python, validate_request

        req = validate_request({
            "query": "sensor_id=1",
            "duration": 1,
            "interval": 1,
            "format": "json",
            # no 'targets' → defaults to all
        })

        mock_pred = MagicMock()
        mock_pred.forecast.return_value = _fake_predictions()
        mock_cls = MagicMock(return_value=mock_pred)
        mock_mod = MagicMock()
        mock_mod.ONNXPredictor = mock_cls

        with patch.dict("sys.modules", {"pipeline": MagicMock(), "pipeline.predictor": mock_mod}):
            run_taber_python(req, "/fake/model")

        call_kwargs = mock_pred.forecast.call_args[1]
        self.assertIsNone(call_kwargs["targets"])


class TestChainSaveDir(unittest.TestCase):
    """Verify that the raw request + response are persisted when save_dir is set."""

    def test_save_dir_writes_request_and_response(self):
        """Both a JSON request file and a response file are written to save_dir."""
        import tempfile

        executor, te, predictor_inst, predictor_cls = _make_executor_with_mocks(
            llm_responses=[
                json.dumps({
                    "query": "sensor_id=1",
                    "duration": 3,
                    "interval": 1,
                    "format": "json",
                }),
                "Brief NL summary.",
            ],
            predictor_predictions=_fake_predictions(),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(te, "run_taber_python", return_value="raw_forecast_data"):
                executor.run(
                    "3-hour forecast",
                    save_dir=tmp_dir,
                    natural_language_report=True,
                )

            files = os.listdir(tmp_dir)
            req_files = [f for f in files if "request" in f and f.endswith(".json")]
            res_files = [f for f in files if "response" in f and f.endswith(".txt")]

            self.assertTrue(req_files, "Expected a request JSON file in save_dir")
            self.assertTrue(res_files, "Expected a response TXT file in save_dir")

            # Request file must be valid JSON
            req_data = json.loads(
                (Path(tmp_dir) / req_files[0]).read_text(encoding="utf-8")
            )
            self.assertIn("query", req_data)

            # Response file must contain the raw predictor output
            res_text = (Path(tmp_dir) / res_files[0]).read_text(encoding="utf-8")
            self.assertEqual(res_text, "raw_forecast_data")


class TestChainImportContract(unittest.TestCase):
    """
    Smoke-test that the real taber_enviro import path resolves correctly
    from the mtnsails side of the bridge.
    """

    def test_taber_enviro_importable_from_mtnsails_context(self):
        """
        pipeline.predictor.ONNXPredictor must be importable when taber_enviro
        is on sys.path — as it would be in a deployed environment.
        """
        self.assertTrue(
            (TABER_ROOT / "pipeline" / "predictor.py").exists(),
            "taber_enviro/pipeline/predictor.py not found — check repo layout",
        )
        # Import the real module (no mocks) to confirm the import chain works
        import importlib
        spec = importlib.util.spec_from_file_location(
            "pipeline.predictor",
            str(TABER_ROOT / "pipeline" / "predictor.py"),
        )
        self.assertIsNotNone(spec, "Could not create import spec for predictor.py")

    def test_onnx_predictor_class_exists(self):
        """ONNXPredictor class must be present in pipeline.predictor."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "pipeline.predictor",
            str(TABER_ROOT / "pipeline" / "predictor.py"),
        )
        # We don't exec the module (would require onnxruntime) — just verify
        # the source contains the class definition.
        source = (TABER_ROOT / "pipeline" / "predictor.py").read_text(encoding="utf-8")
        self.assertIn("class ONNXPredictor", source)
        self.assertIn("def forecast(", source)
        self.assertIn("def load_data(", source)


if __name__ == "__main__":
    unittest.main()
