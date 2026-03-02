"""
TaberBridgeExecutor — connects the mtnsails ONNX LLM to the taber_enviro predictor.

Flow:
  1. Load the mtnsails ONNX model via ChatInterface.
  2. Send the user's natural-language request with a JSON-only system prompt.
  3. Extract and validate the JSON response as a TaberForecastRequest.
  4. Run the taber_enviro predictor — either:
       a. In-process via the Python API (preferred, no subprocess) when
          ``taber_model_dir`` is provided, or
       b. Via subprocess CLI when only ``taber_cmd`` is provided (legacy).
  5. Return (and optionally save) the predictor output.
"""

import json
from pathlib import Path
from typing import Optional

from src.chat_interface import ChatInterface
from src.taber_bridge import (
    TaberForecastRequest,
    extract_json_from_text,
    run_taber,
    run_taber_python,
    validate_request,
)

# Prompt instructing the LLM to emit JSON only
_SYSTEM_PROMPT = """\
You are a forecasting assistant. Given a natural-language forecasting request, \
respond with ONLY a valid JSON object — no prose, no markdown, no extra keys.

Required JSON fields (ALL four must always be present):
  "query"    : comma-separated key=value sensor parameters, \
e.g. "sensor_id=1,latitude=40.0,longitude=-105.0". \
If no specific sensor is mentioned, use "sensor_id=1" as the default.
  "duration" : numeric forecast horizon in hours (e.g. 24). \
If not specified, default to 24.
  "interval" : numeric sampling interval in hours (e.g. 1). \
If not specified, default to 1.
  "format"   : output format, one of "json", "csv", "table". \
If not specified, default to "table".

Optional JSON fields:
  "targets"  : list from ["temp","barometer","light","humidity"] — omit to predict all
  "data"     : path to a single data file
  "data_dir" : path to a directory of data files

Example for a specific sensor request:
{
  "query": "sensor_id=7,latitude=39.5,longitude=-106.2,altitude=2800",
  "duration": 24,
  "interval": 1,
  "format": "json",
  "targets": ["temp", "humidity"]
}

Example for a general phenomenon question (no specific sensor mentioned — defaults to sensor_id=1):
{
  "query": "sensor_id=1",
  "duration": 24,
  "interval": 1,
  "format": "table",
  "targets": ["temp"]
}

Respond with the JSON object only.\
"""


class TaberBridgeExecutor:
    """
    Orchestrates the LLM → JSON → taber_enviro pipeline.

    Args:
        onnx_model_path:  Path to the mtnsails ONNX model directory.
        device:           Inference device ('cpu' or 'cuda').
        max_length:       Max tokeniser input length.
        max_new_tokens:   Max tokens the LLM may generate.
        taber_model_dir:  Path to the taber_enviro model directory that
                          contains an ``onnx/`` sub-directory with the
                          pre-built ``model.onnx`` and ``scaler.onnx`` files.
                          When provided the predictor is called **in-process**
                          via the taber_enviro Python API — no subprocess or
                          separate application is required.
        taber_cmd:        taber_enviro CLI name or full path (legacy fallback
                          used only when ``taber_model_dir`` is *not* set).
    """

    def __init__(
        self,
        onnx_model_path: str,
        device: str = "cpu",
        max_length: int = 512,
        max_new_tokens: int = 256,
        taber_model_dir: Optional[str] = None,
        taber_cmd: str = "taber_enviro",
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.taber_model_dir = taber_model_dir
        self.taber_cmd = taber_cmd

        # Load ONNX model — logging disabled by default
        self._chat = ChatInterface(
            onnx_model_path=onnx_model_path,
            device=device,
            max_length=max_length,
        )

    def _prompt_llm(self, user_request: str) -> str:
        """Build the full prompt and ask the LLM for JSON output."""
        full_prompt = f"{_SYSTEM_PROMPT}\n\nRequest: {user_request}"
        return self._chat.generate_response(full_prompt, max_new_tokens=self.max_new_tokens)

    def run(
        self,
        user_request: str,
        save_dir: Optional[str] = None,
    ) -> str:
        """
        Execute the full bridge pipeline.

        Args:
            user_request: Natural-language forecast description from the user.
            save_dir:     Optional directory in which to save the raw JSON request
                          and predictor response (useful for collecting retraining data).

        Returns:
            The predictor output string (stdout from taber_enviro).
        """
        # Step 1 — ask LLM to produce JSON
        llm_output = self._prompt_llm(user_request)

        # Step 2 — extract and validate JSON
        raw = extract_json_from_text(llm_output)
        request: TaberForecastRequest = validate_request(raw)

        # Step 3 — run taber_enviro predictor
        if self.taber_model_dir:
            # Preferred: in-process Python API — no subprocess needed
            predictor_output = run_taber_python(request, self.taber_model_dir)
        else:
            # Legacy fallback: invoke the taber_enviro CLI via subprocess
            predictor_output = run_taber(request, self.taber_cmd)

        # Step 4 — optionally persist request + response for retraining
        if save_dir:
            self._save_pair(raw, predictor_output, save_dir)

        return predictor_output

    @staticmethod
    def _save_pair(request_dict: dict, response: str, save_dir: str) -> None:
        """
        Persist the JSON request and predictor response to save_dir.

        Files written:
          <save_dir>/taber_request_<timestamp>.json
          <save_dir>/taber_response_<timestamp>.txt
        where <timestamp> is an ISO-8601-like string (UTC, no colons for portability).
        """
        from datetime import datetime, timezone

        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Use a UTC timestamp so filenames sort chronologically
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")

        (out / f"taber_request_{ts}.json").write_text(
            json.dumps(request_dict, indent=2), encoding="utf-8"
        )
        (out / f"taber_response_{ts}.txt").write_text(response, encoding="utf-8")
