"""
Taber Bridge — schema, validation, command construction, and execution
for routing mtnsails LLM output to the taber_enviro ONNX predictor.

Two execution modes are supported:
  * Python API  — imports ``ONNXPredictor`` from the taber_enviro package and
                  runs inference in-process.  Requires the taber_enviro Python
                  package to be importable (``pip install taber_enviro``) and a
                  path to its pre-built model directory.
  * Subprocess  — calls the ``taber_enviro`` CLI via ``subprocess.run``.
                  Requires the taber_enviro application to be installed and
                  available on PATH (legacy fallback).
"""

import ast
import json
import re
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Targets supported by taber_enviro's predictor
VALID_TARGETS = {"temp", "barometer", "light", "humidity"}

# Output formats accepted by taber_enviro
VALID_FORMATS = {"json", "csv", "table"}


@dataclass
class TaberForecastRequest:
    """Structured forecast request forwarded to the taber_enviro predictor."""

    # Comma-separated key=value sensor query, e.g. "sensor_id=1,latitude=40.0"
    query: str
    # Forecast duration (e.g. hours ahead)
    duration: float
    # Sampling interval (e.g. 1 hour)
    interval: float
    # Output format for predictor results
    format: str
    # Subset of VALID_TARGETS; empty list means predict all
    targets: List[str] = field(default_factory=list)
    # Optional path to a single data file
    data: Optional[str] = None
    # Optional path to a directory containing data files
    data_dir: Optional[str] = None


def validate_request(raw: dict) -> TaberForecastRequest:
    """
    Validate a raw dict against the TaberForecastRequest schema.

    Args:
        raw: Parsed JSON dict from LLM output.

    Returns:
        A validated TaberForecastRequest.

    Raises:
        ValueError: If required fields are missing or values are invalid.
    """
    # Check required fields
    for key in ("duration", "interval", "format"):
        if key not in raw:
            raise ValueError(f"Missing required field: '{key}'")

    # "query" defaults to "sensor_id=1" when omitted by the model (e.g. for
    # general phenomenon questions without explicit sensor parameters).
    query = raw.get("query", "sensor_id=1")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("'query' must be a non-empty string")

    try:
        duration = float(raw["duration"])
    except (TypeError, ValueError):
        raise ValueError("'duration' must be a number")

    try:
        interval = float(raw["interval"])
    except (TypeError, ValueError):
        raise ValueError("'interval' must be a number")

    fmt = raw["format"]
    if fmt not in VALID_FORMATS:
        raise ValueError(f"'format' must be one of {sorted(VALID_FORMATS)}, got '{fmt}'")

    # Targets are optional; default to all when omitted or empty
    targets = raw.get("targets") or []
    if not isinstance(targets, list):
        raise ValueError("'targets' must be a list")
    invalid = set(targets) - VALID_TARGETS
    if invalid:
        raise ValueError(f"Invalid target(s): {sorted(invalid)}. Must be from {sorted(VALID_TARGETS)}")

    data = raw.get("data")
    data_dir = raw.get("data_dir")

    return TaberForecastRequest(
        query=query.strip(),
        duration=duration,
        interval=interval,
        format=fmt,
        targets=targets,
        data=data,
        data_dir=data_dir,
    )


def build_taber_command(req: TaberForecastRequest, taber_cmd: str = "taber_enviro") -> List[str]:
    """
    Build the CLI argument list for the taber_enviro predictor.

    Args:
        req:       Validated TaberForecastRequest.
        taber_cmd: Path or name of the taber_enviro CLI (default: 'taber_enviro').

    Returns:
        List of strings suitable for subprocess.run().
    """
    cmd = [
        taber_cmd, "predict",
        "--query", req.query,
        "--duration", str(req.duration),
        "--interval", str(req.interval),
        "--format", req.format,
    ]

    # Add optional targets (comma-separated)
    if req.targets:
        cmd += ["--targets", ",".join(req.targets)]

    # Add optional data sources
    if req.data:
        cmd += ["--data", req.data]
    if req.data_dir:
        cmd += ["--data-dir", req.data_dir]

    return cmd


def run_taber(req: TaberForecastRequest, taber_cmd: str = "taber_enviro") -> str:
    """
    Execute the taber_enviro predictor via subprocess and return stdout.

    Args:
        req:       Validated TaberForecastRequest.
        taber_cmd: Path or name of the taber_enviro CLI.

    Returns:
        Captured stdout from the predictor.

    Raises:
        RuntimeError: If the predictor exits with a non-zero status.
    """
    cmd = build_taber_command(req, taber_cmd)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"taber_enviro exited with code {result.returncode}:\n{result.stderr.strip()}"
        )

    return result.stdout


def _repair_json(candidate: str) -> dict:
    """
    Attempt to repair and parse a JSON-like string produced by an LLM.

    Handles common LLM formatting issues in order of preference:
      1. ``json.loads()`` — valid JSON, no repair needed.
      2. ``ast.literal_eval()`` — Python dict syntax (single quotes,
         ``None``/``True``/``False``, trailing commas, etc.).
      3. Remove trailing commas before closing braces/brackets then
         retry ``json.loads()``.  This covers the residual case of
         double-quoted JSON that only suffers from trailing commas.

    Args:
        candidate: The raw brace-delimited substring to parse.

    Returns:
        Parsed dict.

    Raises:
        ValueError: If none of the repair strategies succeed.
    """
    # Strategy 1: standard JSON
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Python literal (handles single quotes, None/True/False,
    # trailing commas, and other Python dict syntax quirks)
    try:
        result = ast.literal_eval(candidate)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass

    # Strategy 3: remove trailing commas before closing braces/brackets,
    # then retry json.loads — handles double-quoted JSON with trailing commas
    # that ast.literal_eval cannot parse (e.g. when keys are unquoted).
    repaired = re.sub(r',\s*([}\]])', r'\1', candidate)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON parse error: {exc}") from exc


def extract_json_from_text(text: str) -> dict:
    """
    Extract the first JSON object from a string (LLM may wrap it in prose).

    Args:
        text: Raw LLM output.

    Returns:
        Parsed dict.

    Raises:
        ValueError: If no valid JSON object is found.
    """
    # Walk through the string looking for balanced braces
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM output")

    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                return _repair_json(candidate)

    raise ValueError("Unbalanced braces — JSON object not closed")


# Maps each valid target name to the natural-language keywords that imply it
_TARGET_KEYWORDS: Dict[str, List[str]] = {
    "temp": ["temp", "temperature"],
    "barometer": ["barometer", "pressure", "baro"],
    "humidity": ["humidity", "humid"],
    "light": ["light", "solar", "radiation"],
}


def parse_fallback_request(user_request: str) -> dict:
    """
    Heuristically derive a TaberForecastRequest-compatible dict from a
    natural-language query.

    Used as a fallback when the LLM fails to emit valid JSON.  The result
    uses sensible defaults for any parameter that cannot be inferred.

    Args:
        user_request: The raw user request string.

    Returns:
        A dict compatible with ``validate_request()``.
    """
    text = user_request.lower()

    # Duration: match patterns like "6 hours", "next 6 hours", "6-hour", "6 hr"
    # The optional separator ([-\s]?) handles the hyphenated form "6-hour".
    duration = 24.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*[-\s]?(?:hours?|hrs?)\b", text)
    if m:
        duration = float(m.group(1))

    # Targets: collect any whose keywords appear as whole words in the request
    targets = [
        target
        for target, keywords in _TARGET_KEYWORDS.items()
        if any(re.search(r"\b" + kw + r"\b", text) for kw in keywords)
    ]

    # Output format: whole-word match to avoid false positives (e.g. "notable"
    # matching "table" or "jsonify" matching "json"); default to "table".
    fmt = "table"
    for kw in ("json", "csv", "table"):
        if re.search(r"\b" + kw + r"\b", text):
            fmt = kw
            break

    return {
        "query": "sensor_id=1",
        "duration": duration,
        "interval": 1.0,
        "format": fmt,
        "targets": targets,
    }


def parse_query_string(query_str: str) -> Dict[str, object]:
    """
    Parse a ``key=value,key2=value2`` sensor query string into a dict.

    Numeric values are coerced to ``float``; everything else stays as a string.

    Args:
        query_str: Comma-separated ``key=value`` pairs, e.g.
                   ``"sensor_id=1,latitude=40.0,longitude=-105.0"``.

    Returns:
        Dict with string keys and float-or-string values.

    Raises:
        ValueError: If any pair cannot be split on ``=``.
    """
    result: Dict[str, object] = {}
    for pair in query_str.split(","):
        if "=" not in pair:
            raise ValueError(f"Invalid query pair (no '='): '{pair.strip()}'")
        key, _, value = pair.partition("=")
        key = key.strip()
        value = value.strip()
        try:
            result[key] = float(value)
        except ValueError:
            result[key] = value
    return result


def run_taber_python(req: "TaberForecastRequest", taber_model_dir: str) -> str:
    """
    Run the taber_enviro predictor **in-process** using its Python API.

    This function imports ``ONNXPredictor`` from the taber_enviro package and
    calls it directly — no subprocess, no separate application.  Only the
    taber_enviro Python package needs to be importable; the CLI does not need
    to be on PATH.

    Args:
        req:             Validated ``TaberForecastRequest``.
        taber_model_dir: Path to the taber_enviro model directory that
                         contains an ``onnx/`` sub-directory with
                         ``model.onnx`` and ``scaler.onnx``.

    Returns:
        Formatted prediction string (JSON, CSV, or table).

    Raises:
        RuntimeError: If the taber_enviro package cannot be imported.
        ValueError:   If there is insufficient historical data for the query.
        FileNotFoundError: If the model directory or ONNX files are missing.
    """
    try:
        from pipeline.predictor import ONNXPredictor  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "The taber_enviro Python package is not importable. "
            "Install it with:  pip install taber_enviro\n"
            f"Original error: {exc}"
        ) from exc

    predictor = ONNXPredictor(taber_model_dir, data_dir=req.data_dir)
    predictor.load_data(data_file=req.data)

    query = parse_query_string(req.query)
    # req.duration is in hours; ONNXPredictor.forecast() wants minutes
    duration_minutes = int(req.duration * 60)
    # req.interval is in hours; ONNXPredictor.forecast() wants seconds
    interval_seconds = int(req.interval * 3600)
    targets = req.targets if req.targets else None

    predictions = predictor.forecast(
        query=query,
        duration_minutes=duration_minutes,
        interval_seconds=interval_seconds,
        targets=targets,
    )

    return _format_predictions(predictions, req.format, targets)


def _format_predictions(predictions: list, fmt: str, targets: Optional[List[str]]) -> str:
    """
    Render a list of prediction dicts (from ONNXPredictor) as a string.

    Args:
        predictions: List of prediction dicts returned by ``ONNXPredictor.forecast()``.
        fmt:         Output format — ``"json"``, ``"csv"``, or ``"table"``.
        targets:     Target column names that were requested, or ``None`` for all.

    Returns:
        Formatted string.
    """
    _DEFAULT_TARGETS = ["temp", "barometer", "light", "humidity"]

    if fmt == "json":
        return json.dumps(predictions, indent=2)

    if fmt == "csv":
        if not predictions:
            return ""
        headers = list(predictions[0].keys())
        lines = [",".join(headers)]
        for pred in predictions:
            lines.append(",".join(str(pred.get(h, "")) for h in headers))
        return "\n".join(lines)

    # table (default)
    if not predictions:
        return ""
    cols = ["datetime"] + (targets if targets else _DEFAULT_TARGETS)
    header = "  ".join(f"{c:>12}" for c in cols)
    rows = [header, "-" * len(header)]
    for pred in predictions:
        row = []
        for col in cols:
            if col == "datetime":
                row.append(f"{pred.get(col, '')!s:>12}")
            elif col in pred:
                row.append(f"{pred[col]:>12.2f}")
            else:
                row.append(f"{'':>12}")
        rows.append("  ".join(row))
    return "\n".join(rows)
