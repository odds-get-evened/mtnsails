"""
Taber Bridge — schema, validation, command construction, and subprocess execution
for routing mtnsails LLM output to the taber_enviro ONNX predictor.
"""

import json
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

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
    for key in ("query", "duration", "interval", "format"):
        if key not in raw:
            raise ValueError(f"Missing required field: '{key}'")

    query = raw["query"]
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
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSON parse error: {exc}") from exc

    raise ValueError("Unbalanced braces — JSON object not closed")
