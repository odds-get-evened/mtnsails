"""
Retrain Buffer - Validated Interaction Logger

Appends validator-confirmed interactions to JSONL files for LLM fine-tuning.
Only records interactions where intent parsing succeeded AND the LSTM call
returned results without errors, preventing self-training on failures.

Record format (one JSON object per line)::

    {
        "timestamp":             "<ISO-8601>Z",
        "user_query":            "<original user query>",
        "parsed_intent":         {"type": ..., "metric": ..., "duration": ..., "sensor_id": ...},
        "llm_intent_raw_response": "<raw LLM text used for intent parsing, or null>",
        "intent_prompt":         "<prompt sent to LLM, or null>",
        "lstm_output_summary":   {"type": ..., "metric": ...},
        "intent_valid":          true,
        "lstm_success":          true,
        "training_text":         "User: ...\\nAssistant: ..."
    }
"""

import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def append_validated_interaction(
    buffer_dir: str,
    user_query: str,
    parsed_intent: Dict,
    lstm_output: Dict,
    training_text: str,
    llm_intent_raw_response: Optional[str] = None,
    intent_prompt: Optional[str] = None,
) -> Path:
    """
    Append a validated interaction record to the daily JSONL buffer file.

    Only call this when both intent parsing and LSTM execution succeeded
    (i.e. ``'error' not in lstm_output``).

    Args:
        buffer_dir: Directory for JSONL buffer files (created if absent).
        user_query: Original natural-language query from the user.
        parsed_intent: Structured intent dict (type, metric, duration, sensor_id).
        lstm_output: LSTM result dict – must not contain an ``'error'`` key.
        training_text: Fine-tuning string in ``"User: ...\\nAssistant: ..."`` format.
        llm_intent_raw_response: Raw LLM output used for intent parsing (optional).
        intent_prompt: Prompt sent to the LLM for intent parsing (optional).

    Returns:
        Path to the JSONL buffer file that was written.
    """
    buf_dir = Path(buffer_dir)
    buf_dir.mkdir(parents=True, exist_ok=True)

    # One JSONL file per UTC calendar day keeps individual files small
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    buf_file = buf_dir / f"validated_{date_str}.jsonl"

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_query": user_query,
        "parsed_intent": parsed_intent,
        "llm_intent_raw_response": llm_intent_raw_response,
        "intent_prompt": intent_prompt,
        "lstm_output_summary": {
            "type": lstm_output.get("type"),
            "metric": lstm_output.get("metric"),
        },
        "intent_valid": True,
        "lstm_success": True,
        "training_text": training_text,
    }

    # Serialize and append; flock prevents concurrent write corruption
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with open(buf_file, "a", encoding="utf-8") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.write(line)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

    return buf_file


def load_buffer_records(buffer_dir: str) -> List[Dict]:
    """
    Load all validated records from every JSONL file in *buffer_dir*.

    Malformed lines are silently skipped.

    Args:
        buffer_dir: Directory containing ``validated_<date>.jsonl`` files.

    Returns:
        List of record dicts sorted by filename (ascending date order).
    """
    buf_dir = Path(buffer_dir)
    if not buf_dir.is_dir():
        return []

    records: List[Dict] = []
    for jsonl_file in sorted(buf_dir.glob("validated_*.jsonl")):
        with open(jsonl_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # skip malformed lines

    return records


def count_buffer_records(buffer_dir: str) -> int:
    """Return the total number of validated records across all buffer files."""
    return len(load_buffer_records(buffer_dir))
