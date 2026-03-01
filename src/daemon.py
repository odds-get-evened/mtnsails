"""
Background retraining daemon for MTN Sails.

Watches a JSONL feedback file produced by the ChatInterface feedback mode and
automatically retrains the PyTorch model — then exports to ONNX — once enough
new labeled examples have accumulated.

Typical usage (from the CLI):

    python main.py daemon --feedback-file ./live_pairs.jsonl

State is stored in a small JSON sidecar file (default: ./daemon_state.json) so
restarts pick up exactly where the daemon left off.
"""

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def load_daemon_state(state_file: str) -> dict:
    """
    Load persisted daemon state from a JSON file.

    Returns a dict with at least the key ``lines_consumed`` (int) so the
    daemon knows which JSONL lines have already been processed.

    Args:
        state_file: Path to the JSON state file.

    Returns:
        State dict; an empty/default state dict is returned when the file does
        not yet exist or cannot be parsed.
    """
    path = Path(state_file)
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            # Ensure required key is present
            state.setdefault('lines_consumed', 0)
            return state
        except (json.JSONDecodeError, IOError):
            logger.warning("Could not read daemon state from %s; starting fresh.", path)
    return {'lines_consumed': 0}


def save_daemon_state(state_file: str, state: dict) -> None:
    """
    Persist daemon state to a JSON file.

    Args:
        state_file: Path to the JSON state file.
        state: Dict of state values to persist.
    """
    path = Path(state_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# JSONL tail helper
# ---------------------------------------------------------------------------

def count_jsonl_lines(feedback_file: str) -> int:
    """
    Return the total number of non-empty lines in a JSONL file.

    Args:
        feedback_file: Path to the JSONL file.

    Returns:
        Line count (0 if the file does not exist).
    """
    path = Path(feedback_file)
    if not path.exists():
        return 0
    with open(path, 'r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


# ---------------------------------------------------------------------------
# Core daemon loop
# ---------------------------------------------------------------------------

def run_daemon(
    feedback_file: str = "./live_pairs.jsonl",
    state_file: str = "./daemon_state.json",
    retrain_threshold: int = 50,
    model_name: str = "distilgpt2",
    trained_model_dir: str = "./trained_model",
    onnx_output_dir: str = "./onnx_model",
    poll_interval: int = 30,
    batch_size: int = 4,
    max_steps: int = 100,
    device: str = "cpu"
) -> None:
    """
    Watch *feedback_file* and retrain when *retrain_threshold* new examples
    arrive since the last retraining cycle.

    After retraining completes:
    1. The updated PyTorch model is saved to *trained_model_dir*.
    2. The model is exported to ONNX in a staging directory
       (<onnx_output_dir>_next).
    3. The current ONNX directory (if present) is renamed to
       <onnx_output_dir>_prev, then the staging directory is renamed to
       <onnx_output_dir>.

    This atomic rename approach means the chat interface always sees a
    consistent ONNX model (a manual restart is still required to pick up the
    new weights).

    Args:
        feedback_file: Path to the JSONL file written by ChatInterface feedback mode.
        state_file: Path to the JSON daemon state file.
        retrain_threshold: Number of *new* labeled examples needed to trigger a
            retraining cycle (default: 50).
        model_name: Base Hugging Face model name used when no checkpoint exists.
        trained_model_dir: Directory where the PyTorch model is saved/loaded.
        onnx_output_dir: Directory where the production ONNX model lives.
        poll_interval: Seconds to sleep between file checks (default: 30).
        batch_size: Batch size passed to LLMTrainer.
        max_steps: Maximum gradient steps per retraining cycle (default: 100).
            Passed directly to LLMTrainer.train(max_steps=...).  A short cap
            keeps incremental retraining fast; set to -1 to use epoch-based
            training instead.
        device: Torch device string ('cpu' or 'cuda').
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [daemon] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    feedback_path = Path(feedback_file)
    onnx_path = Path(onnx_output_dir)
    onnx_next = Path(str(onnx_output_dir) + "_next")
    onnx_prev = Path(str(onnx_output_dir) + "_prev")

    logger.info("Daemon started.")
    logger.info("  Feedback file  : %s", feedback_path)
    logger.info("  State file     : %s", state_file)
    logger.info("  Retrain every  : %d new examples", retrain_threshold)
    logger.info("  Poll interval  : %d s", poll_interval)
    logger.info("  Max steps/run  : %d", max_steps)

    state = load_daemon_state(state_file)
    logger.info("  Lines consumed : %d (from state)", state['lines_consumed'])

    while True:
        try:
            total_lines = count_jsonl_lines(feedback_file)
            new_lines = total_lines - state['lines_consumed']

            if new_lines >= retrain_threshold:
                logger.info(
                    "%d new examples found (threshold: %d) — starting retraining.",
                    new_lines, retrain_threshold
                )
                _retrain_cycle(
                    feedback_file=feedback_file,
                    state=state,
                    model_name=model_name,
                    trained_model_dir=trained_model_dir,
                    onnx_path=onnx_path,
                    onnx_next=onnx_next,
                    onnx_prev=onnx_prev,
                    batch_size=batch_size,
                    max_steps=max_steps,
                    device=device
                )
                # Update and persist the line offset after a successful cycle
                state['lines_consumed'] = total_lines
                save_daemon_state(state_file, state)
                logger.info("State saved. Lines consumed: %d", state['lines_consumed'])
            else:
                logger.debug(
                    "%d/%d new examples — waiting for more.",
                    new_lines, retrain_threshold
                )
        except Exception as exc:  # pragma: no cover — guard against transient errors
            logger.error("Error during daemon cycle: %s", exc, exc_info=True)

        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Retraining cycle
# ---------------------------------------------------------------------------

def _retrain_cycle(
    feedback_file: str,
    state: dict,
    model_name: str,
    trained_model_dir: str,
    onnx_path: Path,
    onnx_next: Path,
    onnx_prev: Path,
    batch_size: int,
    max_steps: int,
    device: str
) -> None:
    """
    Execute one full retrain-then-export cycle.

    Imports are deferred so the daemon process does not load heavy ML
    libraries until they are actually needed.
    """
    from src.data_handler import ConversationDataHandler
    from src.trainer import LLMTrainer
    from src.onnx_converter import ONNXConverter

    # --- 1. Load new labeled pairs from JSONL ---
    data_handler = ConversationDataHandler()
    lines_read = data_handler.load_from_jsonl(
        feedback_file,
        start_line=state['lines_consumed']
    )
    logger.info("Loaded %d new training pairs from %s.", len(data_handler), feedback_file)

    if len(data_handler) == 0:
        logger.warning("No valid records found in new lines — skipping cycle.")
        return

    train_texts = data_handler.format_for_training()

    # --- 2. Choose base model or existing checkpoint ---
    checkpoint = Path(trained_model_dir)
    _has_checkpoint = (checkpoint / "config.json").exists() and (
        (checkpoint / "model.safetensors").exists()
        or (checkpoint / "pytorch_model.bin").exists()
    )

    if _has_checkpoint:
        base = trained_model_dir
        # Use a low learning rate when fine-tuning an existing checkpoint to
        # prevent catastrophic forgetting of previously learned knowledge.
        lr = 1e-5
        logger.info("Continuing training from checkpoint: %s (lr=%s)", base, lr)
    else:
        base = model_name
        lr = 5e-5
        logger.info("No checkpoint found — training from base model: %s (lr=%s)", base, lr)

    trainer = LLMTrainer(model_name=base, output_dir=trained_model_dir, device=device)
    trainer.train(
        train_texts=train_texts,
        num_epochs=3,       # Fallback if max_steps <= 0; normally overridden below
        batch_size=batch_size,
        learning_rate=lr,
        max_steps=max_steps  # Short cap for incremental fine-tuning
    )
    saved_path = trainer.save_model()
    logger.info("PyTorch model saved to: %s", saved_path)

    # --- 3. Export to ONNX (staging) then swap directories ---
    logger.info("Exporting to ONNX (staging: %s)…", onnx_next)
    converter = ONNXConverter(saved_path)
    converter.convert_to_onnx(output_path=str(onnx_next))
    logger.info("ONNX export complete.")

    # Atomic directory swap so the chat interface always sees a valid ONNX model
    if onnx_path.exists():
        if onnx_prev.exists():
            shutil.rmtree(onnx_prev)
        onnx_path.rename(onnx_prev)
        logger.info("Previous ONNX model archived to: %s", onnx_prev)

    onnx_next.rename(onnx_path)
    logger.info("New ONNX model is live at: %s", onnx_path)
    logger.info(
        "Chat restart required to load the new model (hot-reload not supported)."
    )
