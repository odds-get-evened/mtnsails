"""
Retrain Daemon - Validator-Confirmed Continual Learning

Periodically scans the validated-interaction buffer and fine-tunes the
MTN Sails LLM when the configured example threshold is reached.  Designed
to run as a background daemon process alongside the bridge server.

Only validated interactions (intent_valid=True, lstm_success=True) stored
by :mod:`llm_interface.retrain_buffer` are used for training, preventing
the model from learning from failed or low-quality interactions.

Usage::

    python -m llm_interface.retrain_daemon \\
        --model-dir ./mtnsails_model \\
        --buffer-dir ./llm_interface/retrain_buffer \\
        --min-examples 200 \\
        --check-interval 60 \\
        --epochs 1 \\
        --batch-size 4 \\
        --learning-rate 1e-5
"""

import argparse
import fcntl
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Ensure project root is importable regardless of working directory
_root = str(Path(__file__).parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from llm_interface.retrain_buffer import count_buffer_records, load_buffer_records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _acquire_lock(lock_path: Path) -> int:
    """
    Try to acquire an exclusive non-blocking file lock.

    Returns:
        A file descriptor >= 0 on success, or -1 if the lock is held.
    """
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except (OSError, BlockingIOError):
        return -1


def _release_lock(fd: int) -> None:
    """Release and close the file lock descriptor."""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Core retraining logic
# ---------------------------------------------------------------------------

def retrain_once(
    model_dir: str,
    buffer_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    archive_dir: Optional[str] = None,
) -> bool:
    """
    Perform one retraining cycle from buffered validated interactions.

    Steps:
    1. Load all validated records from *buffer_dir*.
    2. Extract ``training_text`` strings.
    3. Fine-tune the model in *model_dir* via ``LLMTrainer``.
    4. Delete (or archive) the consumed buffer files.

    Args:
        model_dir: Path to the MTN Sails model directory.
        buffer_dir: Path to the JSONL buffer directory.
        epochs: Number of fine-tuning epochs.
        batch_size: Batch size for fine-tuning.
        learning_rate: Learning rate for fine-tuning.
        archive_dir: If provided, move consumed buffer files here instead of
                     deleting them.

    Returns:
        ``True`` if retraining completed successfully, ``False`` otherwise.
    """
    records = load_buffer_records(buffer_dir)
    if not records:
        print(f"[{_timestamp()}] Buffer is empty – nothing to retrain on.")
        return False

    # Collect non-empty training texts from validated records
    training_texts = [r["training_text"] for r in records if r.get("training_text")]
    if not training_texts:
        print(f"[{_timestamp()}] No training_text found in buffer records.")
        return False

    print(f"[{_timestamp()}] Fine-tuning on {len(training_texts)} validated examples...")

    # Import trainer lazily to avoid heavy ML imports at daemon startup
    try:
        from llm_interface.taber_enviro_trainer import TaberEnviroTrainer
        from src.trainer import LLMTrainer
    except ImportError as exc:
        print(f"[{_timestamp()}] ERROR: Cannot import trainer: {exc}")
        print("Install dependencies: pip install transformers torch")
        return False

    try:
        taber_trainer = TaberEnviroTrainer(output_dir=model_dir)
        # Load from existing checkpoint when available; else use base model name
        model_name = model_dir if taber_trainer.is_trained() else taber_trainer.model_name
        llm_trainer = LLMTrainer(
            model_name=model_name,
            output_dir=model_dir,
            device=taber_trainer.device,
        )
        llm_trainer.train(
            train_texts=training_texts,
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        llm_trainer.save_model()
        print(f"[{_timestamp()}] Retraining complete.")
    except Exception as exc:
        print(f"[{_timestamp()}] ERROR: Retraining failed: {exc}")
        return False

    # Consume buffer files – delete or move to archive
    buf_dir = Path(buffer_dir)
    for jsonl_file in sorted(buf_dir.glob("validated_*.jsonl")):
        if archive_dir:
            arch = Path(archive_dir)
            arch.mkdir(parents=True, exist_ok=True)
            jsonl_file.rename(arch / jsonl_file.name)
            print(f"[{_timestamp()}] Archived: {jsonl_file.name}")
        else:
            jsonl_file.unlink()
            print(f"[{_timestamp()}] Deleted:  {jsonl_file.name}")

    return True


# ---------------------------------------------------------------------------
# Daemon loop
# ---------------------------------------------------------------------------

def run_daemon(
    model_dir: str,
    buffer_dir: str,
    min_examples: int,
    check_interval: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    archive_dir: Optional[str] = None,
) -> None:
    """
    Run the retraining daemon loop.

    Checks the buffer every *check_interval* seconds.  When the total number
    of validated examples reaches *min_examples*, a retraining cycle runs.
    A filesystem lock file in *model_dir* prevents concurrent retraining.

    Args:
        model_dir: Path to the MTN Sails model directory.
        buffer_dir: Path to the JSONL buffer directory.
        min_examples: Minimum validated examples before retraining is triggered.
        check_interval: Seconds between buffer size checks.
        epochs: Fine-tuning epochs per cycle.
        batch_size: Batch size per cycle.
        learning_rate: Learning rate per cycle.
        archive_dir: Optional directory for consumed buffer files.
    """
    lock_path = Path(model_dir) / ".retrain.lock"
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    print(f"[{_timestamp()}] Retrain daemon started.")
    print(f"  model-dir:      {model_dir}")
    print(f"  buffer-dir:     {buffer_dir}")
    print(f"  min-examples:   {min_examples}")
    print(f"  check-interval: {check_interval}s")
    print(f"  epochs:         {epochs}")
    print(f"  batch-size:     {batch_size}")
    print(f"  learning-rate:  {learning_rate}")
    if archive_dir:
        print(f"  archive-dir:    {archive_dir}")
    print()

    while True:
        n = count_buffer_records(buffer_dir)
        print(f"[{_timestamp()}] Buffer: {n}/{min_examples} examples.", flush=True)

        if n >= min_examples:
            # Prevent concurrent retraining with a file lock
            fd = _acquire_lock(lock_path)
            if fd < 0:
                print(f"[{_timestamp()}] Another retrain is in progress – skipping.")
            else:
                try:
                    retrain_once(
                        model_dir=model_dir,
                        buffer_dir=buffer_dir,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        archive_dir=archive_dir,
                    )
                finally:
                    _release_lock(fd)

        time.sleep(check_interval)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for the retraining daemon."""
    parser = argparse.ArgumentParser(
        description="Retrain daemon: fine-tunes MTN Sails LLM on validated interactions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python -m llm_interface.retrain_daemon \\
      --model-dir ./mtnsails_model \\
      --buffer-dir ./llm_interface/retrain_buffer \\
      --min-examples 200 \\
      --check-interval 60 \\
      --epochs 1 \\
      --batch-size 4 \\
      --learning-rate 1e-5
        """,
    )
    parser.add_argument("--model-dir", required=True,
                        help="Path to MTN Sails model directory")
    parser.add_argument("--buffer-dir", required=True,
                        help="Path to JSONL buffer directory")
    parser.add_argument("--min-examples", type=int, default=200,
                        help="Minimum validated examples before retraining (default: 200)")
    parser.add_argument("--check-interval", type=int, default=60,
                        help="Seconds between buffer size checks (default: 60)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Fine-tuning epochs per cycle (default: 1)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for fine-tuning (default: 4)")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="Learning rate for fine-tuning (default: 1e-5)")
    parser.add_argument("--archive-dir", default=None,
                        help="Move consumed buffer files here instead of deleting")

    args = parser.parse_args()

    run_daemon(
        model_dir=args.model_dir,
        buffer_dir=args.buffer_dir,
        min_examples=args.min_examples,
        check_interval=args.check_interval,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        archive_dir=args.archive_dir,
    )


if __name__ == "__main__":
    main()
