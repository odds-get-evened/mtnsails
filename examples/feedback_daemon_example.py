#!/usr/bin/env python3
"""
Feedback + Daemon workflow example for MTN Sails.

This script demonstrates how to:
1. Run an interactive chat session in feedback mode (approve/correct responses).
2. Start the background retraining daemon which watches the JSONL file and
   retrains + re-exports the model once enough labeled pairs accumulate.

Run the chat in one terminal:

    python examples/feedback_daemon_example.py chat

Run the daemon in a second terminal:

    python examples/feedback_daemon_example.py daemon

See the README or docs/QUICKSTART.md for the full workflow.
"""

import sys
from pathlib import Path

# Allow running directly from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

FEEDBACK_JSONL = "./live_pairs.jsonl"
ONNX_MODEL_DIR = "./onnx_model"
TRAINED_MODEL_DIR = "./trained_model"
DAEMON_STATE_FILE = "./daemon_state.json"


def chat_with_feedback():
    """Start an interactive chat session in feedback mode."""
    from src.chat_interface import ChatInterface

    if not Path(ONNX_MODEL_DIR).exists():
        print(
            f"No ONNX model found at '{ONNX_MODEL_DIR}'.\n"
            "Run 'python main.py baseline' or 'python main.py pipeline' first."
        )
        sys.exit(1)

    print("=== Feedback Chat Mode ===")
    print(f"Approved/corrected pairs will be saved to: {FEEDBACK_JSONL}")
    print("Once you have collected enough pairs, start the daemon in a second terminal.")
    print()

    chat = ChatInterface(
        onnx_model_path=ONNX_MODEL_DIR,
        feedback_file=FEEDBACK_JSONL
    )
    chat.chat(interactive=True)


def run_daemon():
    """Start the background retraining daemon."""
    from src.daemon import run_daemon as _run_daemon

    print("=== Background Retraining Daemon ===")
    print(f"Watching: {FEEDBACK_JSONL}")
    print(f"Retraining every 50 new examples.")
    print("Press Ctrl-C to stop.\n")

    try:
        _run_daemon(
            feedback_file=FEEDBACK_JSONL,
            state_file=DAEMON_STATE_FILE,
            retrain_threshold=50,
            trained_model_dir=TRAINED_MODEL_DIR,
            onnx_output_dir=ONNX_MODEL_DIR,
            poll_interval=30,
            max_steps=100,
        )
    except KeyboardInterrupt:
        print("\nDaemon stopped.")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("chat", "daemon"):
        print("Usage: python examples/feedback_daemon_example.py [chat|daemon]")
        sys.exit(1)

    if sys.argv[1] == "chat":
        chat_with_feedback()
    else:
        run_daemon()


if __name__ == "__main__":
    main()
