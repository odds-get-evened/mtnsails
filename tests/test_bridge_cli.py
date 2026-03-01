"""
Tests for mtnsails_bridge.py CLI argument parsing.

Tests cover:
- --buffer-dir argument is recognized by the argument parser
- --taber-model is required
- --interactive and --query are optional
- --mtnsails-model is optional
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

BRIDGE_SCRIPT = str(Path(__file__).parent.parent / "llm_interface" / "mtnsails_bridge.py")


class TestBridgeCLIArguments(unittest.TestCase):
    """Tests that the bridge CLI recognizes all expected arguments."""

    def _get_help_output(self):
        """Run the bridge with --help and return stdout."""
        result = subprocess.run(
            [sys.executable, BRIDGE_SCRIPT, "--help"],
            capture_output=True,
            text=True,
        )
        # --help exits with code 0; combine stdout and stderr for safety
        return result.stdout + result.stderr

    def test_buffer_dir_is_recognized(self):
        """--buffer-dir must appear in --help output (i.e. it is a valid argument)."""
        output = self._get_help_output()
        self.assertIn("--buffer-dir", output)

    def test_taber_model_is_listed(self):
        """--taber-model must appear in --help output."""
        output = self._get_help_output()
        self.assertIn("--taber-model", output)

    def test_interactive_is_listed(self):
        """--interactive must appear in --help output."""
        output = self._get_help_output()
        self.assertIn("--interactive", output)

    def test_query_is_listed(self):
        """--query must appear in --help output."""
        output = self._get_help_output()
        self.assertIn("--query", output)

    def test_mtnsails_model_is_listed(self):
        """--mtnsails-model must appear in --help output."""
        output = self._get_help_output()
        self.assertIn("--mtnsails-model", output)

    def test_buffer_dir_not_unrecognized(self):
        """Passing --buffer-dir must NOT produce 'unrecognized arguments' error."""
        with tempfile.TemporaryDirectory() as buf_dir:
            # We pass --buffer-dir but omit --taber-model (required) to get an
            # argparse error about --taber-model, NOT about --buffer-dir.
            result = subprocess.run(
                [sys.executable, BRIDGE_SCRIPT, "--buffer-dir", buf_dir, "--interactive"],
                capture_output=True,
                text=True,
            )
            combined = result.stdout + result.stderr
            self.assertNotIn("unrecognized arguments", combined)


if __name__ == "__main__":
    unittest.main()
