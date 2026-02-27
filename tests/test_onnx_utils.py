"""
Unit tests for ONNX export utility functions.
"""

import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSuppressOnnxExportWarnings(unittest.TestCase):
    """Test cases for suppress_onnx_export_warnings context manager."""

    def _import_with_mock_torch(self):
        """Import suppress_onnx_export_warnings using a mocked torch module."""
        sys.modules.pop('src.onnx_utils', None)
        mock_torch = MagicMock()
        mock_torch.jit.TracerWarning = UserWarning
        with patch.dict('sys.modules', {'torch': mock_torch}):
            from src.onnx_utils import suppress_onnx_export_warnings
        return suppress_onnx_export_warnings

    def test_function_is_callable(self):
        """Test that suppress_onnx_export_warnings is a callable context manager."""
        suppress = self._import_with_mock_torch()
        self.assertTrue(callable(suppress))

    def test_context_manager_works(self):
        """Test that suppress_onnx_export_warnings works as a context manager."""
        suppress = self._import_with_mock_torch()
        executed = False
        with suppress():
            executed = True
        self.assertTrue(executed)

    def test_warning_filters_restored_on_normal_exit(self):
        """Test that warning filters are restored when context exits normally."""
        suppress = self._import_with_mock_torch()
        filters_before = len(warnings.filters)
        with suppress():
            filters_inside = len(warnings.filters)
        filters_after = len(warnings.filters)
        self.assertGreater(filters_inside, filters_before)
        self.assertEqual(filters_before, filters_after)

    def test_warning_filters_restored_on_exception(self):
        """Test that warning filters are restored when an exception occurs."""
        suppress = self._import_with_mock_torch()
        filters_before = len(warnings.filters)
        try:
            with suppress():
                raise RuntimeError("Simulated export error")
        except RuntimeError:
            pass
        filters_after = len(warnings.filters)
        self.assertEqual(filters_before, filters_after)


if __name__ == '__main__':
    unittest.main()
