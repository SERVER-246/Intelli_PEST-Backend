"""Tests for conversion modules"""
# Location: tests/test_conversion.py

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class TestConversionModules(unittest.TestCase):
    """Test suite for model conversion modules"""
    
    def test_imports(self):
        """Test that all conversion modules can be imported"""
        try:
            from src.conversion import onnx_converter
            from src.conversion import tflite_converter
            from src.conversion import model_validator
            from src.conversion import comparison_analyzer
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import conversion modules: {e}")

if __name__ == '__main__':
    unittest.main()
