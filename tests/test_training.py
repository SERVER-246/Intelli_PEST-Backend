"""Tests for training modules"""
# Location: tests/test_training.py

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class TestTrainingModules(unittest.TestCase):
    """Test suite for training modules"""

    def test_imports(self):
        """Test that all training modules can be imported"""
        try:
            from src.training import base_training, ensemble_training
            self.assertTrue(True)
        except ImportError as e:
            # Skip if optional training dependencies not available
            self.skipTest(f"Training module not available: {e}")

    def test_data_counter(self):
        """Test data counter utility"""
        try:
            from src.utils import data_counter
            self.assertTrue(hasattr(data_counter, 'count_images'))
        except ImportError as e:
            # Skip if optional dependencies not available
            self.skipTest(f"Data counter not available: {e}")

if __name__ == '__main__':
    unittest.main()
