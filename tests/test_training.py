"""Tests for training modules"""
# Location: tests/test_training.py

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class TestTrainingModules(unittest.TestCase):
    """Test suite for training modules"""
    
    def test_imports(self):
        """Test that all training modules can be imported"""
        try:
            from src.training import base_training
            from src.training import ensemble_training
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import training modules: {e}")
    
    def test_data_counter(self):
        """Test data counter utility"""
        try:
            from src.utils import data_counter
            self.assertTrue(hasattr(data_counter, 'count_images'))
        except Exception as e:
            self.fail(f"Data counter test failed: {e}")

if __name__ == '__main__':
    unittest.main()
