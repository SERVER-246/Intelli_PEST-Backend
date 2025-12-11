"""Tests for inference"""
# Location: tests/test_inference.py

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class TestInference(unittest.TestCase):
    """Test suite for inference functionality"""
    
    def test_module_structure(self):
        """Test that all required modules exist"""
        src_dir = PROJECT_ROOT / 'src'
        required_dirs = ['training', 'conversion', 'deployment', 'utils']
        
        for dirname in required_dirs:
            module_dir = src_dir / dirname
            self.assertTrue(module_dir.exists(), f"Missing {dirname} module")

if __name__ == '__main__':
    unittest.main()
