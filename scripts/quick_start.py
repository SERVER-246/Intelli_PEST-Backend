#!/usr/bin/env python3
"""
Quick Start Script for TFLite Conversion
Run this script to convert ONNX models to TFLite format
Location: scripts/run_tflite_conversion.bat calls this
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("TFLite Conversion Quick Start")
    print("=" * 50)
    print("\nAvailable conversion options:")
    print("1. Convert all ONNX models to TFLite")
    print("2. Validate TFLite accuracy")
    print("3. Compare ONNX vs TFLite")
    print("4. Full pipeline (convert + validate + compare)")
    print("\nUpdate this script with your preferred settings")

if __name__ == "__main__":
    main()
