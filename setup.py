#!/usr/bin/env python3
"""
Setup configuration for Intelli_PEST-Backend package
Allows installation via: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = [
    line.strip() 
    for line in requirements_path.read_text().splitlines()
    if line.strip() and not line.startswith('#')
]

setup(
    name="intelli-pest-backend",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Professional ML Pipeline for Pest Detection with Ensemble Methods",
    long_description=(Path(__file__).parent / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Intelli_PEST-Backend",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/Intelli_PEST-Backend/issues",
        "Documentation": "https://github.com/yourusername/Intelli_PEST-Backend/blob/main/docs/README.md",
        "Source Code": "https://github.com/yourusername/Intelli_PEST-Backend",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "intelli-pest=scripts.quick_start:main",
        ],
    },
    include_package_data=True,
    keywords="machine-learning deep-learning ensemble pest-detection pytorch onnx tflite",
    zip_safe=False,
)
