"""
Setup script for MTN Sails - LLM Training and ONNX Conversion System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="mtnsails",
    version="0.1.0",
    description="LLM Training and ONNX Conversion System for CPU-friendly conversational AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MTN Sails Team",
    license="GPL-3.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "optimum[onnxruntime]>=1.13.0",
        "onnxruntime>=1.15.0",
        "datasets>=2.12.0",
        "accelerate>=0.20.0",
        "safetensors>=0.3.0",
    ],
    entry_points={
        "console_scripts": [
            "mtnsails=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
