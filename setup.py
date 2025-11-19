"""
HoverNet + TCAV: Interpretable pCR Prediction from H&E Slides
Setup configuration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Read requirements
def read_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="hovernet-tcav-pcr",
    version="0.1.0",
    author="AI Research Team",
    author_email="rafik.salama@codebase",
    description="Interpretable pathological complete response prediction using HoverNet and TCAV",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hovernet-tcav-pcr",

    packages=find_packages(where="src"),
    package_dir={"": "src"},

    python_requires=">=3.8",

    install_requires=read_requirements("requirements.txt"),

    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "pytest-timeout>=2.1.0",
        ],
    },

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],

    entry_points={
        "console_scripts": [
            "hovernet-segment=hovernet_pipeline.cli:segment_cli",
            "tcav-analyze=tcav_integration.cli:tcav_cli",
            "mil-train=mil_model.cli:train_cli",
            "pcr-predict=interpretability.cli:predict_cli",
        ],
    },
)
