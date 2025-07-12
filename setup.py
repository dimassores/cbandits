#!/usr/bin/env python3
"""
Setup script for the cbandits package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cbandits",
    version="0.1.0",
    author="Dima Soares Lima",
    author_email="dimassoareslima@gmail.com",
    description="Budget-Constrained Bandits with General Cost and Reward Distributions",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/dimassores/cbandits",
    project_urls={
        "Bug Tracker": "https://github.com/dimassores/cbandits/issues",
        "Documentation": "https://cbandits.readthedocs.io/",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Reinforcement Learning",
        "Topic :: Scientific/Engineering :: Multi-armed Bandits",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "jupyterlab>=4.2.0,<4.3.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cbandits-demo=cbandits.examples.simple_examples.simple_ucb_b1_example:main",
        ],
    },
    keywords="bandits, multi-armed-bandits, budget-constrained, reinforcement-learning, optimization",
    include_package_data=True,
    zip_safe=False,
) 