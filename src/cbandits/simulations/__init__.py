# src/simulations/__init__.py

from .runner import run_simulations

# Define __all__ to specify what gets imported when someone does
# 'from src.simulations import *'
__all__ = [
    "run_simulations",
]