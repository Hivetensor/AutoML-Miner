"""Genetic Programming Module for PyTorch.

This package provides tools for evolving neural network components using genetic programming.
The main components include:

- GeneticInterpreter: Executes genetic programs as PyTorch operations
- GeneticOperators: Provides mutation, crossover and selection operations  
- GeneticLibrary: Contains standard genetic functions
- FunctionConverter: Converts between genetic code and Python functions
"""

# from .genetic_library import GeneticLibrary # Removed as it's obsolete
from .genetic_operators import GeneticOperators
from .interpreter import GeneticInterpreter
from .function_converter import FunctionConverter

__all__ = [
    # 'GeneticLibrary', # Removed as it's obsolete
    'GeneticOperators',
    'GeneticInterpreter',
    'FunctionConverter'
]

__version__ = '0.1.0'
