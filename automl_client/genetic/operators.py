"""Utility functions for genetic operations."""

import random
from typing import List, Dict, Optional

def generate_random_program(complexity: int = 5, favor_categories: Optional[List[str]] = None) -> List:
    """
    Generate a random genetic program with optional category preference.
    
    Args:
        complexity: Approximate program length
        favor_categories: Optional list of categories to favor
        
    Returns:
        Random genetic program
    """
    from .strategies import SteadyStateGeneticAlgorithm
    strategy = SteadyStateGeneticAlgorithm()
    return strategy._generate_random_program(complexity, favor_categories)

def log_genetic_code_stats(genetic_code: List) -> None:
    """
    Log statistics about a genetic code.
    
    Args:
        genetic_code: Genetic code to analyze
    """
    from .strategies import SteadyStateGeneticAlgorithm
    strategy = SteadyStateGeneticAlgorithm()
    strategy._log_genetic_code_stats(genetic_code)

def weighted_choice(weights: List[float]) -> int:
    """
    Select an index based on weights.
    
    Args:
        weights: List of weights
        
    Returns:
        Selected index
    """
    total = sum(weights)
    r = random.uniform(0, total)
    cumulative_weight = 0
    
    for i, weight in enumerate(weights):
        cumulative_weight += weight
        if r <= cumulative_weight:
            return i
            
    return len(weights) - 1  # Fallback
