"""Convergence tests for the genetic programming system."""

import torch
import numpy as np
import pytest
from ..interpreter import GeneticInterpreter
from ..genetic_operators import GeneticOperators
# from ..genetic_library import GeneticLibrary # Removed - Obsolete
from ...evolution_engine import EvolutionEngine

def test_convergence():
    """Test the genetic algorithm converges toward better solutions with more generations."""
    
    # Create a suboptimal version of MSE as a starting point
    # This is intentionally imperfect so we can measure improvement
    suboptimal_mse = [
        [100, 1, 0],     # LOAD R1, S0 (y_true)
        [100, 2, 1],     # LOAD R2, S1 (y_pred)
        [2, 1, 2, 3],    # SUB R1, R2, R3 (y_true - y_pred)
        [3, 3, 3, 4],    # MUL R3, R3, R4 (using multiply instead of square)
        [201, 4, -1, 5],   # REDUCE_MEAN R7, axis=-1, R5
        [201, 5, 0, 9],  # REDUCE_MEAN R4, axis=0, R9 (missing a dimension reduction)
        [453, 9]         # RETURN R9
    ]
    
    # Task definition
    task = {
        'functions': [{
            'function': suboptimal_mse,
            'id': 'suboptimal_mse'
        }],
        'component_type': 'loss',
        'batch_id': 'convergence_test'
    }
    
    # Run with few generations
    engine_few_gen = EvolutionEngine()
    engine_few_gen.config.update({
        'generations': 5,       # Few generations
        'population_size': 10,
        'mutation_rate': 0.3,
        'evaluation_rounds': 2  # Faster evaluation for testing
    })
    
    few_gen_result = engine_few_gen.evolve_function(task.copy())
    assert few_gen_result is not None
    
    # Run with more generations
    engine_more_gen = EvolutionEngine()
    engine_more_gen.config.update({
        'generations': 15,      # More generations
        'population_size': 10,  # Same population size
        'mutation_rate': 0.3,   # Same mutation rate
        'evaluation_rounds': 2  # Same evaluation speed
    })
    
    more_gen_result = engine_more_gen.evolve_function(task.copy())
    assert more_gen_result is not None
    
    # Create test data for evaluation
    test_y_true = torch.rand((10, 5)).detach()
    test_y_pred = torch.rand((10, 5)).detach()
    
    # Reference MSE calculation
    standard_mse = torch.mean((test_y_true - test_y_pred) ** 2)
    
    # Evaluate both solutions against reference MSE
    interpreter = GeneticInterpreter()
    
    # Evaluate few generations solution
    interpreter.initialize({'y_true': test_y_true, 'y_pred': test_y_pred})
    interpreter.load_program(few_gen_result['genetic_code'])
    few_gen_output = interpreter.execute()
    
    # Evaluate more generations solution
    interpreter.initialize({'y_true': test_y_true, 'y_pred': test_y_pred})
    interpreter.load_program(more_gen_result['genetic_code'])
    more_gen_output = interpreter.execute()
    
    # Calculate error relative to standard MSE
    few_gen_error = torch.abs(few_gen_output - standard_mse) / (standard_mse + 1e-10)
    more_gen_error = torch.abs(more_gen_output - standard_mse) / (standard_mse + 1e-10)
    
    print(f"Standard MSE: {standard_mse.item():.6f}")
    print(f"Few generations output: {few_gen_output.item():.6f}")
    print(f"More generations output: {more_gen_output.item():.6f}")
    print(f"Error after {engine_few_gen.config['generations']} generations: {few_gen_error.item():.4f}")
    print(f"Error after {engine_more_gen.config['generations']} generations: {more_gen_error.item():.4f}")
    
    # More generations should typically produce a result closer to standard MSE
    # Due to stochastic nature, we use a relaxed comparison
    assert more_gen_error.item() <= few_gen_error.item() * 1.25, \
        "More generations should not result in significantly worse approximation"
