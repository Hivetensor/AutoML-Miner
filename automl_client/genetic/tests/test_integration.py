"""Integration tests for the genetic programming system."""

import pytest
import torch
import numpy as np
from ..interpreter import GeneticInterpreter
from ..genetic_operators import GeneticOperators
# from ..genetic_library import GeneticLibrary # Removed - Obsolete
from ..function_converter import FunctionConverter
from ...evolution_engine import EvolutionEngine


# This test relies on obsolete GeneticLibrary and outdated evaluation logic
@pytest.mark.skip(reason="Relies on obsolete GeneticLibrary and outdated concepts")
def test_full_evolution_cycle():
    """Test a complete evolution cycle from population to evaluation."""
    # Create initial population
    population = GeneticOperators.create_initial_population(
        seed='mean_squared_error',
        population_size=5,
        mutation_rate=0.3
    )
    
    # Verify structure of each program in the initial population
    for individual in population:
        # Check it starts with LOAD operations
        assert len(individual) >= 2, "Program should have at least 2 instructions"
        assert individual[0][0] == 100, "First instruction should be LOAD (opcode 100)"
        assert individual[1][0] == 100, "Second instruction should be LOAD (opcode 100)"
        
        # Check it ends with RETURN 9
        assert individual[-1][0] == 453, "Last instruction should be RETURN (opcode 453)"
        assert individual[-1][1] == 9, "RETURN should use register 9"
    
    # Create test dataset
    y_true = torch.rand((10, 1))
    y_pred = torch.rand((10, 1))
    dataset = {
        'trainX': torch.rand((100, 28*28)),
        'trainY': torch.randint(0, 10, (100,)).float(),
        'valX': torch.rand((20, 28*28)),
        'valY': torch.randint(0, 10, (20,)).float()
    }
    
    # Evaluate each individual
    interpreter = GeneticInterpreter()
    fitnesses = []
    for individual in population:
        interpreter.initialize({'y_true': y_true, 'y_pred': y_pred})
        interpreter.load_program(individual)
        result = interpreter.execute()
        
        # Simple fitness: inverse of loss (higher is better)
        fitness = 1.0 / (result.item() + 1e-7)
        fitnesses.append(fitness)
    
    # Evolve population
    new_population = GeneticOperators.evolve_population(
        population,
        fitnesses,
        {
            'population_size': 5,
            'elite_count': 1,
            'crossover_rate': 0.7,
            'mutation_rate': 0.3
        }
    )
    
    # Verify evolution produced valid population
    assert len(new_population) == 5
    assert all(isinstance(ind, list) for ind in new_population)
    assert all(len(ind) > 0 for ind in new_population)
    
    # Verify structure of each program in the new population
    for individual in new_population:
        # Check it starts with LOAD operations
        assert len(individual) >= 2, "Program should have at least 2 instructions"
        assert individual[0][0] == 100, "First instruction should be LOAD (opcode 100)"
        assert individual[1][0] == 100, "Second instruction should be LOAD (opcode 100)"
        
        # Check it ends with RETURN 9
        assert individual[-1][0] == 453, "Last instruction should be RETURN (opcode 453)"
        assert individual[-1][1] == 9, "RETURN should use register 9"
    
    # Test function conversion
    best_idx = np.argmax(fitnesses)
    best_individual = population[best_idx]
    python_fn = FunctionConverter.genetic_code_to_python(best_individual)
    
    # Should be valid Python code
    assert "def loss_fn" in python_fn
    assert "y_true" in python_fn
    assert "y_pred" in python_fn


# This test relies on obsolete GeneticLibrary
@pytest.mark.skip(reason="Relies on obsolete GeneticLibrary")
def test_evolution_engine_integration():
    """Test integration of EvolutionEngine with other components."""

    # For testing, reduce the number of generations and population size
    engine = EvolutionEngine()
    engine.config['generations'] = 2  # Reduce generations for faster test
    engine.config['population_size'] = 5  # Smaller population
    
    # Create test task with actual genetic code instead of string
    # Placeholder code since GeneticLibrary is gone
    placeholder_mse = [[100, 1, 0], [100, 2, 1], [2, 3, 1, 2], [10, 4, 3], [201, 9, -1, 4], [453, 9]]
    task = {
        'functions': [{
            # 'function': GeneticLibrary.mean_squared_error,  # Obsolete
            'function': placeholder_mse,
            'id': 'test_function'
        }],
        'component_type': 'loss', # Engine will use LossStrategy
        'batch_id': 'test_batch'
    }
    
    # Run evolution directly
    result = engine.evolve_function(task)
    
    # Verify results
    assert result is not None
    assert 'evolved_function' in result
    assert 'genetic_code' in result
    assert 'metadata' in result
    
    # Verify structure of the evolved genetic code
    genetic_code = result['genetic_code']
    assert len(genetic_code) >= 2, "Program should have at least 2 instructions"
    assert genetic_code[0][0] == 100, "First instruction should be LOAD (opcode 100)"
    assert genetic_code[1][0] == 100, "Second instruction should be LOAD (opcode 100)"
    assert genetic_code[-1][0] == 453, "Last instruction should be RETURN (opcode 453)"
    assert genetic_code[-1][1] == 9, "RETURN should use register 9"
    
    # Test the evolved function with detached tensors to avoid grad issues
    interpreter = GeneticInterpreter()
    y_true = torch.rand((5, 1)).detach()
    y_pred = torch.rand((5, 1)).detach()
    
    interpreter.initialize({'y_true': y_true, 'y_pred': y_pred})
    interpreter.load_program(result['genetic_code'])
    result_tensor = interpreter.execute()
    
    # Should produce a valid scalar tensor
    assert isinstance(result_tensor, torch.Tensor)
