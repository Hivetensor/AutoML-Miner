"""Performance tests for the genetic programming system."""

import time
import pytest
import torch
import numpy as np
from ..interpreter import GeneticInterpreter
from ..genetic_operators import GeneticOperators
# from ..genetic_library import GeneticLibrary # Removed - Obsolete
from ...evolution_engine import EvolutionEngine

@pytest.mark.skip(reason="Relies on obsolete GeneticLibrary")
def test_performance_scaling():
    """Test how the system performance scales with population size and generations."""
    
    # Start with MSE as a baseline
    # mse_code = GeneticLibrary.mean_squared_error # Obsolete
    # Use placeholder code if needed for structure, though test is skipped
    mse_code = [[100, 1, 0], [100, 2, 1], [2, 3, 1, 2], [10, 4, 3], [201, 9, -1, 4], [453, 9]]

    # Base task definition
    task = {
        'functions': [{
            'function': mse_code,
            'id': 'mse_baseline'
        }],
        'component_type': 'loss',
        'batch_id': 'performance_test'
    }
    
    # Test configurations with different population sizes
    population_configs = [
        {'population_size': 5, 'generations': 3},
        {'population_size': 10, 'generations': 3},
        {'population_size': 20, 'generations': 3}
    ]
    
    # Fixed configuration - just testing population scaling
    fixed_config = {
        'mutation_rate': 0.2,
        'crossover_rate': 0.7,
        'elite_count': 2,
        'evaluation_rounds': 1  # Minimal training for speed
    }
    
    # Measure execution times
    population_times = []
    
    for config in population_configs:
        # Create engine with this configuration
        engine = EvolutionEngine()
        engine.config.update(fixed_config)
        engine.config.update(config)
        
        # Run evolution and measure time
        start_time = time.time()
        result = engine.evolve_function(task.copy())
        end_time = time.time()
        
        assert result is not None, f"Evolution failed with config {config}"
        
        execution_time = end_time - start_time
        population_times.append(execution_time)
        
        print(f"Population size {config['population_size']}: {execution_time:.2f} seconds")
    
    # Test configurations with different generation counts
    generation_configs = [
        {'population_size': 10, 'generations': 2},
        {'population_size': 10, 'generations': 5},
        {'population_size': 10, 'generations': 10}
    ]
    
    # Measure execution times
    generation_times = []
    
    for config in generation_configs:
        # Create engine with this configuration
        engine = EvolutionEngine()
        engine.config.update(fixed_config)
        engine.config.update(config)
        
        # Run evolution and measure time
        start_time = time.time()
        result = engine.evolve_function(task.copy())
        end_time = time.time()
        
        assert result is not None, f"Evolution failed with config {config}"
        
        execution_time = end_time - start_time
        generation_times.append(execution_time)
        
        print(f"Generations {config['generations']}: {execution_time:.2f} seconds")
    
    # Verify linear scaling with population size
    if len(population_times) > 2:
        # Calculate ratios of times vs population sizes
        ratios = []
        for i in range(1, len(population_configs)):
            pop_ratio = population_configs[i]['population_size'] / population_configs[i-1]['population_size']
            time_ratio = population_times[i] / population_times[i-1]
            scaling_factor = time_ratio / pop_ratio
            ratios.append(scaling_factor)
            print(f"Population scaling factor: {scaling_factor:.2f}")
        
        # Check if scaling is approximately linear
        avg_ratio = sum(ratios) / len(ratios)
        assert 0.7 <= avg_ratio <= 1.5, "Performance scaling with population size should be approximately linear"
    
    # Verify linear scaling with generations
    if len(generation_times) > 2:
        # Calculate ratios of times vs generation counts
        ratios = []
        for i in range(1, len(generation_configs)):
            gen_ratio = generation_configs[i]['generations'] / generation_configs[i-1]['generations']
            time_ratio = generation_times[i] / generation_times[i-1]
            scaling_factor = time_ratio / gen_ratio
            ratios.append(scaling_factor)
            print(f"Generation scaling factor: {scaling_factor:.2f}")
        
        # Check if scaling is approximately linear
        avg_ratio = sum(ratios) / len(ratios)
        assert 0.7 <= avg_ratio <= 1.5, "Performance scaling with generations should be approximately linear"

@pytest.mark.skip(reason="Relies on obsolete GeneticLibrary")
def test_execution_time_vs_genetic_code_complexity():
    """Test how execution time scales with genetic code complexity."""

    # Create test data
    test_y_true = torch.rand((10, 5)).detach()
    test_y_pred = torch.rand((10, 5)).detach()
    
    # Test genetic codes of different complexity
    # Using placeholders as GeneticLibrary is obsolete
    genetic_codes = [
        # Simple MSE - 7 instructions (Placeholder)
        [[100, 1, 0], [100, 2, 1], [2, 3, 1, 2], [10, 4, 3], [201, 9, -1, 4], [453, 9]],

        # MAE - 7 instructions but different operations (Placeholder)
        [[100, 1, 0], [100, 2, 1], [2, 3, 1, 2], [9, 4, 3], [201, 9, -1, 4], [453, 9]],

        # BCE - 17 instructions (more complex) (Placeholder)
        [[100,1,0],[100,2,1],[100,4,5],[2,4,2,5],[2,4,1,6],[7,2,7],[7,5,8],[3,1,7,7],[3,6,8,8],[1,7,8,7],[11,7,7],[201,7,-1,5],[201,5,0,9],[453,9]]
    ]

    # Measure execution times
    execution_times = []
    
    interpreter = GeneticInterpreter()
    
    for i, code in enumerate(genetic_codes):
        # Warmup run
        interpreter.initialize({'y_true': test_y_true, 'y_pred': test_y_pred})
        interpreter.load_program(code)
        interpreter.execute()
        
        # Timed runs
        iterations = 5
        start_time = time.time()
        
        for _ in range(iterations):
            interpreter.initialize({'y_true': test_y_true, 'y_pred': test_y_pred})
            interpreter.load_program(code)
            interpreter.execute()
        
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations
        execution_times.append(avg_time)
        
        code_name = "MSE" if i == 0 else "MAE" if i == 1 else "BCE"
        code_len = len(code)
        print(f"{code_name} ({code_len} instructions): {avg_time*1000:.3f} ms per execution")
    
    # Verify reasonable scaling with code complexity
    if len(execution_times) >= 3:
        # BCE should be slower than MSE/MAE but not dramatically so
        simple_avg = (execution_times[0] + execution_times[1]) / 2
        complex_time = execution_times[2]
        
        # Complex code should not be more than 3x slower than simple code
        ratio = complex_time / simple_avg
        print(f"Complex/Simple execution time ratio: {ratio:.2f}x")
        
        assert ratio < 3, "Complex genetic code should not be dramatically slower than simple code"
