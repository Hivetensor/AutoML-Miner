"""Tests for Genetic Interpreter in PyTorch."""

import pytest
import torch
import numpy as np
from ..interpreter import GeneticInterpreter

def test_scalar_operations():
    """Test basic scalar operations."""
    interpreter = GeneticInterpreter()
    
    # Input tensors (scalars)
    a = torch.tensor(5.0)
    b = torch.tensor(3.0)
    
    # Simple ADD program
    add_program = [
    [100, 1, 0],    # LOAD from special register 0 (y_true) to register 1
    [100, 2, 1],    # LOAD from special register 1 (y_pred) to register 2
    [1, 1, 2, 9],   # ADD register 1 and register 2, store in register 9 (OUTPUT)
    [453, 9]        # RETURN register 9
    ]
    
    interpreter.initialize({'y_true': a, 'y_pred': b})
    interpreter.load_program(add_program)
    result = interpreter.execute()
    
    # Verify result
    expected = a + b
    assert torch.allclose(result, expected)

def test_mse_genetic_code():
    """Test execution of MSE genetic code."""
    interpreter = GeneticInterpreter()
    
    # Create test tensor data
    y_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_pred = torch.tensor([[1.1, 1.9], [2.8, 4.2]])
    
    # MSE genetic code
    mse_code = [
        [100, 1, 0],      # LOAD R1, S0 (y_true)
        [100, 2, 1],      # LOAD R2, S1 (y_pred)
        [2, 1, 2, 3],     # SUB R3, R1, R2
        [10, 3, 4],       # SQUARE R4, R3
        [201, 4, -1, 5],  # REDUCE_MEAN R4, axis=-1, R5 (reduce last dimension)
        [201, 5, 0, 9],   # REDUCE_MEAN R5, axis=0, R9 (reduce remaining dimension)
        [453, 9]          # RETURN R9
    ]
    
    interpreter.initialize({'y_true': y_true, 'y_pred': y_pred})
    interpreter.load_program(mse_code)
    result = interpreter.execute()
    
    # Calculate expected MSE manually
    diff = y_true - y_pred
    squared_diff = diff ** 2
    expected = torch.mean(squared_diff)
    
    assert torch.allclose(result, expected)

def test_mae_genetic_code():
    """Test execution of MAE genetic code."""
    interpreter = GeneticInterpreter()
    
    # Create test tensor data
    y_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_pred = torch.tensor([[1.1, 1.9], [2.8, 4.2]])
    
    # MAE genetic code
    mae_code = [
        [100, 1, 0],     # LOAD R1, S0 (y_true)
        [100, 2, 1],     # LOAD R2, S1 (y_pred)
        [2, 2, 1, 3],    # SUB R3, R1, R2 (calculates y_true - y_pred)
        [9, 3, 4],       # ABS R4, R3 (calculates abs(y_true - y_pred))
        [201, 4, -1, 5], # REDUCE_MEAN R5, R4, axis=-1 (mean along last dim)
        [201, 5, 0, 9],  # REDUCE_MEAN R9, R5, axis=0 (mean across all remaining dims)
        [453, 9]         # RETURN R9
    ]


    
    interpreter.initialize({'y_true': y_true, 'y_pred': y_pred})
    interpreter.load_program(mae_code)
    result = interpreter.execute()
    
    # Calculate expected MAE manually
    diff = y_true - y_pred
    abs_diff = torch.abs(diff)
    expected = torch.mean(abs_diff)
    
    assert torch.allclose(result, expected)

def test_tensor_operations():
    """Test basic tensor operations."""
    interpreter = GeneticInterpreter()
    
    # Input tensors
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    
    # ADD program for tensors
    add_program = [
        [100, 1, 0],  # LOAD R1, S0 (a)
        [100, 2, 1],  # LOAD R2, S1 (b)
        [1, 1, 2, 3], # ADD R3, R1, R2
        [453, 3]      # RETURN R3
    ]
    
    interpreter.initialize({'y_true': a, 'y_pred': b})
    interpreter.load_program(add_program)
    result = interpreter.execute()
    
    # Verify result
    expected = a + b
    assert torch.allclose(result, expected)

def test_decode_to_function():
    """Test conversion of genetic code to Python function."""
    # MAE genetic code
    mae_code = [
        [100, 1, 0],     # LOAD R1, S0 (y_true)
        [100, 2, 1],     # LOAD R2, S1 (y_pred)
        [2, 1, 2, 3],    # SUB R3, R1, R2
        [9, 3, 4],       # ABS R4, R3
        [201, 4, -1, 9], # REDUCE_MEAN R9, R4, axis=-1
        [453, 9]         # RETURN R9
    ]
    
    # Convert to function
    mae_fn = GeneticInterpreter.decode_to_function(mae_code)
    
    # Test with sample inputs
    y_true = torch.tensor([1.0, 2.0, 3.0])
    y_pred = torch.tensor([1.1, 1.9, 2.8])
    
    result = mae_fn(y_true, y_pred)
    expected = torch.mean(torch.abs(y_true - y_pred))
    
    assert torch.allclose(result, expected)
