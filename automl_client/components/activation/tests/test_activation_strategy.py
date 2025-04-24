import pytest
import numpy as np
import torch
from ..strategy import ActivationStrategy, DynamicActivation
from ....models.model_factory import ModelFactory

@pytest.fixture
def strategy():
    return ActivationStrategy({
        'batch_size': 32,
        'dataset_name': 'test',
        'training_config': {'iterations': 3}
    })

@pytest.fixture
def synthetic_dataset():
    """Small synthetic dataset for testing"""
    return {
        'trainX': torch.randn(50, 10),
        'trainY': torch.randint(0, 2, (50,)),
        'valX': torch.randn(20, 10),
        'valY': torch.randint(0, 2, (20,))
    }

def test_strategy_initialization(strategy):
    """Test strategy initializes with proper config"""
    assert strategy.activation_config['batch_size'] == 32
    assert strategy.activation_config['training_config']['iterations'] == 3

def test_model_creation(strategy, synthetic_dataset):
    """Test model creation with synthetic dataset"""
    model = strategy.create_model(synthetic_dataset)
    assert isinstance(model, torch.nn.Module)
    assert len(list(model.children())) > 0

def test_evaluation_returns_valid_score(strategy, synthetic_dataset):
    """Test evaluation returns a valid fitness score"""
    # Test with ReLU genetic code
    relu_code = [[100, 1, 0], [101, 2, 0.0], [300, 3, 1, 2], [453, 3]]
    fitness = strategy.evaluate(relu_code, synthetic_dataset)
    assert not np.isnan(fitness)
    assert fitness > -np.inf

def test_empty_genetic_code_handling(strategy, synthetic_dataset):
    """Test evaluation handles empty genetic code"""
    fitness = strategy.evaluate([], synthetic_dataset)
    assert fitness == -np.inf

def test_activation_replacement(strategy, synthetic_dataset):
    """Test activation layers are properly replaced"""
    test_code = [[100, 1, 0], [453, 1]]  # Identity function
    model = strategy.create_model(synthetic_dataset, test_code)
    
    # Count replaced layers
    replaced = sum(1 for m in model.modules() 
                  if isinstance(m, DynamicActivation))
    assert replaced > 0, "No activations were replaced"
