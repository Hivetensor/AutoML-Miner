import torch
import pytest
from ..strategy import DynamicActivation
from ....genetic.interpreter import GeneticInterpreter

@pytest.fixture
def relu_code():
    return [[100, 1, 0], [101, 2, 0.0], [300, 1, 2, 3], [453, 3]]

def test_activation_initialization(relu_code):
    """Test DynamicActivation initializes with valid genetic code"""
    activation = DynamicActivation(relu_code)
    assert activation.genetic_code == relu_code
    assert isinstance(activation.interpreter, GeneticInterpreter)

def test_forward_pass(relu_code):
    """Test forward pass with simple input tensor"""
    activation = DynamicActivation(relu_code)
    input_tensor = torch.tensor([-1.0, 0.0, 2.0])
    output = activation(input_tensor)
    expected = torch.nn.functional.relu(input_tensor)
    assert torch.allclose(output, expected, atol=1e-4)

