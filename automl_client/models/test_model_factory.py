import torch
import pytest
from automl_client.models.model_factory import ModelFactory

def test_model_determinism():
    """Test that models created with the same seed have identical parameters."""
    # Test with different component types
    for component_type in ["loss", "optimizer", "activation", "regularization"]:
        # Create models with same seed
        seed1 = 42
        seed2 = 123
        
        # Create first model with seed1
        model1 = ModelFactory.create_model(component_type, {"model_seed": seed1})
        
        # Create second model with same seed
        model2 = ModelFactory.create_model(component_type, {"model_seed": seed1})
        
        # Verify parameter equality between models with same seed
        for (p1, p2) in zip(model1.parameters(), model2.parameters()):
            assert torch.all(torch.eq(p1, p2)), f"Parameters differ for {component_type} models with same seed"
        
        # Create third model with different seed
        model3 = ModelFactory.create_model(component_type, {"model_seed": seed2})
        
        # Verify parameters differ for models with different seeds
        param_differences_found = False
        for (p1, p3) in zip(model1.parameters(), model3.parameters()):
            if not torch.all(torch.eq(p1, p3)):
                param_differences_found = True
                break
        
        assert param_differences_found, f"No parameter differences found for {component_type} models with different seeds"

if __name__ == "__main__":
    pytest.main([__file__])
