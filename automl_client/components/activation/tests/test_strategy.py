"""Tests for the Activation Component Strategy."""

import unittest
import torch
import numpy as np

# Adjust imports based on actual project structure
# Use absolute imports
from automl_client.tests.utils.base_test import ComponentTestBase, SimpleLinearModel
from ..strategy import ActivationStrategy
from ..library import ActivationLibrary

class ActivationComponentTest(ComponentTestBase):
    """Tests specific to the ActivationStrategy."""

    # Set the strategy class for the base class setUp
    

    def setUp(self):
        """Override setUp to call base and potentially add activation-specific setup."""
        self.strategy_class = ActivationStrategy
        super().setUp()
        #self.strategy = ActivationStrategy({'debug_level': 0})
        # Ensure strategy was instantiated by base class
        if not self.strategy_class:
             self.skipTest("Base class setUp failed to instantiate strategy")

    # --- Specific Tests for ActivationStrategy ---

    def test_create_initial_solution_is_relu(self):
        """Verify the default initial solution is ReLU."""
        initial_solution = self.strategy.create_initial_solution()
        relu_code = ActivationLibrary.get_function("relu") # Assuming 'relu' is the key
        self.assertIsNotNone(relu_code, "ReLU code not found in library")
        # Compare genetic codes (simple list comparison)
        self.assertListEqual(initial_solution, relu_code, "Initial solution should be ReLU")

    def test_evaluate_relu_correctness(self):
        """Test ReLU evaluation with known values."""
        relu_code = ActivationLibrary.get_function("relu")
        self.assertIsNotNone(relu_code, "ReLU code not found in library")

        # Create simple predictable data (input to activation)
        # Note: Activation evaluation might expect input at S0
        input_tensor = torch.tensor([[-1.0], [0.0], [2.5]])
        test_dataset = {'activation_input': input_tensor} # Custom key for activation input

        # Expected ReLU output = [0.0, 0.0, 2.5]
        # Fitness for activation is tricky. Let's assume evaluate returns a dummy value or
        # maybe a measure of non-linearity? For testing correctness, we'll execute
        # the code directly using an interpreter instance.

        # --- Direct Interpreter Execution for Correctness Check ---
        # Use absolute import
        from automl_client.genetic.interpreter import GeneticInterpreter
        interpreter = GeneticInterpreter()
        interpreter.load_program(relu_code)
        interpreter.initialize({'y_true':input_tensor.clone()}) # Assume input at S0
        output_tensor = interpreter.execute()
        interpreter.dispose()

        expected_output = torch.tensor([[0.0], [0.0], [2.5]])
        self.assertIsNotNone(output_tensor, "ReLU code execution failed")
        self.assertTrue(torch.allclose(output_tensor, expected_output, atol=1e-6),
                        f"ReLU output incorrect. Expected {expected_output}, Got {output_tensor}")

        # Test the strategy's evaluate method (basic check inherited from base)
        # This doesn't check correctness, just return type/value
        super().test_evaluate_returns_float_or_inf()


    def test_evaluate_sigmoid_correctness(self):
        """Test Sigmoid evaluation with known values."""
        sigmoid_code = ActivationLibrary.get_function("sigmoid")
        self.assertIsNotNone(sigmoid_code, "Sigmoid code not found in library")

        input_tensor = torch.tensor([[0.0], [-1.0], [1.0]])
        test_dataset = {'activation_input': input_tensor}

        # --- Direct Interpreter Execution ---
        # Use absolute import
        from automl_client.genetic.interpreter import GeneticInterpreter
        interpreter = GeneticInterpreter()
        interpreter.load_program(sigmoid_code)
        interpreter.initialize({'y_true': input_tensor.clone()})
        output_tensor = interpreter.execute()
        interpreter.dispose()

        expected_output = torch.sigmoid(input_tensor)
        
        self.assertIsNotNone(output_tensor, "Sigmoid code execution failed")
        self.assertTrue(torch.allclose(output_tensor, expected_output, atol=1e-6),
                        f"Sigmoid output incorrect. Expected {expected_output}, Got {output_tensor}")

        # Test the strategy's evaluate method (basic check)
        super().test_evaluate_returns_float_or_inf()



    # Add more tests:
    # - Test other activation functions (tanh, etc.)
    # - Test how evaluate handles different input shapes/types if necessary

if __name__ == '__main__':
    unittest.main()
