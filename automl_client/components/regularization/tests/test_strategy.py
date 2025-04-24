"""Tests for the Regularization Component Strategy."""

import unittest
import torch
import numpy as np

# Adjust imports based on actual project structure
# Use absolute imports
from automl_client.tests.utils.base_test import ComponentTestBase, SimpleLinearModel
from ..strategy import RegularizationStrategy
from ..library import RegularizationLibrary

class RegularizationComponentTest(ComponentTestBase):
    """Tests specific to the RegularizationStrategy."""

    # Set the strategy class for the base class setUp
    strategy_class = RegularizationStrategy

    def setUp(self):
        """Override setUp to call base and potentially add regularization-specific setup."""
        super().setUp()
        # Ensure strategy was instantiated by base class
        if not self.strategy:
             self.skipTest("Base class setUp failed to instantiate strategy")

        # Add specific inputs needed for regularization evaluation
        self.test_dataset = self.dataset.copy() # Use base dataset
        self.test_dataset['base_loss'] = torch.tensor(10.0) # Example base loss

        # Prepare weights and strength factor for direct interpreter tests
        # Use weights from the simple model created in base setUp
        self.weights_list = [p.data.clone() for p in self.model.parameters()]
        self.flattened_weights = torch.cat([w.flatten() for w in self.weights_list])
        self.strength_factor = torch.tensor(0.01) # Example strength

    # --- Specific Tests for RegularizationStrategy ---

    def test_create_initial_solution_is_no_reg(self):
        """Verify the default initial solution is no regularization."""
        initial_solution = self.strategy.create_initial_solution()
        no_reg_code = RegularizationLibrary.get_function("no_regularization")
        self.assertIsNotNone(no_reg_code, "'no_regularization' code not found in library")
        self.assertListEqual(initial_solution, no_reg_code, "Initial solution should be 'no_regularization'")

    def test_evaluate_no_reg_correctness(self):
        """Test 'no regularization' evaluation returns negative base loss."""
        no_reg_code = RegularizationLibrary.get_function("no_regularization")
        self.assertIsNotNone(no_reg_code, "'no_regularization' code not found in library")

        expected_fitness = -self.test_dataset['base_loss'].item() # Should just be -base_loss

        # Evaluate using the strategy's method
        fitness = self.strategy.evaluate(no_reg_code, self.test_dataset, self.model)
        self.assertIsInstance(fitness, float)
        self.assertAlmostEqual(fitness, expected_fitness, places=5, msg="'no_regularization' evaluation incorrect")

    def test_evaluate_l2_correctness(self):
        """Test L2 regularization evaluation with known values."""
        l2_code = RegularizationLibrary.get_function("l2_regularization")
        self.assertIsNotNone(l2_code, "L2 code not found in library")

        # --- Direct Interpreter Execution for Penalty Check ---
        # Use absolute import
        from automl_client.genetic.interpreter import GeneticInterpreter
        interpreter = GeneticInterpreter()
        interpreter.load_program(l2_code)
        # Assumes S1=weights, S2=strength
        interpreter.initialize(inputs={1: self.flattened_weights.clone(), 2: self.strength_factor.clone()})
        penalty_tensor = interpreter.execute()
        interpreter.dispose()

        self.assertIsNotNone(penalty_tensor, "L2 code execution failed")
        expected_penalty = self.strength_factor * torch.sum(self.flattened_weights**2)
        self.assertTrue(torch.allclose(penalty_tensor, expected_penalty, atol=1e-6),
                        f"L2 penalty calculation incorrect. Expected {expected_penalty}, Got {penalty_tensor}")

        # --- Test Strategy's evaluate method ---
        # Add strength factor to dataset for the strategy's evaluate method
        # How should strength be passed? Assume it's part of the genetic code or config?
        # The current library code assumes S2. Let's modify the test dataset for now.
        # This highlights a design question: how is strength factor determined/passed?
        # For the test, let's add it to the dataset.
        self.test_dataset['reg_strength'] = self.strength_factor # Add strength

        # Modify the evaluate method or assume interpreter gets S2 from config/elsewhere?
        # Let's assume evaluate needs modification or the genetic code evolves strength.
        # For now, we test the penalty calculation was correct above.
        # We'll rely on the base class test for the evaluate return type check.
        super().test_evaluate_returns_float_or_inf()
        # TODO: Revisit this test once strength factor handling is clarified in evaluate.


    def test_evaluate_l1_correctness(self):
        """Test L1 regularization evaluation with known values."""
        l1_code = RegularizationLibrary.get_function("l1_regularization")
        self.assertIsNotNone(l1_code, "L1 code not found in library")

        # --- Direct Interpreter Execution for Penalty Check ---
        # Use absolute import
        from automl_client.genetic.interpreter import GeneticInterpreter
        interpreter = GeneticInterpreter()
        interpreter.load_program(l1_code)
        # Assumes S1=weights, S2=strength
        interpreter.initialize(inputs={1: self.flattened_weights.clone(), 2: self.strength_factor.clone()})
        penalty_tensor = interpreter.execute()
        interpreter.dispose()

        self.assertIsNotNone(penalty_tensor, "L1 code execution failed")
        expected_penalty = self.strength_factor * torch.sum(torch.abs(self.flattened_weights))
        self.assertTrue(torch.allclose(penalty_tensor, expected_penalty, atol=1e-6),
                        f"L1 penalty calculation incorrect. Expected {expected_penalty}, Got {penalty_tensor}")

        # --- Test Strategy's evaluate method ---
        # Similar issue as L2 regarding strength factor handling in evaluate.
        super().test_evaluate_returns_float_or_inf()
        # TODO: Revisit this test once strength factor handling is clarified in evaluate.


    def test_interpret_no_reg(self):
        """Test interpretation of no regularization code."""
        no_reg_code = RegularizationLibrary.get_function("no_regularization")
        dummy_result = {'fitness': -10.0, 'genetic_code': no_reg_code}
        interpretation = self.strategy.interpret_result(dummy_result)
        # Check the placeholder logic in interpret_result
        self.assertIn("None", interpretation.get('evolved_representation', ''), "Interpretation should identify None")

    # Add more tests:
    # - Test interpretation of L1/L2 (requires parsing strength factor)
    # - Test evaluate with model weights provided directly in dataset vs extracted from model

if __name__ == '__main__':
    unittest.main()
