"""Tests for the Loss Component Strategy."""

import unittest
import torch
import numpy as np

# Adjust imports based on actual project structure
# Use absolute imports
from automl_client.tests.utils.base_test import ComponentTestBase, SimpleLinearModel
from ..strategy import LossStrategy
from ..library import LossLibrary

class LossComponentTest(ComponentTestBase):
    """Tests specific to the LossStrategy."""

    # Set the strategy class for the base class setUp
    strategy_class = LossStrategy

    def setUp(self):
        """Override setUp to call base and potentially add loss-specific setup."""
        super().setUp()
        # Ensure strategy was instantiated by base class
        if not self.strategy:
             self.skipTest("Base class setUp failed to instantiate strategy")

    # --- Specific Tests for LossStrategy ---

    def test_create_initial_solution_is_mse(self):
        """Verify the default initial solution is MSE."""
        initial_solution = self.strategy.create_initial_solution()
        mse_code = LossLibrary.get_function("mse_loss")
        self.assertIsNotNone(mse_code, "MSE code not found in library")
        # Compare genetic codes (simple list comparison)
        self.assertListEqual(initial_solution, mse_code, "Initial solution should be MSE")

    def test_evaluate_mse_correctness(self):
        """Test MSE evaluation with known values."""
        mse_code = LossLibrary.get_function("mse_loss")
        self.assertIsNotNone(mse_code, "MSE code not found in library")

        # Create simple predictable data
        y_pred = torch.tensor([[2.0], [4.0]])
        y_true = torch.tensor([[1.0], [5.0]])
        test_dataset = {'val_y': y_true, 'val_pred': y_pred} # Provide pre-computed preds

        # Expected MSE = mean([(2-1)^2, (4-5)^2]) = mean([1, 1]) = 1.0
        expected_loss = 1.0
        expected_fitness = -expected_loss

        fitness = self.strategy.evaluate(mse_code, test_dataset, model=None) # No model needed if preds provided
        self.assertIsInstance(fitness, float)
        self.assertAlmostEqual(fitness, expected_fitness, places=5, msg="MSE evaluation incorrect")

    def test_evaluate_mae_correctness(self):
        """Test MAE evaluation with known values."""
        mae_code = LossLibrary.get_function("mae_loss")
        self.assertIsNotNone(mae_code, "MAE code not found in library")

        # Create simple predictable data
        y_pred = torch.tensor([[2.0], [4.0]])
        y_true = torch.tensor([[1.0], [5.0]])
        test_dataset = {'val_y': y_true, 'val_pred': y_pred} # Provide pre-computed preds

        # Expected MAE = mean([abs(2-1), abs(4-5)]) = mean([1, 1]) = 1.0
        expected_loss = 1.0
        expected_fitness = -expected_loss

        fitness = self.strategy.evaluate(mae_code, test_dataset, model=None)
        self.assertIsInstance(fitness, float)
        self.assertAlmostEqual(fitness, expected_fitness, places=5, msg="MAE evaluation incorrect")

    def test_evaluate_bce_correctness(self):
        """Test BCE evaluation with known values (requires careful input)."""
        bce_code = LossLibrary.get_function("bce_loss")
        self.assertIsNotNone(bce_code, "BCE code not found in library")

        # BCE requires inputs between 0 and 1 (logits or probabilities)
        # Let's use probabilities and assume the genetic code handles them directly
        y_pred_prob = torch.tensor([[0.8], [0.3]]) # Probabilities
        y_true_labels = torch.tensor([[1.0], [0.0]]) # True labels (0 or 1)

        # Need 'ones' tensor for BCE calculation in the library code (assumed S5)
        # The interpreter needs to handle loading constants or specific inputs
        # For testing, we might need to mock the interpreter or ensure S5 is loaded.
        # Let's calculate expected BCE manually:
        # -(1 * log(0.8) + (1-1)*log(1-0.8)) = -log(0.8) = 0.2231
        # -(0 * log(0.3) + (1-0)*log(1-0.3)) = -log(0.7) = 0.3567
        # Mean = (0.2231 + 0.3567) / 2 = 0.2899
        expected_loss = 0.2899
        expected_fitness = -expected_loss

        # Prepare dataset - How does the genetic code get the 'ones' tensor?
        # Assuming interpreter handles loading constants or it's passed via special register
        # Let's assume S5 is implicitly '1.0' for the test
        test_dataset = {'val_y': y_true_labels, 'val_pred': y_pred_prob}

        # This test might fail if the genetic code/interpreter assumptions are wrong
        # It highlights the need for clear definition of interpreter inputs/capabilities
        try:
            fitness = self.strategy.evaluate(bce_code, test_dataset, model=None)
            self.assertIsInstance(fitness, float)
            self.assertAlmostEqual(fitness, expected_fitness, places=3, msg="BCE evaluation incorrect")
        except Exception as e:
            self.fail(f"BCE evaluation failed with error: {e}. Check genetic code assumptions (e.g., 'ones' tensor input).")


    def test_interpret_mse(self):
        """Test interpretation of MSE code."""
        mse_code = LossLibrary.get_function("mse_loss")
        dummy_result = {'fitness': -1.0, 'genetic_code': mse_code}
        interpretation = self.strategy.interpret_result(dummy_result)
        self.assertIn("MSE", interpretation.get('notes', [''])[0], "Interpretation should mention MSE")
        # Ideally, check evolved_representation if FunctionConverter is robust

    # Add more tests:
    # - Test evaluation when model is needed to generate predictions
    # - Test edge cases (e.g., zero loss)
    # - Test interpretation of other standard functions (MAE, BCE)

if __name__ == '__main__':
    unittest.main()
