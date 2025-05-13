"""Base class for testing AI Component Strategies."""

import unittest
import torch
import numpy as np
from typing import List, Dict, Any

# Assume AIComponentStrategy is the base class for all strategies
# Adjust the import path based on the actual location of base.py
from ...components.base import AIComponentStrategy

# A simple model for testing purposes
class SimpleLinearModel(torch.nn.Module):
    def __init__(self, input_dim=10, output_dim=1):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class ComponentTestBase(unittest.TestCase):
    """Base class providing common setup and tests for component strategies."""

    strategy_class: type[AIComponentStrategy] = None # Subclasses should override this

    def setUp(self):
        """Set up common test data and configurations."""
        self.input_dim = 5
        self.output_dim = 1
        self.batch_size = 8

        # Sample data
        self.x_train = torch.randn(self.batch_size, self.input_dim)
        self.y_train = torch.randn(self.batch_size, self.output_dim)
        self.x_val = torch.randn(self.batch_size // 2, self.input_dim)
        self.y_val = torch.randn(self.batch_size // 2, self.output_dim)

        # Sample dataset dictionary
        self.dataset = {
            'trainX': self.x_train,
            'trainY': self.y_train,
            'valX': self.x_val,
            'valY': self.y_val,
            # Add other potential keys strategies might need
            'base_loss': torch.tensor(1.0), # Example for regularization
        }

        # Sample model
        self.model = SimpleLinearModel(self.input_dim, self.output_dim)
        # Add model weights to dataset (example for regularization)
        self.dataset['model_weights'] = [p.data.clone() for p in self.model.parameters()]


        # Default config
        self.config = {
            'debug_level': 0,
            # Add other common config keys if needed
        }

        # Instantiate the strategy (subclass must define strategy_class)
        
        if self.strategy_class:
            self.strategy = self.strategy_class(self.config)
        else:
            self.strategy = None
        


    def test_instantiation(self):
        """Test if the strategy can be instantiated."""
        if not self.strategy_class:
             self.skipTest("strategy_class not defined in subclass")
        self.assertIsNotNone(self.strategy, "Strategy instantiation failed")
        self.assertIsInstance(self.strategy, AIComponentStrategy)

    def test_create_initial_solution_not_empty(self):
        """Test that create_initial_solution returns a non-empty list."""
        if not self.strategy:
             self.skipTest("Strategy not instantiated")
        solution = self.strategy.create_initial_solution()
        self.assertIsInstance(solution, list, "Initial solution should be a list")
        self.assertTrue(len(solution) > 0, "Initial solution should not be empty")
        # Basic check for instruction format (list of lists)
        if solution:
            self.assertIsInstance(solution[0], list, "Instructions should be lists")

    def test_evaluate_returns_float_or_inf(self):
        """Test that evaluate returns a float or +/- infinity."""
        if not self.strategy:
             self.skipTest("Strategy not instantiated")
        # Use the initial solution for a basic evaluation test
        initial_solution = self.strategy.create_initial_solution()
        if not initial_solution:
             self.skipTest("Could not create initial solution for evaluation test")

        # Provide necessary inputs (dataset, potentially model)
        fitness = self.strategy.evaluate(initial_solution, self.dataset, self.model)
        self.assertTrue(isinstance(fitness, float), f"Evaluate should return a float, got {type(fitness)}")
        # Allow -inf for errors, but not NaN
        self.assertFalse(np.isnan(fitness), "Evaluate should not return NaN")


    def test_evaluate_handles_bad_code(self):
        """Test that evaluate handles None or empty genetic code gracefully."""
        if not self.strategy:
             self.skipTest("Strategy not instantiated")
        fitness_none = self.strategy.evaluate(None, self.dataset, self.model)
        self.assertEqual(fitness_none, -np.inf, "Evaluate with None code should return -inf")

        fitness_empty = self.strategy.evaluate([], self.dataset, self.model)
        self.assertEqual(fitness_empty, -np.inf, "Evaluate with empty code should return -inf")

    def test_interpret_result_structure(self):
        """Test the basic structure of the interpret_result output."""
        if not self.strategy:
             self.skipTest("Strategy not instantiated")
        # Create a dummy result dictionary
        dummy_result = {
            'fitness': -1.23,
            'genetic_code': self.strategy.create_initial_solution(),
            # Add other potential keys from EvolutionEngine result
        }
        interpretation = self.strategy.interpret_result(dummy_result)
        self.assertIsInstance(interpretation, dict, "Interpretation should be a dictionary")
        self.assertIn('component_type', interpretation, "Interpretation missing 'component_type'")
        self.assertIn('evolved_representation', interpretation, "Interpretation missing 'evolved_representation'")
        self.assertIn('notes', interpretation, "Interpretation missing 'notes'")
        self.assertIsInstance(interpretation['notes'], list, "'notes' should be a list")

    def test_get_component_config_structure(self):
        """Test that get_component_config returns a dictionary."""
        if not self.strategy:
             self.skipTest("Strategy not instantiated")
        config = self.strategy.get_component_config()
        self.assertIsInstance(config, dict, "get_component_config should return a dictionary")
