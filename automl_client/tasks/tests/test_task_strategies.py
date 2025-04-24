"""Tests for task strategies."""

import unittest
import torch
from automl_client.tasks.strategy import TaskStrategy
from automl_client.tasks.classification import MNISTClassificationTask
from automl_client.tasks.regression import SimpleRegressionTask

class TestTaskStrategies(unittest.TestCase):
    """Test cases for task strategies."""
    
    def test_mnist_classification_task(self):
        """Test MNISTClassificationTask."""
        # Create task with default config
        task = MNISTClassificationTask()
        
        # Test prepare_dataset
        dataset = task.prepare_dataset()
        self.assertIn('trainX', dataset)
        self.assertIn('trainY', dataset)
        self.assertIn('valX', dataset)
        self.assertIn('valY', dataset)
        
        # Check shapes
        self.assertEqual(dataset['trainX'].shape[0], 1000)  # Default num_train_samples
        self.assertEqual(dataset['trainX'].shape[1], 28*28)  # Default input_shape
        self.assertEqual(dataset['trainY'].shape[0], 1000)  # Default num_train_samples
        self.assertEqual(dataset['trainY'].shape[1], 10)  # Default num_classes
        
        # Test get_model_factory
        model_factory = task.get_model_factory()
        model = model_factory()
        self.assertIsInstance(model, torch.nn.Sequential)
        
        # Test create_initial_solution
        solution = task.create_initial_solution()
        self.assertIsInstance(solution, list)
        self.assertGreater(len(solution), 0)
        
        # Test interpret_result
        result = {
            'fitness': 0.85,
            'genetic_code': solution
        }
        interpretation = task.interpret_result(result)
        self.assertEqual(interpretation['task_type'], 'classification')
        self.assertEqual(interpretation['dataset'], 'mnist')
        self.assertIn('metrics', interpretation)
        self.assertIn('accuracy', interpretation['metrics'])
        
        # Test get_task_config
        config = task.get_task_config()
        self.assertIsInstance(config, dict)
        self.assertEqual(config['num_train_samples'], 1000)
        self.assertEqual(config['num_classes'], 10)
    
    def test_regression_task(self):
        """Test SimpleRegressionTask."""
        # Create task with custom config
        custom_config = {
            'num_train_samples': 500,
            'input_dim': 5,
            'output_dim': 2,
            'noise_level': 0.1
        }
        task = SimpleRegressionTask(custom_config)
        
        # Test prepare_dataset
        dataset = task.prepare_dataset()
        self.assertIn('trainX', dataset)
        self.assertIn('trainY', dataset)
        self.assertIn('valX', dataset)
        self.assertIn('valY', dataset)
        
        # Check shapes match custom config
        self.assertEqual(dataset['trainX'].shape[0], 500)  # Custom num_train_samples
        self.assertEqual(dataset['trainX'].shape[1], 5)  # Custom input_dim
        self.assertEqual(dataset['trainY'].shape[0], 500)  # Custom num_train_samples
        self.assertEqual(dataset['trainY'].shape[1], 2)  # Custom output_dim
        
        # Test get_model_factory
        model_factory = task.get_model_factory()
        model = model_factory()
        self.assertIsInstance(model, torch.nn.Sequential)
        
        # Test create_initial_solution
        solution = task.create_initial_solution()
        self.assertIsInstance(solution, list)
        self.assertGreater(len(solution), 0)
        
        # Test interpret_result
        result = {
            'fitness': 0.75,
            'genetic_code': solution
        }
        interpretation = task.interpret_result(result)
        self.assertEqual(interpretation['task_type'], 'regression')
        self.assertIn('metrics', interpretation)
        self.assertIn('mse', interpretation['metrics'])
        self.assertIn('rmse', interpretation['metrics'])
        
        # Test get_task_config
        config = task.get_task_config()
        self.assertIsInstance(config, dict)
        self.assertEqual(config['num_train_samples'], 500)
        self.assertEqual(config['input_dim'], 5)
        self.assertEqual(config['output_dim'], 2)
        self.assertEqual(config['noise_level'], 0.1)

if __name__ == '__main__':
    unittest.main()
