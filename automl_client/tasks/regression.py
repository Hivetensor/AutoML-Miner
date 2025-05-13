"""Regression task strategies for AutoML."""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Any, Callable, Optional
from .strategy import TaskStrategy

logger = logging.getLogger(__name__)

class SimpleRegressionTask(TaskStrategy):
    """Simple regression task."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize with optional configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = {
            'num_train_samples': 1000,
            'num_test_samples': 200,
            'input_dim': 10,
            'output_dim': 1,
            'hidden_size': 32,
            'noise_level': 0.1,
            **(config or {})
        }
        logger.info(f"Initialized {self.__class__.__name__} with config: {self.config}")
    
    def prepare_dataset(self) -> Dict[str, torch.Tensor]:
        """
        Prepare regression dataset with linear relationship and noise.
        
        Returns:
            Dictionary with training and validation tensors
        """
        logger.info("Preparing regression dataset")
        
        # Extract parameters from config
        num_train_samples = self.config['num_train_samples']
        num_test_samples = self.config['num_test_samples']
        input_dim = self.config['input_dim']
        output_dim = self.config['output_dim']
        noise_level = self.config['noise_level']
        
        # Create a simple linear relationship with noise
        weights = torch.randn(input_dim, output_dim)
        bias = torch.randn(output_dim)
        
        # Training data
        train_X = torch.randn(num_train_samples, input_dim)
        train_Y = torch.matmul(train_X, weights) + bias + noise_level * torch.randn(num_train_samples, output_dim)
        
        # Validation data
        val_X = torch.randn(num_test_samples, input_dim)
        val_Y = torch.matmul(val_X, weights) + bias + noise_level * torch.randn(num_test_samples, output_dim)
        
        # Log dataset details
        logger.info(f"Dataset created: {num_train_samples} training samples, {num_test_samples} validation samples")
        logger.debug(f"Training data shapes: X={train_X.shape}, Y={train_Y.shape}")
        logger.debug(f"Validation data shapes: X={val_X.shape}, Y={val_Y.shape}")
        
        return {
            'trainX': train_X,
            'trainY': train_Y,
            'valX': val_X,
            'valY': val_Y
        }
    
    def get_model_factory(self) -> Callable:
        """
        Get a factory function for creating models.
        
        Returns:
            Function that creates a PyTorch model for regression
        """
        def create_model():
            return nn.Sequential(
                nn.Linear(self.config['input_dim'], self.config['hidden_size']),
                nn.ReLU(),
                nn.Linear(self.config['hidden_size'], self.config['output_dim'])
            )
        return create_model
    
    def create_initial_solution(self) -> List:
        """
        Create an initial solution for regression.
        
        Returns:
            Initial genetic code for regression loss function
        """
        # Create a basic MSE loss function in genetic code
        return [
            [100, 1, 0],  # LOAD R1, S0 (y_true)
            [100, 2, 1],  # LOAD R2, S1 (y_pred)
            [350, 1, 2, 9],  # MSE R1, R2, R9 (mean squared error)
            [453, 9]  # RETURN R9
        ]
    
    def interpret_result(self, result: Any) -> Dict:
        """
        Interpret the evolved solution.
        
        Args:
            result: Result from evolution
            
        Returns:
            Dictionary with regression-specific interpretation
        """
        # For regression, fitness is typically inverse of MSE
        # So we convert it back to MSE for reporting
        mse = 1.0 - result.get('fitness', 0.0) if result.get('fitness', 0.0) < 1.0 else 0.0
        
        return {
            'task_type': 'regression',
            'metrics': {
                'mse': mse,
                'rmse': mse ** 0.5 if mse > 0 else 0.0,
                'fitness': result.get('fitness', 0.0)
            },
            'config': {
                'input_dim': self.config['input_dim'],
                'output_dim': self.config['output_dim'],
                'noise_level': self.config['noise_level']
            }
        }
    
    def get_task_config(self) -> Dict:
        """
        Get task-specific configuration.
        
        Returns:
            Dictionary with task configuration
        """
        return self.config
