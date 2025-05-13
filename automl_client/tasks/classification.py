"""Classification task strategies for AutoML."""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Any, Callable, Optional
from .strategy import TaskStrategy

logger = logging.getLogger(__name__)

class MNISTClassificationTask(TaskStrategy):
    """MNIST image classification task."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize with optional configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = {
            'num_train_samples': 1000,
            'num_test_samples': 200,
            'input_shape': 28 * 28,  # 784 pixels
            'num_classes': 10,       # 10 digits
            'hidden_size': 64,       # Hidden layer size
            **(config or {})
        }
        logger.info(f"Initialized {self.__class__.__name__} with config: {self.config}")
    
    def prepare_dataset(self) -> Dict[str, torch.Tensor]:
        """
        Prepare MNIST-like dataset.
        
        Returns:
            Dictionary with training and validation tensors
        """
        logger.info("Preparing MNIST-like dataset")
        
        # Extract parameters from config
        num_train_samples = self.config['num_train_samples']
        num_test_samples = self.config['num_test_samples']
        input_shape = self.config['input_shape']
        num_classes = self.config['num_classes']
        
        # Create training data
        train_X = torch.rand((num_train_samples, input_shape))
        train_Y = torch.randint(0, num_classes, (num_train_samples,))
        train_Y = torch.nn.functional.one_hot(train_Y, num_classes).float()
        
        # Create validation data
        val_X = torch.rand((num_test_samples, input_shape))
        val_Y = torch.randint(0, num_classes, (num_test_samples,))
        val_Y = torch.nn.functional.one_hot(val_Y, num_classes).float()
        
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
            Function that creates a PyTorch model for MNIST classification
        """
        def create_model():
            return nn.Sequential(
                nn.Linear(self.config['input_shape'], self.config['hidden_size']),
                nn.ReLU(),
                nn.Linear(self.config['hidden_size'], self.config['num_classes']),
                nn.Softmax(dim=1)
            )
        return create_model
    
    def create_initial_solution(self) -> List:
        """
        Create an initial solution for MNIST classification.
        
        Returns:
            Initial genetic code for classification loss function
        """
        # Create a basic cross-entropy loss function in genetic code
        # This is a simplified version - in practice, you might want to
        # use a more sophisticated initial solution
        return [
            [100, 1, 0],  # LOAD R1, S0 (y_true)
            [100, 2, 1],  # LOAD R2, S1 (y_pred)
            [352, 1, 2, 9],  # BCE R1, R2, R9 (binary cross-entropy)
            [453, 9]  # RETURN R9
        ]
    
    def interpret_result(self, result: Any) -> Dict:
        """
        Interpret the evolved solution.
        
        Args:
            result: Result from evolution
            
        Returns:
            Dictionary with classification-specific interpretation
        """
        return {
            'task_type': 'classification',
            'dataset': 'mnist',
            'metrics': {
                'accuracy': result.get('fitness', 0.0),
                'loss': 1.0 - result.get('fitness', 0.0)
            },
            'config': {
                'input_shape': self.config['input_shape'],
                'num_classes': self.config['num_classes']
            }
        }
    
    def get_task_config(self) -> Dict:
        """
        Get task-specific configuration.
        
        Returns:
            Dictionary with task configuration
        """
        return self.config
