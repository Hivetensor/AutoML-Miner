"""Task strategy interface for AutoML."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Callable, Optional
import torch

class TaskStrategy(ABC):
    """Interface for task-specific strategies."""
    
    @abstractmethod
    def prepare_dataset(self) -> Dict[str, torch.Tensor]:
        """
        Prepare dataset specific to this task.
        
        Returns:
            Dictionary containing training and validation tensors
        """
        pass
        
    @abstractmethod
    def get_model_factory(self) -> Callable:
        """
        Get a factory function that creates models for this task.
        
        Returns:
            Function that creates a PyTorch model
        """
        pass
        
    @abstractmethod
    def create_initial_solution(self) -> List:
        """
        Create an initial solution for this task.
        
        Returns:
            Initial genetic code for the task
        """
        pass
        
    @abstractmethod
    def interpret_result(self, result: Any) -> Dict:
        """
        Interpret the evolved solution in task-specific terms.
        
        Args:
            result: Result from evolution
            
        Returns:
            Dictionary with task-specific interpretation
        """
        pass
        
    @abstractmethod
    def get_task_config(self) -> Dict:
        """
        Get task-specific configuration.
        
        Returns:
            Dictionary with task configuration
        """
        pass
