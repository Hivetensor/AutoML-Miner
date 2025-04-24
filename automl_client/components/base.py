"""Base classes and interfaces for AI component strategies."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import torch

class AIComponentStrategy(ABC):
    """Base class for all AI component strategies."""

    def __init__(self, config: Dict = None):
        """Initialize with optional configuration."""
        self.config = config or {}

    @abstractmethod
    def create_initial_solution(self) -> List:
        """Create an initial genetic code solution for this component."""
        pass

    @abstractmethod
    def evaluate(self, genetic_code: List, dataset: Dict, model: torch.nn.Module = None) -> float:
        """
        Evaluate a genetic solution for this component.

        Args:
            genetic_code: The genetic code representing the component.
            dataset: Dictionary containing training/validation data tensors.
            model: Optional PyTorch model instance if evaluation requires it.

        Returns:
            Fitness score (higher is better).
        """
        pass

    @abstractmethod
    def interpret_result(self, result: Dict) -> Dict:
        """
        Interpret the final evolved result for this component.

        Args:
            result: Dictionary containing evolution results (fitness, genetic_code, etc.).

        Returns:
            Dictionary with component-specific interpretation.
        """
        pass

    def get_component_config(self) -> Dict:
        """Get configuration specific to this component strategy."""
        return self.config.get("component_config", {})

    # Optional: Add common utility methods if needed across strategies
    # def common_utility(self, ...):
    #     pass
