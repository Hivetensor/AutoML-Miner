"""Optimizer Component Strategy Package."""

from .strategy import OptimizerStrategy
from ..factory import ComponentStrategyFactory

# Register the strategy with the factory
ComponentStrategyFactory.register_strategy("optimizer", OptimizerStrategy)

__all__ = ["OptimizerStrategy"]
