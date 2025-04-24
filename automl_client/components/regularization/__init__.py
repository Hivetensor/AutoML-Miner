"""Regularization Component Strategy Package."""

from .strategy import RegularizationStrategy
from ..factory import ComponentStrategyFactory

# Register the strategy with the factory
ComponentStrategyFactory.register_strategy("regularization", RegularizationStrategy)

__all__ = ["RegularizationStrategy"]
