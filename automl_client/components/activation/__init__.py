"""Activation Component Strategy Package."""

from .strategy import ActivationStrategy
from ..factory import ComponentStrategyFactory

# Register the strategy with the factory
ComponentStrategyFactory.register_strategy("activation", ActivationStrategy)

__all__ = ["ActivationStrategy"]
