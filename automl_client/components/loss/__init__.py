"""Loss Component Strategy Package."""

from .strategy import LossStrategy
from ..factory import ComponentStrategyFactory

# Register the strategy with the factory
ComponentStrategyFactory.register_strategy("loss", LossStrategy)

__all__ = ["LossStrategy"]
