"""Factory for creating AI component strategies."""

from typing import Dict, Type
from .base import AIComponentStrategy
# Import specific strategy classes as they are created
# from .loss.strategy import LossStrategy
# from .optimizer.strategy import OptimizerStrategy
# from .activation.strategy import ActivationStrategy
# ... etc.

class ComponentStrategyFactory:
    """Factory for creating component strategies."""

    _strategies: Dict[str, Type[AIComponentStrategy]] = {
        # Register strategies here as they are implemented
        # "loss": LossStrategy,
        # "optimizer": OptimizerStrategy,
        # "activation": ActivationStrategy,
        # ... etc.
    }

    @classmethod
    def register_strategy(cls, component_type: str, strategy_class: Type[AIComponentStrategy]):
        """Register a new component strategy."""
        if component_type in cls._strategies:
            print(f"Warning: Overwriting existing strategy for component type '{component_type}'")
        cls._strategies[component_type] = strategy_class

    @classmethod
    def create_strategy(cls, component_type: str, config: Dict = None) -> AIComponentStrategy:
        """Create a strategy for the specified component type."""
        strategy_class = cls._strategies.get(component_type)

        if strategy_class is None:
            # Fallback or error handling
            # Option 1: Raise an error
            raise ValueError(f"Unknown or unregistered component type: {component_type}")

            # Option 2: Return a default strategy (if applicable)
            # print(f"Warning: Unknown component type '{component_type}'. Using default strategy.")
            # return DefaultStrategy(config) # Assuming a DefaultStrategy exists

        return strategy_class(config)

    @classmethod
    def list_available_strategies(cls) -> list[str]:
        """List the names of all registered component types."""
        return list(cls._strategies.keys())

# Example of how strategies would be registered (likely in their respective __init__.py)
# from .loss.strategy import LossStrategy
# ComponentStrategyFactory.register_strategy("loss", LossStrategy)
