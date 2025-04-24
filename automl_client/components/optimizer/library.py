"""Genetic Library for standard Optimizer components."""

from typing import List, Dict, Optional

class OptimizerLibrary:
    """Provides genetic code representations for common optimizers."""

    _library: Dict[str, List] = {
        "sgd": [
            # Simplified SGD: param = param - lr * grad
            # Assumes S0=params, S1=grads, S2=learning_rate passed to interpreter
            # NOTE: This is highly simplified. Real SGD needs iteration over parameters
            # and a mechanism to apply updates back to the model.
            # This code represents a single parameter update step conceptually.
            [100, 1, 0],      # LOAD R1, S0 (current parameter value)
            [100, 2, 1],      # LOAD R2, S1 (gradient for this parameter)
            [100, 3, 2],      # LOAD R3, S2 (learning_rate)
            [4, 4, 3, 2],     # MULTIPLY R4, R3, R2 (learning_rate * gradient)
            [2, 5, 1, 4],     # SUBTRACT R5, R1, R4 (param - lr * grad) -> New parameter value
            # --- How to apply this update back to the model parameter? ---
            # Option 1: A dedicated opcode updates the parameter associated with S0
            # [450, 0, 5],    # Hypothetical UPDATE_PARAM S0, R5
            # Option 2: Return the updated value, strategy handles application
            [453, 5]          # RETURN R5 (the new parameter value)
        ],
        # TODO: Add Adam, RMSprop etc. These are significantly more complex
        # due to maintaining state (momentum, variance estimates) across steps.
        # Representing them genetically might require more advanced opcodes or
        # a different approach where genetic code configures a standard optimizer.
    }

    @staticmethod
    def get_function(name: str) -> Optional[List]:
        """
        Retrieve the genetic code for a standard optimizer by name.

        Args:
            name: The name of the optimizer (e.g., "sgd").

        Returns:
            The genetic code as a list of lists, or None if not found.
        """
        return OptimizerLibrary._library.get(name)

    @staticmethod
    def list_functions() -> List[str]:
        """Return a list of available standard optimizer names."""
        return list(OptimizerLibrary._library.keys())
