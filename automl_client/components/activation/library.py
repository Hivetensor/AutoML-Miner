"""Genetic Library for standard Activation Function components."""

from typing import List, Dict, Optional

class ActivationLibrary:
    """Provides genetic code representations for common activation functions."""

    _library: Dict[str, List] = {
        "relu": [
            # ReLU: max(0, x)
            # Assumes input x is in S0
            [100, 1, 0],      # LOAD R1, S0 (input x)
            [101, 2, 0.0],    # LOAD_CONST R2, 0.0
            [300, 1, 2, 3],   # MAX R3, R1, R2 (max(x, 0))
            [453, 3]          # RETURN R3
            # Opcode assumptions: 100=LOAD, 101=LOAD_CONST, 300=MAX, 453=RETURN
        ],
        "sigmoid": [
            # Sigmoid: 1 / (1 + exp(-x))
            # Assumes input x is in S0
            [100, 1, 0],      # LOAD R1, S0 (x)
            [11, 1, 2],        # NEGATE R2, R1 (-x) - Assuming opcode 3 for NEGATE
            [8, 2, 3],       # EXP R3, R2 (exp(-x)) - Assuming opcode 12 for EXP
            [101, 4, 1.0],    # LOAD_CONST R4, 1.0
            [1, 4, 3, 5],     # ADD R5, R4, R3 (1 + exp(-x)) - Assuming opcode 1 for ADD
            [4, 4, 5, 6],     # DIVIDE R6, R4, R5 (1 / (1 + exp(-x))) - Assuming opcode 5 for DIVIDE
            [453, 6]          # RETURN R6
        ],
        "identity": [
            # Identity: f(x) = x
            [100, 1, 0],      # LOAD R1, S0
            [453, 1]          # RETURN R1
        ]
        # TODO: Add other standard activations like Tanh, LeakyReLU, etc.
    }

    @staticmethod
    def get_function(name: str) -> Optional[List]:
        """
        Retrieve the genetic code for a standard activation function by name.

        Args:
            name: The name of the activation function (e.g., "relu", "sigmoid").

        Returns:
            The genetic code as a list of lists, or None if not found.
        """
        return ActivationLibrary._library.get(name)

    @staticmethod
    def list_functions() -> List[str]:
        """Return a list of available standard activation function names."""
        return list(ActivationLibrary._library.keys())
