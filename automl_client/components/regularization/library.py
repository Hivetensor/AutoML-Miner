"""Genetic Library for standard Regularization components."""

from typing import List, Dict, Optional
import json

class RegularizationLibrary:
    """Provides genetic code representations for common regularization techniques."""

    _library: Dict[str, List] = {
        "no_regularization": [
            [101, 9, 0.0],  # LOAD_CONST R9, 0.0 (Return zero penalty)
            [453, 9]        # RETURN R9
        ],
        "l2_regularization": [
            # Calculates strength * sum(weights^2)
            # Assumes S1 = flattened weights, S2 = strength factor
            [100, 1, 1],      # LOAD R1, S1 (Load weights)
            [10, 2, 1],       # SQUARE R2, R1 (Square weights)
            [200, 3, 2],      # REDUCE_SUM R3, R2 (Sum squared weights)
            [100, 4, 2],      # LOAD R4, S2 (Load strength factor)
            [3, 9, 3, 4],     # MUL R9, R3, R4 (penalty = sum * strength)
            [453, 9]          # RETURN R9 (Return penalty)
        ],
        "l1_regularization": [
            # Calculates strength * sum(abs(weights))
            # Assumes S1 = flattened weights, S2 = strength factor
            [100, 1, 1],      # LOAD R1, S1 (Load weights)
            [9, 2, 1],        # ABS R2, R1 (Absolute value of weights)
            [200, 3, 2],      # REDUCE_SUM R3, R2 (Sum absolute weights)
            [100, 4, 2],      # LOAD R4, S2 (Load strength factor)
            [3, 9, 3, 4],     # MUL R9, R3, R4 (penalty = sum * strength)
            [453, 9]          # RETURN R9 (Return penalty)
        ],
        # TODO: Add other regularization types like Elastic Net?
    }

    @staticmethod
    def get_function(name: str) -> Optional[List]:
        """
        Retrieve the genetic code for a standard regularization function by name.

        Args:
            name: The name of the function (e.g., "l2_regularization").

        Returns:
            The genetic code as a list of lists, or None if not found.
        """
        return RegularizationLibrary._library.get(name)

    @staticmethod
    def list_functions() -> List[str]:
        """Return a list of available standard regularization function names."""
        return list(RegularizationLibrary._library.keys())

    @staticmethod
    def _is_like(genetic_code: List, standard_name: str) -> bool:
        """
        Basic check if genetic code matches a standard function.
        NOTE: This is a placeholder and likely needs a more robust comparison.
        """
        standard_code = RegularizationLibrary.get_function(standard_name)
        if not standard_code or not genetic_code:
            return False
        # Simple comparison based on JSON representation (can be brittle)
        try:
            return json.dumps(genetic_code) == json.dumps(standard_code)
        except Exception:
            return False
