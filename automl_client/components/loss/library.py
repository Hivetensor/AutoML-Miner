"""Genetic Library for standard Loss Function components."""

from typing import List, Dict, Optional

class LossLibrary:
    """Provides genetic code representations for common loss functions."""

    _library: Dict[str, List] = {
        "mse_loss": [
            [100, 1, 0],      # LOAD R1, S0 (y_true)
            [100, 2, 1],      # LOAD R2, S1 (y_pred)
            [2, 1, 2, 3],     # SUB R3, R1, R2 (diff = y_true - y_pred)
            [10, 3, 4],       # SQUARE R4, R3 (diff_sq = diff^2)
            [201, 4, -1, 9],  # REDUCE_MEAN R9, axis=-1, R4 (mean_diff_sq)
            [453, 9]          # RETURN R9
        ],
        "mae_loss": [
            [100, 1, 0],      # LOAD R1, S0 (y_true)
            [100, 2, 1],      # LOAD R2, S1 (y_pred)
            [2, 1, 2, 3],     # SUB R3, R1, R2 (diff = y_true - y_pred)
            [11, 3, 4],       # ABS R4, R3 (abs_diff = abs(diff)) - Assuming opcode 11 for ABS
            [201, 4, -1, 9],  # REDUCE_MEAN R9, axis=-1, R4 (mean_abs_diff)
            [453, 9]          # RETURN R9
        ],
        "bce_loss": [
            [100, 1, 0],       # LOAD R1, S0 (y_true)
            [100, 2, 1],       # LOAD R2, S1 (y_pred)
            # Note: Original code included epsilon clipping - omitting for standard BCE
            # [100, 3, 4],       # LOAD R3, S4 (epsilon)
            # [3, 2, 3, 4],      # MUL R2, R3, R4 -> Max?
            # [1, 4, 2, 2],      # ADD R4, R2, R2 -> Clip?
            [100, 4, 5],       # LOAD R4, S5 (ones) - Assuming ones tensor is provided at S5
            [2, 2, 5, 4],      # SUB R4, R2, R5 (1 - y_pred) -> R5 = 1 - R2
            [2, 6, 4, 1],      # SUB R4, R1, R6 (1 - y_true) -> R6 = 1 - R1
            [7, 2, 7],         # LOG R2, R7 (log(y_pred)) -> R7 = log(R2)
            [7, 5, 8],         # LOG R5, R8 (log(1-y_pred)) -> R8 = log(R5)
            [3, 1, 7, 7],      # MUL R1, R7, R7 (y_true * log(y_pred)) -> R7 = R1 * R7
            [3, 6, 8, 8],      # MUL R6, R8, R8 ((1-y_true) * log(1-y_pred)) -> R8 = R6 * R8
            [1, 7, 8, 7],      # ADD R7, R8, R7 -> R7 = R7 + R8
            [11, 7, 7],        # NEGATE R7, R7 (-result) -> R7 = -R7 (Assuming opcode 11 is NEGATE)
            [201, 7, -1, 5],   # REDUCE_MEAN R7, axis=-1, R5 -> R5 = mean(R7)
            [201, 5, 0, 9],    # REDUCE_MEAN R5, axis=0, R9 (reduce batch dimension) -> R9 = mean(R5)
            [453, 9]           # RETURN R9
        ],
    }

    @staticmethod
    def get_function(name: str) -> Optional[List]:
        """
        Retrieve the genetic code for a standard loss function by name.

        Args:
            name: The name of the loss function (e.g., "mse_loss").

        Returns:
            The genetic code as a list of lists, or None if not found.
        """
        return LossLibrary._library.get(name)

    @staticmethod
    def list_functions() -> List[str]:
        """Return a list of available standard loss function names."""
        return list(LossLibrary._library.keys())
