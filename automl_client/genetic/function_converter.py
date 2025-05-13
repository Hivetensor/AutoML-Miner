"""Function Converter for Genetic Language in PyTorch."""

import torch
from typing import List
# from .genetic_library import GeneticLibrary # Removed - Obsolete

def log_tensor_state(tensor: torch.Tensor, op_name: str) -> None:
    """Log tensor state for debugging."""
    print(f"[TensorDebug] Operation: {op_name}", {
        'isDefined': tensor is not None,
        'isTensor': isinstance(tensor, torch.Tensor),
        'shape': tensor.shape if isinstance(tensor, torch.Tensor) else None,
        'dtype': tensor.dtype if isinstance(tensor, torch.Tensor) else None,
        'memory': tensor.detach().cpu().numpy().flatten()[:5] 
                 if isinstance(tensor, torch.Tensor) else None
    })

class FunctionConverter:
    """Converts between genetic code and Python functions."""
    
    @staticmethod
    def genetic_code_to_python(genetic_code: List[List]) -> str:
        """
        Convert genetic code to a Python function string.
        
        Args:
            genetic_code: Genetic code to convert
            
        Returns:
            Python function as string
        """
        if not isinstance(genetic_code, list) or any(not isinstance(instr, list) for instr in genetic_code):
            raise ValueError("Invalid genetic_code: must be list of lists")
            
        # Check for known patterns
        if FunctionConverter._is_genetic_mse(genetic_code):
            return """def loss_fn(y_true, y_pred):
    with torch.no_grad():
        diff = y_true - y_pred
        squared_diff = diff ** 2
        return torch.mean(squared_diff)
"""
        elif FunctionConverter._is_genetic_mae(genetic_code):
            return """def loss_fn(y_true, y_pred):
    with torch.no_grad():
        diff = y_true - y_pred
        abs_diff = torch.abs(diff)
        return torch.mean(abs_diff)
"""
        elif FunctionConverter._is_genetic_bce(genetic_code):
            return """def loss_fn(y_true, y_pred):
    with torch.no_grad():
        epsilon = torch.tensor(1e-7)
        clipped = torch.clamp(y_pred, epsilon, 1.0 - epsilon)
        log_pred = torch.log(clipped)
        log_one_minus = torch.log(1.0 - clipped)
        first_term = y_true * log_pred
        second_term = (1 - y_true) * log_one_minus
        loss = -(first_term + second_term)
        return torch.mean(loss)
"""
        # Fallback to interpreter-based function
        return """def loss_fn(y_true, y_pred):
    from .interpreter import GeneticInterpreter
    interpreter = GeneticInterpreter()
    interpreter.initialize({{'y_true': y_true, 'y_pred': y_pred}})
    interpreter.load_program({})
    return interpreter.execute()
""".format(repr(genetic_code))

    
    @staticmethod
    def _is_genetic_mse(genetic_code: List[List]) -> bool:
        """Check if genetic code is MSE."""
        return len(genetic_code) == 6 and any(instr[0] == 10 for instr in genetic_code)  # SQUARE op
    
    @staticmethod
    def _is_genetic_mae(genetic_code: List[List]) -> bool:
        """Check if genetic code is MAE."""
        return len(genetic_code) >= 5 and any(instr[0] == 9 for instr in genetic_code)  # ABS op
    
    @staticmethod
    def _is_genetic_bce(genetic_code: List[List]) -> bool:
        """Check if genetic code is BCE."""
        return len(genetic_code) >= 10 and sum(instr[0] == 7 for instr in genetic_code) >= 2  # LOG ops

    # Removed get_standard_function and list_standard_functions as GeneticLibrary is obsolete
