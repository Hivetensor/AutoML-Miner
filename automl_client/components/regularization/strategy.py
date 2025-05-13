"""Strategy for evolving and evaluating Regularization components."""

import torch
import numpy as np
import logging
import traceback
import json
from typing import List, Dict, Any, Optional

from ..base import AIComponentStrategy
# Use absolute import from project root
from automl_client.genetic.interpreter import GeneticInterpreter
from automl_client.genetic.function_converter import FunctionConverter
from .library import RegularizationLibrary # Use component-specific library

logger = logging.getLogger(__name__)

class RegularizationStrategy(AIComponentStrategy):
    """Handles the evolution and evaluation of regularization components."""

    def __init__(self, config: Dict = None):
        """Initialize RegularizationStrategy."""
        super().__init__(config)
        # Specific config for regularization strategy
        self.reg_config = {
            'debug_level': self.config.get('debug_level', 0),
            # Add any other regularization-specific configs here
        }
        logger.info(f"RegularizationStrategy initialized with config: {self.reg_config}")

    def create_initial_solution(self) -> List:
        """
        Create an initial genetic code solution for regularization.
        Defaults to 'no regularization'.
        """
        logger.info("Creating initial solution for RegularizationStrategy (defaulting to 'no regularization').")
        try:
            # Use RegularizationLibrary to get the default code
            no_reg_code = RegularizationLibrary.get_function("no_regularization")
            if not no_reg_code:
                logger.error("Could not retrieve 'no_regularization' code from RegularizationLibrary! Falling back.")
                # Fallback: code that returns 0
                no_reg_code = [
                    [101, 9, 0.0], # LOAD_CONST R9, 0.0
                    [453, 9]       # RETURN R9
                ]
            return no_reg_code
        except Exception as e:
            logger.exception(f"Error creating initial regularization solution: {e}")
            # Provide a minimal valid code on error (returns 0)
            return [[101, 9, 0.0], [453, 9]]

    def evaluate(self, genetic_code: List, dataset: Dict, model: Optional[torch.nn.Module] = None) -> float:
        """
        Evaluate a regularization term represented by genetic code.

        Args:
            genetic_code: The genetic code for the regularization term calculation.
            dataset: Dictionary potentially containing:
                     - 'base_loss': The loss calculated by the primary loss function.
                     - 'model_weights': A list or tensor of model parameters.
                     - Other data if needed by specific regularization types.
            model: The PyTorch model (might be used if weights aren't directly in dataset).

        Returns:
            Fitness score. For regularization, this is tricky.
            Option 1: Return -penalty (lower penalty is better).
            Option 2: Return -(base_loss + penalty). Requires base_loss input.
            Let's assume Option 2 for now, returning negative combined loss.
            Returns -infinity if evaluation fails.
        """
        if not genetic_code:
            logger.warning("Evaluate called with empty genetic code for RegularizationStrategy.")
            return -np.inf

        interpreter = None
        try:
            base_loss_tensor = dataset.get('base_loss')
            model_weights = dataset.get('model_weights') # How are weights provided? List? Tensor?

            if base_loss_tensor is None:
                 logger.error("Regularization evaluation requires 'base_loss' in dataset.")
                 return -np.inf
            if not isinstance(base_loss_tensor, torch.Tensor):
                 base_loss_tensor = torch.tensor(float(base_loss_tensor)) # Ensure it's a tensor

            # --- How to get weights? ---
            # Option A: Directly from dataset['model_weights']
            if model_weights is None and model:
                 # Option B: Extract from provided model
                 logger.debug("Extracting weights from provided model for regularization.")
                 # This needs careful implementation: flatten, handle different layers, etc.
                 # Placeholder: just get parameters as a list of tensors
                 model_weights = [p.data.detach().clone() for p in model.parameters() if p.requires_grad]
                 if not model_weights:
                      logger.error("Could not extract weights from the provided model.")
                      return -np.inf
            elif model_weights is None:
                 logger.error("Regularization evaluation requires 'model_weights' in dataset or a 'model'.")
                 return -np.inf
            # --- End Weight Handling ---

            # --- Execute Genetic Code for Penalty ---
            interpreter = GeneticInterpreter(config={'debug': self.reg_config['debug_level'] > 1})
            interpreter.load_program(genetic_code)

            # Prepare inputs for the interpreter
            # S0: Base Loss
            # S1: Model Weights (How to represent? Flattened tensor? List of tensors?)
            # Placeholder: Assume S1 expects a flattened tensor of all weights
            # This needs to align with the genetic code in the library!
            try:
                 # Flatten weights into a single tensor if they are a list
                 if isinstance(model_weights, list):
                      flattened_weights = torch.cat([w.flatten() for w in model_weights])
                 elif isinstance(model_weights, torch.Tensor):
                      flattened_weights = model_weights.flatten()
                 else:
                      raise TypeError("Unsupported type for model_weights")

                 interpreter_inputs = {
                     0: base_loss_tensor.detach().clone(), # S0: Base Loss
                     1: flattened_weights                 # S1: Flattened Weights
                     # Add other inputs if needed (e.g., regularization strength factor at S2?)
                 }
            except Exception as input_prep_err:
                 logger.error(f"Failed to prepare inputs for regularization interpreter: {input_prep_err}")
                 return -np.inf

            interpreter.initialize(inputs=interpreter_inputs)
            penalty_tensor = interpreter.execute() # Genetic code calculates the penalty term

            if penalty_tensor is None or not isinstance(penalty_tensor, torch.Tensor):
                logger.warning("Regularization interpreter did not return a valid penalty tensor.")
                return -np.inf

            # Ensure penalty is a scalar
            if penalty_tensor.numel() != 1:
                logger.warning(f"Regularization penalty is non-scalar (shape: {penalty_tensor.shape}). Taking mean.")
                penalty_value = penalty_tensor.mean().item()
            else:
                penalty_value = penalty_tensor.item()

            # Handle NaN/Infinity for penalty
            if np.isnan(penalty_value) or np.isinf(penalty_value):
                logger.warning(f"Regularization penalty is NaN or Infinity. Treating as zero penalty.")
                penalty_value = 0.0

            # Combine base loss and penalty
            final_loss = base_loss_tensor.item() + penalty_value

            # Fitness: Lower combined loss is better
            fitness = -final_loss
            return fitness

        except Exception as e:
            logger.error(f"Error during regularization evaluation: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            # logger.debug(f"Problematic genetic code: {json.dumps(genetic_code)}") # Careful with large codes
            return -np.inf # Return worst fitness on error
        finally:
            if interpreter:
                try:
                    interpreter.dispose()
                except Exception as dispose_err:
                    logger.warning(f"Error disposing interpreter resources: {dispose_err}")

    def interpret_result(self, result: Dict) -> Dict:
        """
        Interpret the final evolved regularization component.

        Args:
            result: Dictionary containing evolution results (fitness, genetic_code).

        Returns:
            Dictionary with interpretation (e.g., type, strength).
        """
        logger.info(f"Interpreting result for RegularizationStrategy. Best fitness: {result.get('fitness', -np.inf):.6f}")
        genetic_code = result.get('genetic_code')
        interpretation = {
            'component_type': 'regularization',
            'best_fitness_raw': result.get('fitness', -np.inf),
            # 'best_penalty_value': ? # How to get this? Re-run interpreter?
            'genetic_code_length': len(genetic_code) if genetic_code else 0,
            'evolved_representation': "Unknown", # Placeholder
            'notes': []
        }

        if not genetic_code:
            interpretation['notes'].append("No genetic code found in result.")
            return interpretation

        try:
            # Try to identify known patterns (this is simplified)
            # A more robust approach might involve analyzing opcodes or using FunctionConverter
            # Placeholder logic:
            if RegularizationLibrary._is_like(genetic_code, "l2_regularization"):
                 interpretation['evolved_representation'] = "{'type': 'L2', 'strength': 'evolved'}" # Need to parse strength
                 interpretation['notes'].append("Evolved function resembles L2 regularization.")
            elif RegularizationLibrary._is_like(genetic_code, "l1_regularization"):
                 interpretation['evolved_representation'] = "{'type': 'L1', 'strength': 'evolved'}" # Need to parse strength
                 interpretation['notes'].append("Evolved function resembles L1 regularization.")
            elif RegularizationLibrary._is_like(genetic_code, "no_regularization"):
                 interpretation['evolved_representation'] = "{'type': 'None'}"
                 interpretation['notes'].append("Evolved function resembles no regularization.")
            else:
                 interpretation['notes'].append("Could not identify standard regularization pattern.")
                 # Optionally convert to Python code if feasible
                 # python_code = FunctionConverter.genetic_code_to_python(genetic_code)
                 # interpretation['evolved_representation'] = python_code

        except Exception as e:
            logger.error(f"Failed to interpret regularization genetic code: {e}")
            interpretation['notes'].append(f"Failed interpretation: {e}")
            interpretation['evolved_representation'] = "# Interpretation Error"

        return interpretation

    def get_component_config(self) -> Dict:
        """Get configuration specific to this component."""
        # Return any default configs needed for regularization tasks
        return {
            "requires_model_weights": True, # Indicate that evaluation needs weights
            "modifies_loss": True          # Indicate it adjusts the base loss
        }
