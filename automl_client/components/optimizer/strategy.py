"""Strategy for evolving and evaluating Optimizer components."""

import torch
import numpy as np
import logging
from typing import List, Dict, Any, Optional
import traceback
import json

from ..base import AIComponentStrategy
# Use absolute imports
from automl_client.genetic.interpreter import GeneticInterpreter
from automl_client.genetic.function_converter import FunctionConverter
from automl_client.evaluation_strategy import log_genetic_code
# from automl_client.genetic.genetic_library import GeneticLibrary # Removed
from .library import OptimizerLibrary # Use component-specific library

logger = logging.getLogger(__name__)

class OptimizerStrategy(AIComponentStrategy):
    """Handles the evolution and evaluation of optimizer components."""

    def __init__(self, config: Dict = None):
        """Initialize OptimizerStrategy."""
        super().__init__(config)
        # Specific config for optimizer strategy
        self.optimizer_config = {
            'evaluation_epochs': self.config.get('evaluation_epochs', 3),
            'batch_size': self.config.get('batch_size', 32),
            'learning_rate': self.config.get('learning_rate', 0.01), # Default LR for evaluation
            'debug_level': self.config.get('debug_level', 0),
            # Add any other optimizer-specific configs here
        }
        logger.info(f"OptimizerStrategy initialized with config: {self.optimizer_config}")

    def create_initial_solution(self) -> List:
        """
        Create an initial genetic code solution for an optimizer.
        Defaults to basic SGD.
        Genetic representation needs to define how parameters are updated.
        Example: Parameter Update = Parameter - LearningRate * Gradient
        """
        logger.info("Creating initial solution for OptimizerStrategy (defaulting to SGD).")
        try:
            # Use OptimizerLibrary to get the standard SGD genetic code
            sgd_code = OptimizerLibrary.get_function("sgd")
            if not sgd_code:
                logger.error("Could not retrieve standard SGD code from OptimizerLibrary! Falling back.")
                # Fallback basic structure if library fails unexpectedly
                sgd_code = [
                    [100, 1, 0], [100, 2, 1], [100, 3, 3], [4, 4, 3, 2], [2, 5, 1, 4], [453, 5] # Simplified SGD
                ]
            return sgd_code
        except Exception as e:
            logger.exception(f"Error creating initial optimizer solution: {e}")
            # Provide a minimal valid code on error (e.g., return parameter unchanged)
            return [[100, 1, 0], [453, 1]] # Identity update

    def evaluate(self, genetic_code: list, dataset: dict) -> float:
        """
        Evaluate genetic code with specified optimizer.
        
        Args:
            genetic_code: Genetic code to evaluate
            dataset: Dictionary with trainX, trainY, valX, valY tensors
            
        Returns:
            Validation accuracy as fitness score
        """
        self.evaluation_count += 1
        
        # Log at regular intervals or in debug mode
        should_log = (self.evaluation_count % self.config['log_interval'] == 0)
        debug_level = self.config['debug_level']
        
        if should_log:
            logger.debug(f"Starting evaluation #{self.evaluation_count} with {self.config['optimizer']}")
        
        # Log genetic code structure
        if debug_level >= 1:
            logger.debug(f"Evaluation #{self.evaluation_count}: Analyzing genetic code structure")
            log_genetic_code(genetic_code)
        
        try:
            logger.debug(f"Evaluation #{self.evaluation_count}: Decoding genetic code to loss function")
            # Create interpreter config with computation budget settings
            interpreter_config = {
                'max_computation_factor': 10,  # Default: 10x program length
                'min_computation_budget': 10  # Default: minimum 1000 instructions
            }
            loss_fn = GeneticInterpreter.decode_to_function(genetic_code, interpreter_config)
            
            # Create model using factory if provided, otherwise use default
            model_factory = self.config.get('model_factory')
            if model_factory:
                model = model_factory()
            else:
                # Default model for MNIST
                model = torch.nn.Sequential(
                    torch.nn.Linear(28*28, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, 10),
                    torch.nn.Softmax(dim=1)
                )
            
            # Log model info
            if debug_level >= 2:
                logger.debug(f"Model architecture: {model}")
                for name, param in model.named_parameters():
                    logger.debug(f"Parameter {name}: shape={param.shape}, requires_grad={param.requires_grad}")
            
            # Get optimizer
            optimizer_class = getattr(torch.optim, self.config['optimizer'])
            optimizer = optimizer_class(model.parameters(), lr=self.config['learning_rate'])
            
            # Log dataset info
            if debug_level >= 1:
                for key, tensor in dataset.items():
                    log_tensor_info(key, tensor)
            
            # Early stopping configuration
            early_stopping_patience = self.config.get('early_stopping_patience', 3)  # Default: 3 epochs
            min_improvement = self.config.get('min_improvement', 0.001)  # Minimum improvement threshold
            best_loss = float('inf')
            patience_counter = 0
            
            # Training
            for epoch in range(self.config['evaluation_rounds']):
                model.train()
                epoch_losses = []
                
                for i in range(0, len(dataset['trainX']), self.config['batch_size']):
                    batch_X = dataset['trainX'][i:i+self.config['batch_size']]
                    batch_Y = dataset['trainY'][i:i+self.config['batch_size']]
                    
                    # Log batch info in verbose mode
                    if debug_level >= 2 and i == 0:  # Only log first batch
                        log_tensor_info(f"epoch{epoch}_batch{i}_X", batch_X)
                        log_tensor_info(f"epoch{epoch}_batch{i}_Y", batch_Y)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    
                    # Log outputs in verbose mode
                    if debug_level >= 2 and i == 0:
                        log_tensor_info(f"epoch{epoch}_batch{i}_outputs", outputs)
                    
                    try:
                        # Log pre-loss function state
                        if debug_level >= 2 and i == 0:
                            logger.debug(f"Epoch {epoch}, Batch {i}: Calling loss function")
                            logger.debug(f"batch_Y shape: {batch_Y.shape}, requires_grad: {batch_Y.requires_grad}")
                            logger.debug(f"outputs shape: {outputs.shape}, requires_grad: {outputs.requires_grad}")
                        
                        # Execute loss function
                        loss = loss_fn(batch_Y, outputs)
                        
                        # Log loss info
                        if debug_level >= 2 and i == 0:
                            logger.debug(f"Loss value: {loss.item():.6f}, shape: {loss.shape}, "
                                        f"requires_grad: {loss.requires_grad}, "
                                        f"grad_fn: {loss.grad_fn}")
                        
                        # Backward pass
                        try:
                            loss.backward()
                            
                            # Log gradient info in verbose mode
                            if debug_level >= 2 and i == 0:
                                for name, param in model.named_parameters():
                                    if param.grad is not None:
                                        logger.debug(f"Gradient for {name}: "
                                                    f"min={param.grad.min().item():.6f}, "
                                                    f"max={param.grad.max().item():.6f}, "
                                                    f"mean={param.grad.mean().item():.6f}")
                                    else:
                                        logger.debug(f"No gradient for {name}")
                        except Exception as e:
                            error_details = traceback.format_exc()
                            logger.debug(f"Error during backward pass: {e}")
                            logger.debug(f"Backward error details: {error_details}")
                            return 0.0
                        
                        optimizer.step()
                        epoch_losses.append(loss.item())
                        
                    except Exception as e:
                        error_details = traceback.format_exc()
                        logger.debug(f"Error during training: {e}")
                        logger.debug(f"Training error details: {error_details}")
                        
                        # Log more details about the specific error
                        if "Unsupported reduction operation" in str(e):
                            logger.debug(f"Detected unsupported reduction operation. "
                                        f"This may be due to an invalid operation code in the genetic code.")
                            
                        elif "element 0 of tensors does not require grad" in str(e):
                            logger.debug(f"Detected gradient computation error. "
                                        f"This may be due to tensors not requiring gradients.")
                            
                            # Log tensor gradient status
                            logger.debug(f"batch_Y requires_grad: {batch_Y.requires_grad}")
                            logger.debug(f"outputs requires_grad: {outputs.requires_grad}")
                            if hasattr(loss, 'requires_grad'):
                                logger.debug(f"loss requires_grad: {loss.requires_grad}")
                        
                        return 0.0
                
                # Calculate average epoch loss
                avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
                
                if should_log or debug_level >= 1:
                    logger.debug(f"Epoch {epoch+1}/{self.config['evaluation_rounds']}, Loss: {avg_epoch_loss:.6f}")
                
                # Early stopping check
                if avg_epoch_loss < best_loss - min_improvement:
                    # We've improved significantly
                    best_loss = avg_epoch_loss
                    patience_counter = 0
                    logger.debug(f"New best loss: {best_loss:.6f}")
                else:
                    # No significant improvement
                    patience_counter += 1
                    logger.debug(f"No improvement for {patience_counter} epochs. Best: {best_loss:.6f}, Current: {avg_epoch_loss:.6f}")
                    
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping triggered after {epoch+1} epochs. No improvement for {early_stopping_patience} epochs.")
                        return 0.0  # Early stopping, return poor fitness
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                val_outputs = model(dataset['valX'])
                
                if debug_level >= 1:
                    log_tensor_info("val_outputs", val_outputs)
                
                _, predicted = torch.max(val_outputs, 1)
                _, labels = torch.max(dataset['valY'], 1)
                correct = (predicted == labels).sum().item()
                accuracy = correct / len(dataset['valY'])
                
                if should_log:
                    logger.debug(f"Evaluation complete, Accuracy: {accuracy:.6f}")
                
                return accuracy
                
        except Exception as e:
            error_details = traceback.format_exc()
            logger.debug(f"Error in evaluation: {e}")
            logger.debug(f"Evaluation error details: {error_details}")
            return 0.0

    def interpret_result(self, result: Dict) -> Dict:
        """
        Interpret the final evolved optimizer.

        Args:
            result: Dictionary containing evolution results (fitness, genetic_code).

        Returns:
            Dictionary with interpretation, potentially including a Python function string
            or a description of the optimizer logic.
        """
        logger.info(f"Interpreting result for OptimizerStrategy. Best fitness: {result.get('fitness', -np.inf):.6f}")
        genetic_code = result.get('genetic_code')
        interpretation = {
            'component_type': 'optimizer',
            'best_fitness_raw': result.get('fitness', -np.inf),
            'best_val_loss': -result.get('fitness', np.inf) if result.get('fitness') is not None else 'N/A', # Convert back
            'genetic_code_length': len(genetic_code) if genetic_code else 0,
            'evolved_representation': None, # Placeholder for python code or description
            'notes': []
        }

        if not genetic_code:
            interpretation['notes'].append("No genetic code found in result.")
            return interpretation

        try:
            # TODO: Implement a more sophisticated conversion/description
            # For now, just include the genetic code as a string representation
            interpretation['evolved_representation'] = f"Genetic Code: {json.dumps(genetic_code)}"
            interpretation['notes'].append("Basic interpretation: Genetic code provided.")

            # Potential future work: Convert to a PyTorch Optimizer class or descriptive text
            # python_code = FunctionConverter.genetic_code_to_optimizer_logic(genetic_code) # Hypothetical
            # interpretation['evolved_representation'] = python_code

        except Exception as e:
            logger.error(f"Failed to interpret genetic code for optimizer: {e}")
            interpretation['notes'].append(f"Failed to interpret genetic code: {e}")
            interpretation['evolved_representation'] = "# Interpretation Error"

        return interpretation
