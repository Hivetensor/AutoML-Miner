"""Strategy for evolving and evaluating Loss Function components."""

import torch
import numpy as np
import logging
import traceback
import json
from typing import List, Dict, Any, Optional

from ..base import AIComponentStrategy
# Use absolute imports
from automl_client.genetic.interpreter import GeneticInterpreter
from automl_client.genetic.function_converter import FunctionConverter
from automl_client.models.model_factory import ModelFactory
from automl_client.data.dataset_factory import DatasetFactory
from .library import LossLibrary # Component-specific library
from automl_client.utils.fec import get_fixed_tensor, fec_lookup_or_evaluate

logger = logging.getLogger(__name__)

class LossStrategy(AIComponentStrategy):
    """Handles the evolution and evaluation of loss function components."""

    def __init__(self, config: Dict = None):
        """Initialize LossStrategy."""
        super().__init__(config)
        # Specific config for loss strategy, potentially overriding engine defaults
        self.loss_config = {
            'evaluation_rounds': self.config.get('evaluation_rounds', 1),
            'batch_size': self.config.get('batch_size', 32),
            'debug_level': self.config.get('debug_level', 0),
            'model_seed': self.config.get('model_seed', 42),  # Add seed for deterministic initialization
            'dataset_name': self.config.get('dataset_name', 'mnist'),  # Default dataset
            'dataset_config': self.config.get('dataset_config', {
                'flatten': True,  # Ensure flattened input for MLP models
                'one_hot': True,  # One-hot encoding for classification
                'normalize': True,  # Normalize inputs for stable training
            }),
            'model_config': self.config.get('model_config', {
                'architecture': 'simple_mlp',  # Default architecture
                'hidden_dim': 64,  # Default hidden dimension
            }),
        }
        logger.info(f"LossStrategy initialized with config: {self.loss_config}")
        # Cache for reference model to avoid repeated initialization
        self._reference_model = None
        self._reference_state_dict = None

    def create_initial_solution(self) -> List:
        """
        Create an initial genetic code solution for a loss function.
        Defaults to Mean Squared Error (MSE).
        """
        logger.info("Creating initial solution for LossStrategy (defaulting to MSE).")
        try:
            # Use LossLibrary to get the standard MSE genetic code
            mse_code = LossLibrary.get_function("mse_loss")
            if not mse_code:
                logger.error("Could not retrieve standard MSE code from LossLibrary! Falling back.")
                # Fallback basic structure if library fails unexpectedly
                mse_code = [
                    [100, 1, 0],  # LOAD R1, S0 (y_true)
                    [100, 2, 1],  # LOAD R2, S1 (y_pred)
                    [2, 3, 1, 2],   # SUB R3, R1, R2 (diff = y_true - y_pred)
                    [10, 4, 3],   # SQUARE R4, R3 (diff_sq = diff^2)
                    [201, 9, -1, 4], # REDUCE_MEAN R9, axis=-1, R4 (mean_diff_sq)
                    [453, 9]      # RETURN R9
                ]
            return mse_code
        except Exception as e:
            logger.exception(f"Error creating initial loss solution: {e}")
            # Provide a minimal valid code on error
            return [[100, 1, 0], [100, 2, 1], [350, 9, 1, 2], [453, 9]] # Minimal MSE op

    def create_dataset(self) -> Dict[str, torch.Tensor]:
        """
        Create an appropriate dataset for loss function evaluation.
        Called when no dataset is provided to evaluate().
        """
        dataset_name = self.loss_config.get('dataset_name', 'mnist')
        dataset_config = self.loss_config.get('dataset_config', {})
        
        logger.info(f"Creating {dataset_name} dataset for loss function evaluation")
        EVAL_SEED = 42
        dataset = DatasetFactory.create_dataset(dataset_name, dataset_config, eval_seed=EVAL_SEED)
        logger.info(f"Created dataset: trainX shape={dataset['trainX'].shape}, " 
                       f"trainY shape={dataset['trainY'].shape}")
        return dataset

    def create_model(self, dataset: Dict[str, torch.Tensor]) -> torch.nn.Module:
        """
        Create an appropriate model for loss function evaluation.
        Uses dataset dimensions to ensure compatible input/output sizes.
        """
        # Extract dimensions from dataset
        input_dim = dataset['trainX'].shape[1]  # Feature dimension
        
        # Determine output dimension from trainY
        if len(dataset['trainY'].shape) > 1:
            output_dim = dataset['trainY'].shape[1]  # Multi-class or one-hot encoded
        else:
            # For single-class labels, count unique classes
            output_dim = len(torch.unique(dataset['trainY']))
            
        logger.info(f"Creating model with input_dim={input_dim}, output_dim={output_dim}")
        
        # Get architecture type and config from loss_config
        architecture = self.loss_config.get('model_config', {}).get('architecture', 'simple_mlp')
        
        # Prepare model config with correct dimensions
        model_config = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'hidden_dim': self.loss_config.get('model_config', {}).get('hidden_dim', 64),
            'model_seed': self.loss_config.get('model_seed', 42)
        }
        
        # Set seed for deterministic initialization
        ModelFactory.set_seed(model_config['model_seed'])
        
        try:
            # Use ModelFactory to create the model with correct dimensions
            model = ModelFactory.create_model(
                component_type="loss",
                config=model_config
            )
            logger.info(f"Created model using ModelFactory: {type(model)}")
            return model
        except Exception as e:
            logger.error(f"Error creating model with ModelFactory: {e}")
            
            # Fallback to direct model creation if ModelFactory fails
            logger.info("Using fallback model creation")
            model = torch.nn.Sequential(
                torch.nn.Linear(input_dim, model_config['hidden_dim']),
                torch.nn.ReLU(),
                torch.nn.Linear(model_config['hidden_dim'], output_dim),
                torch.nn.Softmax(dim=1) if output_dim > 1 else torch.nn.Identity()
            )
            logger.info(f"Created fallback model: {model}")
            return model

    def evaluate(self, genetic_code: List, dataset=None, model=None, stop_flag=None) -> float:
        # —— FAST EQUIVALENCE CHECK ——
        # 1) Ensure dataset for shapes
        if dataset is None:
            dataset = self.create_dataset()
        batch_size = self.loss_config['batch_size']
        # If trainY is one‑hot, shape is (batch_size, num_classes)
        y_shape = dataset['trainY'].shape[1] if dataset['trainY'].dim() > 1 else 1

        # 2) Two fixed tensors: y_true and y_pred
        test_y = get_fixed_tensor('loss_test_y', (batch_size, y_shape))
        test_p = get_fixed_tensor('loss_test_p', (batch_size, y_shape))

        # 3) Single execute on the genetic loss
        from automl_client.genetic.interpreter import GeneticInterpreter
        interp = GeneticInterpreter()
        interp.initialize({'y_true': test_y, 'y_pred': test_p})
        interp.load_program(genetic_code)
        out = interp.execute()

        # 4) Quantize + hash + cache
        arr = torch.round(out.unsqueeze(0) * 1e4).to(torch.float16).cpu().numpy()
        return fec_lookup_or_evaluate(arr.tobytes(),
            lambda: self._full_evaluate(genetic_code, dataset, model, stop_flag)
        )

    def _full_evaluate(self, genetic_code: List, dataset: Optional[Dict] = None, model: Optional[torch.nn.Module] = None, stop_flag=None) -> float:
        """
        Evaluate a loss function represented by genetic code by training a model with it
        and measuring validation performance.
        
        Args:
            genetic_code: The genetic code representing a loss function
            dataset: Optional dictionary with trainX, trainY, valX, valY tensors
            model: Optional pre-initialized model
            stop_flag: Optional StopFlag object to check for stopping
            
        Returns:
            Fitness score (higher is better) based on validation metrics after training
            or -np.inf if evaluation is stopped
        """
        # Check for stop signal before starting
        if stop_flag and stop_flag.is_stopped():
            logger.info("Loss evaluation stopped before starting")
            return -np.inf
            
        if not genetic_code:
            logger.warning("Evaluate called with empty genetic code.")
            return -np.inf
            
        # Create dataset if not provided
        if dataset is None:
            logger.info("No dataset provided, creating one")
            dataset = self.create_dataset()
            
        # Create or reset model if not provided
        if model is None:
            if self._reference_model is None:
                logger.info("Creating new model for evaluation")
                self._reference_model = self.create_model(dataset)
                self._reference_state_dict = self._reference_model.state_dict().copy()
                model = self._reference_model
            else:
                logger.debug("Using cached reference model (reset to initial weights)")
                model = self._reference_model
                model.load_state_dict(self._reference_state_dict)

        # --- Training Configuration ---
        training_config = {
            'num_iterations': self.config.get('training_iterations', 10),  # Default: 10 iterations
            'batch_size': self.loss_config.get('batch_size', 32),
            'learning_rate': self.config.get('learning_rate', 0.001),
            'eval_metric': self.config.get('eval_metric', 'accuracy'),  # 'accuracy', 'cross_entropy'
            'optimizer_type': self.config.get('optimizer_type', 'adam'),
            'early_stopping_patience': self.config.get('early_stopping_patience', 3),  # Default patience: 3 iterations
            'min_improvement': self.config.get('min_improvement', 0.001)  # Minimum improvement to consider progress
        }
        logger.info(f"Training configuration: {training_config}")

        interpreter = None  # Ensure interpreter is cleaned up
        try:
            # Check for stop signal before interpreter setup
            if stop_flag and stop_flag.is_stopped():
                logger.info("Loss evaluation stopped before interpreter setup")
                return -np.inf
                
            # --- Setup Genetic Interpreter for Loss Function ---
            logger.debug("Setting up genetic interpreter for loss function")
            interpreter = GeneticInterpreter(config={'debug': self.loss_config.get('debug_level', 0) > 1})
            
            try:
                logger.debug(f"Loading program with {len(genetic_code)} instructions")
                interpreter.load_program(genetic_code)
                logger.debug("Program loaded successfully")
            except Exception as load_err:
                logger.error(f"Error loading genetic code: {load_err}")
                logger.debug(f"Load error details: {traceback.format_exc()}")
                return -np.inf

            # --- Create optimizer ---
            optimizer_type = training_config['optimizer_type'].lower()
            if optimizer_type == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=training_config['learning_rate'])
            elif optimizer_type == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=training_config['learning_rate'])
            else:
                logger.warning(f"Unknown optimizer type '{optimizer_type}', defaulting to Adam")
                optimizer = torch.optim.Adam(model.parameters(), lr=training_config['learning_rate'])
            
            # --- Get training data ---
            train_x = dataset.get('trainX')
            train_y = dataset.get('trainY')
            if train_x is None or train_y is None:
                logger.error("Training dataset missing 'trainX' or 'trainY'")
                return -np.inf
            
            # Check for stop signal before training
            if stop_flag and stop_flag.is_stopped():
                logger.info("Loss evaluation stopped before training")
                return -np.inf
                
            # --- Training Loop ---
            logger.info(f"Starting training for {training_config['num_iterations']} iterations")
            batch_size = training_config['batch_size']
            num_samples = train_x.size(0)
            
            model.train()
            training_history = []
            total_loss = 0.0
            
            # Early stopping variables
            best_loss = float('inf')
            patience_counter = 0
            patience = training_config['early_stopping_patience']
            min_improvement = training_config['min_improvement']
            
            # Run for fixed number of iterations
            for iteration in range(training_config['num_iterations']):
                # Check for stop signal at each iteration
                if stop_flag and stop_flag.is_stopped():
                    logger.info(f"Loss evaluation stopped during training iteration {iteration}")
                    return -np.inf
                    
                # Sample a random batch for this iteration
                batch_indices = torch.randint(0, num_samples, (batch_size,))
                
                # Get batch data
                batch_x = train_x[batch_indices]
                batch_y = train_y[batch_indices]
                
                # Forward pass
                optimizer.zero_grad()
                batch_pred = model(batch_x)
                
                # Calculate loss using genetic loss function
                try:
                    # Initialize interpreter with current batch data
                    interpreter_inputs = {
                        'y_true': batch_y,  # S0: True values
                        'y_pred': batch_pred  # S1: Predicted values
                    }
                    interpreter.initialize(inputs=interpreter_inputs)
                    
                    # Execute genetic loss function
                    loss_tensor = interpreter.execute()
                    
                    # Check if loss is valid tensor
                    if loss_tensor is None or not isinstance(loss_tensor, torch.Tensor):
                        logger.warning("Genetic loss function did not return a valid tensor.")
                        return -np.inf
                        
                    # Ensure the loss is a scalar
                    if loss_tensor.numel() != 1:
                        logger.warning(f"Loss function returned non-scalar tensor (shape: {loss_tensor.shape}). Taking mean.")
                        loss_value = loss_tensor.mean()
                    else:
                        loss_value = loss_tensor
                        
                    # Check for NaN/Inf
                    if torch.isnan(loss_value) or torch.isinf(loss_value):
                        logger.warning(f"Loss value is NaN or Infinity ({loss_value.item()}). Terminating training.")
                        return -np.inf
                        
                    # Backward pass and optimization
                    loss_value.backward()
                    optimizer.step()
                    
                    # Log the current iteration loss
                    current_loss = loss_value.item()
                    total_loss += current_loss
                    training_history.append(current_loss)
                    
                    # Log every few iterations
                    if (iteration + 1) % max(1, training_config['num_iterations'] // 5) == 0:
                        logger.info(f"Iteration {iteration+1}/{training_config['num_iterations']}, Loss: {current_loss:.6f}, Avg Loss: {total_loss/(iteration+1):.6f}")
                    
                    # Early stopping check
                    if current_loss < best_loss - min_improvement:
                        # We've improved by at least min_improvement
                        best_loss = current_loss
                        patience_counter = 0
                        logger.debug(f"New best loss: {best_loss:.6f}")
                    else:
                        # No significant improvement
                        patience_counter += 1
                        logger.debug(f"No improvement for {patience_counter} iterations. Best: {best_loss:.6f}, Current: {current_loss:.6f}")
                        
                        if patience_counter >= patience:
                            logger.info(f"Early stopping triggered after {iteration+1} iterations. No improvement for {patience} iterations.")
                            return -np.inf  # Early stopping, return poor fitness
                        
                except Exception as exec_err:
                    logger.error(f"Error during loss calculation or backprop: {exec_err}")
                    logger.debug(f"Error details: {traceback.format_exc()}")
                    return -np.inf
            
            # Check for stop signal before validation
            if stop_flag and stop_flag.is_stopped():
                logger.info("Loss evaluation stopped before validation")
                return -np.inf
                
            # --- Validation ---
            logger.info("Training complete. Evaluating on validation set.")
            val_x = dataset.get('valX')
            val_y = dataset.get('valY')
            
            if val_x is None or val_y is None:
                logger.error("Validation dataset missing 'valX' or 'valY'")
                return -np.inf
            
            # Set model to evaluation mode
            model.eval()
            
            try:
                with torch.no_grad():
                    # Generate predictions
                    val_pred = model(val_x)
                    
                    # Calculate validation metric
                    metric_type = training_config['eval_metric'].lower()
                    
                    if metric_type == 'accuracy':
                        # For classification: get class predictions and compare
                        if len(val_y.shape) > 1 and val_y.shape[1] > 1:  # one-hot encoded
                            _, predicted_classes = torch.max(val_pred, dim=1)
                            _, true_classes = torch.max(val_y, dim=1)
                        else:  # class indices
                            _, predicted_classes = torch.max(val_pred, dim=1)
                            true_classes = val_y
                        
                        # Calculate accuracy
                        correct = (predicted_classes == true_classes).sum().item()
                        total = true_classes.size(0)
                        accuracy = correct / total
                        logger.info(f"Validation Accuracy: {accuracy:.4f}")
                        
                        # Return accuracy as fitness (higher is better)
                        fitness = accuracy
                        
                    elif metric_type == 'cross_entropy':
                        # Calculate cross-entropy loss (lower is better)
                        if len(val_y.shape) > 1 and val_y.shape[1] > 1:  # one-hot encoded
                            ce_loss = torch.nn.functional.cross_entropy(val_pred, val_y)
                        else:  # class indices
                            ce_loss = torch.nn.functional.cross_entropy(val_pred, val_y)
                        
                        logger.info(f"Validation Cross-Entropy: {ce_loss.item():.6f}")
                        
                        # Return negative cross-entropy as fitness (higher is better)
                        fitness = -ce_loss.item()
                        
                    elif metric_type == 'mse':
                        # Calculate MSE (lower is better)
                        mse_loss = torch.nn.functional.mse_loss(val_pred, val_y)
                        logger.info(f"Validation MSE: {mse_loss.item():.6f}")
                        
                        # Return negative MSE as fitness (higher is better)
                        fitness = -mse_loss.item()
                        
                    else:
                        logger.warning(f"Unknown eval_metric '{metric_type}', defaulting to accuracy")
                        # Default to accuracy
                        _, predicted_classes = torch.max(val_pred, dim=1)
                        if len(val_y.shape) > 1 and val_y.shape[1] > 1:
                            _, true_classes = torch.max(val_y, dim=1)
                        else:
                            true_classes = val_y
                        
                        correct = (predicted_classes == true_classes).sum().item()
                        total = true_classes.size(0)
                        accuracy = correct / total
                        logger.info(f"Validation Accuracy: {accuracy:.4f}")
                        
                        # Return accuracy as fitness
                        fitness = accuracy
            
            except Exception as val_err:
                logger.error(f"Error during validation: {val_err}")
                logger.debug(f"Validation error details: {traceback.format_exc()}")
                return -np.inf
                
            logger.debug(f"Evaluation successful, fitness: {fitness}")
            return fitness

        except Exception as e:
            logger.error(f"Error during loss evaluation: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return -np.inf  # Return worst fitness on error
        
        finally:
            # Clean up interpreter resources if necessary
            if interpreter:
                try:
                    logger.debug("Disposing interpreter resources")
                    interpreter.dispose()
                except Exception as dispose_err:
                    logger.warning(f"Error disposing interpreter resources: {dispose_err}")


    def interpret_result(self, result: Dict) -> Dict:
        """
        Interpret the final evolved loss function.

        Args:
            result: Dictionary containing evolution results (fitness, genetic_code).

        Returns:
            Dictionary with interpretation, including python function string.
        """
        logger.info(f"Interpreting result for LossStrategy. Best fitness: {-result.get('fitness', np.inf):.6f}")
        genetic_code = result.get('genetic_code')
        interpretation = {
            'component_type': 'loss',
            'best_fitness_raw': result.get('fitness', -np.inf),
            'best_loss_value': -result.get('fitness', np.inf), # Convert back to positive loss
            'genetic_code_length': len(genetic_code) if genetic_code else 0,
            'evolved_representation': None, # Placeholder for python code
            'notes': []
        }

        if not genetic_code:
            interpretation['notes'].append("No genetic code found in result.")
            return interpretation

        try:
            # Convert genetic code to Python function string
            python_code = FunctionConverter.genetic_code_to_python(genetic_code)
            interpretation['evolved_representation'] = python_code
            interpretation['notes'].append("Successfully converted genetic code to Python function.")

            # Optional: Add checks for standard functions
            if FunctionConverter._is_genetic_mse(genetic_code):
                 interpretation['notes'].append("Evolved function matches standard MSE.")
            elif FunctionConverter._is_genetic_mae(genetic_code):
                 interpretation['notes'].append("Evolved function matches standard MAE.")
            elif FunctionConverter._is_genetic_bce(genetic_code):
                 interpretation['notes'].append("Evolved function matches standard BCE.")

        except Exception as e:
            logger.error(f"Failed to convert genetic code to Python: {e}")
            interpretation['notes'].append(f"Failed to convert genetic code to Python: {e}")
            interpretation['evolved_representation'] = "# Conversion Error"

        return interpretation