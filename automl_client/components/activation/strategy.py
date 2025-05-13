"""Strategy for evolving and evaluating Activation Function components."""

import torch
import numpy as np
import logging
import traceback
import json
from typing import List, Dict, Any, Optional

from ..base import AIComponentStrategy
from ...genetic.interpreter import GeneticInterpreter
from ...genetic.function_converter import FunctionConverter
from ...data.dataset_factory import DatasetFactory
from ...models.model_factory import ModelFactory

from ...genetic.strategies import RegisterValidator
from ...utils.fec import get_fixed_tensor, fec_lookup_or_evaluate
# Import the specific library once created, e.g.:
from .library import ActivationLibrary # Use component-specific library

logger = logging.getLogger(__name__)

class DynamicActivation(torch.nn.Module):
    """Wrapper module for genetic activation functions"""
    def __init__(self, genetic_code):
        super().__init__()
        # Convert genetic code to its string representation for storage
        # This avoids "unhashable type: 'list'" errors when PyTorch tries to hash the module
        genetic_code = RegisterValidator.validate_and_fix_genetic_code(genetic_code)
        self._genetic_code_str = json.dumps(genetic_code)
        self.interpreter = GeneticInterpreter()
        logger.debug(f"Creating DynamicActivation with genetic code:\n{self._format_genetic_code(genetic_code)}")
        
    def _format_genetic_code(self, code):
        """Format genetic code for logging in a readable way"""
        if not isinstance(code, list):
            return f"Invalid code type: {type(code)}"
        return "\n".join([f"{i}: {instr}" for i, instr in enumerate(code)])
    
    @property
    def genetic_code(self):
        """Retrieve genetic code from storage"""
        try:
            return json.loads(self._genetic_code_str)
        except:
            logger.error("Failed to parse stored genetic code")
            return [[100, 1, 0], [453, 1]]  # Return identity function as fallback
        
    def forward(self, x):
        """Execute genetic code on input tensor with gradient preservation"""
        # Important: ensure x requires grad if we're in training mode
        if self.training and not x.requires_grad:
            x = x.detach().requires_grad_(True)
            
        # Execute with gradient tracking
        genetic_code = self.genetic_code
        self.interpreter.load_program(genetic_code)
        self.interpreter.initialize(inputs={'y_true': x})
        result = self.interpreter.execute()
        
        # Ensure result has proper gradient connection
        if self.training and x.requires_grad and not result.requires_grad:
            # Force gradient connection if missing
            dummy = torch.ones(1, requires_grad=True)
            result = result * dummy
            
        # Handle shape issues while preserving gradients
        if result.shape != x.shape:
            # Try to preserve gradient flow while fixing shape
            if result.numel() == 1:  # Scalar result
                result = result.expand_as(x)
            else:
                # Log warning but proceed with best effort
                logger.warning(f"Shape mismatch: input {x.shape}, output {result.shape}")
                
        return result

class ActivationStrategy(AIComponentStrategy):
    """Handles the evolution and evaluation of activation function components."""

    def __init__(self, config: Dict = None):
        """Initialize ActivationStrategy."""
        super().__init__(config)
        self.activation_config = {
            'evaluation_rounds': self.config.get('evaluation_rounds', 1),
            'batch_size': self.config.get('batch_size', 64),
            'debug_level': self.config.get('debug_level', 0),
            'dataset_name': self.config.get('dataset_name', 'mnist'),
            'dataset_config': self.config.get('dataset_config', {
                'flatten': True,
                'normalize': True
            }),
            'model_config': self.config.get('model_config', {
                'architecture': 'simple_mlp',
                'hidden_dim': 8
            }),
            'training_config': self.config.get('training_config', {
                'learning_rate': 0.001,
                'optimizer': 'adam',
                'iterations': 10
            })
        }
        logger.info(f"ActivationStrategy initialized with config: {self.activation_config}")
        self._reference_model = None
        self._reference_state_dict = None

    def _format_genetic_code(self, code):
        """Format genetic code for logging in a readable way"""
        if not isinstance(code, list):
            return f"Invalid code type: {type(code)}"
        return "\n".join([f"{i}: {instr}" for i, instr in enumerate(code)])
        
    def create_dataset(self) -> Dict[str, torch.Tensor]:
        """Create dataset using DatasetFactory based on config"""
        dataset_name = self.activation_config.get('dataset_name', 'mnist')
        dataset_config = self.activation_config.get('dataset_config', {})
        
        logger.info(f"Creating {dataset_name} dataset for activation evaluation")
        EVAL_SEED = 42
        dataset = DatasetFactory.create_dataset(dataset_name, dataset_config, eval_seed=EVAL_SEED)
        logger.info(f"Created dataset: { {k: v.shape for k, v in dataset.items()} }")
        return dataset

    def create_model(self, dataset: Dict[str, torch.Tensor], genetic_code: List = None) -> torch.nn.Module:
        """Create model with dynamic activation using ModelFactory"""
        # Get input/output dimensions from dataset
        input_dim = dataset['trainX'].shape[1]
        output_dim = dataset['trainY'].shape[1] if len(dataset['trainY'].shape) > 1 else 1
        
        # Get model config
        model_config = self.activation_config.get('model_config', {}).copy()
        model_config.update({
            'input_dim': input_dim,
            'output_dim': output_dim,
            'model_seed': self.activation_config.get('model_seed', 42)
        })
        
        # Create base model from factory
        logger.info(f"Creating model with config: {model_config}")
        model = ModelFactory.create_model("activation", model_config)
        
        # Replace activations with genetic code implementation
        if genetic_code:
            logger.info("Replacing model activations with genetic implementation")
            self._replace_activations(model, genetic_code)
            
        return model

    def _replace_activations(self, model: torch.nn.Module, genetic_code: List):
        """Recursively replace activation layers in model with genetic implementation"""
        for name, module in model.named_children():
            if isinstance(module, torch.nn.ModuleList):
                for i, submodule in enumerate(module):
                    self._replace_activations(submodule, genetic_code)
            elif isinstance(module, torch.nn.Sequential):
                self._replace_activations(module, genetic_code)
            elif isinstance(module, (torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Tanh)):
                # Replace standard activation with our genetic version
                model._modules[name] = DynamicActivation(genetic_code)
                
    def create_initial_solution(self) -> List:
        """
        Create an initial genetic code solution for an activation function.
        Defaults to ReLU (Rectified Linear Unit).
        """
        logger.info("Creating initial solution for ActivationStrategy (defaulting to ReLU).")
        try:
            # Use ActivationLibrary to get the standard ReLU genetic code
            relu_code = ActivationLibrary.get_function("relu")
            if not relu_code:
                logger.error("Could not retrieve standard ReLU code from ActivationLibrary! Falling back.")
                # Fallback basic structure if library fails unexpectedly
                relu_code = [
                    [100, 1, 0], [101, 2, 0.0], [300, 3, 1, 2], [453, 3] # ReLU
                ]
            return relu_code
        except Exception as e:
            logger.exception(f"Error creating initial activation solution: {e}")
            # Provide a minimal valid code (e.g., identity function) on error
            identity_code = ActivationLibrary.get_function("identity")
            if identity_code:
                return identity_code
            return [[100, 1, 0], [453, 1]] # Identity: LOAD R1, S0; RETURN R1

    def evaluate(self, genetic_code: List, dataset=None, model=None, stop_flag=None) -> float:
        # —— FAST EQUIVALENCE CHECK ——
        # 1) Ensure we have dataset so we can get input_dim
        if dataset is None:
            dataset = self.create_dataset()
        input_dim = dataset['trainX'].shape[1]

        # 2) Grab (and cache) a fixed 32×input_dim tensor
        test_x = get_fixed_tensor('activation_test_x', (32, input_dim))

        # 3) One forward pass through your dynamic activation
        dyn_act = DynamicActivation(genetic_code).eval()
        with torch.no_grad():
            out = dyn_act(test_x)

        # 4) Quantize + hash + cache
        arr = torch.round(out * 1e4).to(torch.float16).cpu().numpy()
        return fec_lookup_or_evaluate(arr.tobytes(),
            lambda: self._full_evaluate(genetic_code, dataset, model, stop_flag)
        )

    def _full_evaluate(self, genetic_code: List, dataset: Optional[Dict] = None, model: Optional[torch.nn.Module] = None, stop_flag=None) -> float:
        """
        Evaluate an activation function represented by genetic code.
        
        Args:
            genetic_code: The genetic code representing an activation function
            dataset: Optional dictionary with trainX, trainY, valX, valY tensors
            model: Optional pre-initialized model
            stop_flag: Optional StopFlag object to check for stopping
                
        Returns:
            Fitness score (higher is better) or -np.inf if stopped
        """
        # Check for stop signal before starting
        if stop_flag and stop_flag.is_stopped():
            logger.info("Activation evaluation stopped before starting")
            return -np.inf
            
        if not genetic_code:
            logger.warning("Evaluate called with empty genetic code.")
            return -np.inf
            
        # Log the received genetic code for debugging
        if isinstance(genetic_code, list):
            logger.info(f"Received genetic code for evaluation, length: {len(genetic_code)}")
            logger.debug(f"Genetic code to evaluate:\n{self._format_genetic_code(genetic_code)}")
        else:
            logger.warning(f"Received invalid genetic code type: {type(genetic_code)}")
                
        # Create dataset if not provided
        if dataset is None:
            logger.info("No dataset provided, creating one")
            dataset = self.create_dataset()
        
        # Create or reset model if not provided
        if model is None:
            if self._reference_model is None:
                logger.info("Creating new model for evaluation")
                # Create model WITH the genetic activation function
                self._reference_model = self.create_model(dataset, genetic_code)
                self._reference_state_dict = self._reference_model.state_dict().copy()
                model = self._reference_model
            else:
                logger.debug("Using cached reference model (reset to initial weights)")
                model = self._reference_model
                model.load_state_dict(self._reference_state_dict)
                # Replace activations with new genetic code
                self._replace_activations(model, genetic_code)

        try:
            # Check for stop signal before training setup
            if stop_flag and stop_flag.is_stopped():
                logger.info("Activation evaluation stopped before training setup")
                return -np.inf
                
            # Prepare training components
            train_x = dataset['trainX']
            train_y = dataset['trainY']
            val_x = dataset['valX']
            val_y = dataset['valY']
            
            if train_x is None or train_y is None or val_x is None or val_y is None:
                logger.error("Dataset missing required components.")
                return -np.inf
            
            # Get training config
            training_config = self.activation_config.get('training_config', {})
            learning_rate = training_config.get('learning_rate', 0.001)
            iterations = training_config.get('iterations', 10)
            batch_size = self.activation_config.get('batch_size', 64)
            
            # Early stopping config
            patience = self.config.get('early_stopping_patience', 3)  # Default: 3 iterations
            min_improvement = self.config.get('min_improvement', 0.001)  # Minimum improvement threshold
            
            # Setup optimizer and loss function
            optimizer_name = training_config.get('optimizer', 'adam').lower()
            if optimizer_name == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            elif optimizer_name == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            # Use standard loss function (not the genetic code!)
            loss_fn = torch.nn.CrossEntropyLoss()
            
            # Check for stop signal before training
            if stop_flag and stop_flag.is_stopped():
                logger.info("Activation evaluation stopped before training")
                return -np.inf
                
            # Train the model for a few epochs
            model.train()
            
            # Early stopping variables
            best_loss = float('inf')
            patience_counter = 0
            
            for iteration in range(iterations):
                # Check for stop at each iteration
                if stop_flag and stop_flag.is_stopped():
                    logger.info(f"Activation evaluation stopped during iteration {iteration}")
                    return -np.inf
                    
                total_loss = 0.0
                batches = 0
                
                # Process mini-batches
                for i in range(0, train_x.size(0), batch_size):
                    # Check for stop every few batches
                    if i % (batch_size * 5) == 0 and stop_flag and stop_flag.is_stopped():
                        logger.info(f"Activation evaluation stopped during batch processing at iteration {iteration}")
                        return -np.inf
                        
                    # Get mini-batch
                    end_idx = min(i + batch_size, train_x.size(0))
                    batch_x = train_x[i:end_idx]
                    batch_y = train_y[i:end_idx]
                    
                    # Forward pass (uses our genetic activation internally)
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = loss_fn(outputs, batch_y)
                    
                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batches += 1
                
                # Calculate average loss for this iteration
                avg_loss = total_loss / batches if batches > 0 else float('inf')
                logger.debug(f"Iteration {iteration+1}/{iterations}, Avg Loss: {avg_loss:.4f}")
                
                # Early stopping check
                if avg_loss < best_loss - min_improvement:
                    # We've improved significantly
                    best_loss = avg_loss
                    patience_counter = 0
                    logger.debug(f"New best loss: {best_loss:.6f}")
                else:
                    # No significant improvement
                    patience_counter += 1
                    logger.debug(f"No improvement for {patience_counter} iterations. Best: {best_loss:.6f}, Current: {avg_loss:.6f}")
                    
                    if patience_counter >= patience:
                        logger.info(f"Early stopping triggered after {iteration+1} iterations. No improvement for {patience} iterations.")
                        return -np.inf  # Early stopping, return poor fitness
            
            # Check for stop before validation
            if stop_flag and stop_flag.is_stopped():
                logger.info("Activation evaluation stopped before validation")
                return -np.inf
                
            # Evaluate on validation set
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for i in range(0, val_x.size(0), batch_size):
                    # Check for stop periodically during validation
                    if i % (batch_size * 5) == 0 and stop_flag and stop_flag.is_stopped():
                        logger.info("Activation evaluation stopped during validation")
                        return -np.inf
                        
                    end_idx = min(i + batch_size, val_x.size(0))
                    batch_x = val_x[i:end_idx]
                    batch_y = val_y[i:end_idx]
                    
                    outputs = model(batch_x)
                    _, predicted = torch.max(outputs.data, 1)
                    _, actual = torch.max(batch_y.data, 1) if batch_y.dim() > 1 else (None, batch_y)
                    
                    total += batch_y.size(0)
                    correct += (predicted == actual).sum().item()
            
            accuracy = correct / total
            logger.info(f"Validation accuracy with this activation: {accuracy:.4f}")
            
            # Return accuracy as fitness (higher is better)
            return accuracy
            
        except Exception as e:
            logger.error(f"Error during activation evaluation: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return -np.inf


    def interpret_result(self, result: Dict) -> Dict:
        """
        Interpret the final evolved activation function.

        Args:
            result: Dictionary containing evolution results (fitness, genetic_code).

        Returns:
            Dictionary with interpretation, including python function string.
        """
        logger.info(f"Interpreting result for ActivationStrategy. Best fitness (raw): {result.get('fitness', -np.inf):.6f}")
        genetic_code = result.get('genetic_code')
        interpretation = {
            'component_type': 'activation',
            'best_fitness_raw': result.get('fitness', -np.inf), # Raw fitness (0 if ok, -inf if error)
            'genetic_code_length': len(genetic_code) if genetic_code else 0,
            'evolved_representation': None, # Placeholder for python code
            'notes': ["Basic evaluation checks execution; real performance depends on model context."]
        }

        if not genetic_code:
            interpretation['notes'].append("No genetic code found in result.")
            return interpretation

        try:
            # Convert genetic code to Python function string
            python_code = FunctionConverter.genetic_code_to_python(genetic_code)
            interpretation['evolved_representation'] = python_code
            interpretation['notes'].append("Successfully converted genetic code to Python function.")

            # Optional: Add checks for standard functions once FunctionConverter supports them
            # if FunctionConverter._is_genetic_relu(genetic_code):
            #      interpretation['notes'].append("Evolved function matches standard ReLU.")
            # elif FunctionConverter._is_genetic_sigmoid(genetic_code):
            #      interpretation['notes'].append("Evolved function matches standard Sigmoid.")
            # etc.

        except Exception as e:
            logger.error(f"Failed to convert genetic code to Python: {e}")
            interpretation['notes'].append(f"Failed to convert genetic code to Python: {e}")
            interpretation['evolved_representation'] = "# Conversion Error"

        return interpretation
