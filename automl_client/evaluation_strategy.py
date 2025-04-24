"""Evaluation Strategies for Genetic Functions in PyTorch."""

import torch
import logging
import traceback
import json
from typing import Optional, Dict, Any, List
from .genetic.interpreter import GeneticInterpreter

# Configure logger for this module
logger = logging.getLogger(__name__)

def log_tensor_info(name: str, tensor: torch.Tensor) -> None:
    """Log detailed information about a tensor."""
    try:
        logger.debug(f"Tensor '{name}' info: shape={tensor.shape}, dtype={tensor.dtype}, "
                    f"requires_grad={tensor.requires_grad}, "
                    f"has_grad_fn={tensor.grad_fn is not None}, "
                    f"device={tensor.device}, "
                    f"min={tensor.min().item():.6f}, max={tensor.max().item():.6f}")
    except Exception as e:
        logger.debug(f"Error logging tensor '{name}' info: {e}")

def log_genetic_code(genetic_code: List) -> None:
    """Log information about the genetic code structure."""
    try:
        if not isinstance(genetic_code, list):
            logger.debug(f"Invalid genetic code type: {type(genetic_code)}")
            return
            
        op_codes = {}
        for instruction in genetic_code:
            if not isinstance(instruction, list) or len(instruction) == 0:
                continue
                
            op_code = instruction[0]
            op_codes[op_code] = op_codes.get(op_code, 0) + 1
            
        logger.debug(f"Genetic code length: {len(genetic_code)}")
        logger.debug(f"Operation code distribution: {json.dumps(op_codes)}")
        
        # Log a sample of the genetic code (first 5 instructions)
        sample = genetic_code[:5] if len(genetic_code) > 5 else genetic_code
        logger.debug(f"Sample instructions: {sample}")
    except Exception as e:
        logger.debug(f"Error logging genetic code: {e}")

class EvaluationStrategy:
    """Interface for pluggable fitness evaluation strategies."""
    
    def evaluate(self, genetic_code: list, dataset: dict) -> float:
        """
        Evaluates the fitness of an evolved function.
        
        Args:
            genetic_code: The genetic code to evaluate
            dataset: Training/validation dataset
            
        Returns:
            Fitness score (higher is better)
        """
        raise NotImplementedError("evaluate() must be implemented by concrete strategy")

class DefaultPytorchEvaluation(EvaluationStrategy):
    """Default evaluation strategy using PyTorch model training."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the evaluation strategy.
        
        Args:
            config: Configuration dictionary with:
                - evaluation_rounds: Number of training epochs
                - batch_size: Batch size for training
                - model_factory: Optional function to create models
        """
        self.config = {
            'evaluation_rounds': 3,
            'batch_size': 32,
            'log_interval': 500,  # Log every 500 evaluations
            'debug_level': 1,     # 0=minimal, 1=normal, 2=verbose
            **(config or {})
        }
        self.evaluation_count = 0
        logger.debug(f"DefaultPytorchEvaluation initialized with config: {self.config}")
    
    def evaluate(self, genetic_code: list, dataset: dict) -> float:
        """
        Evaluate genetic code by training a model with it as loss function.
        
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
            logger.debug(f"Starting evaluation #{self.evaluation_count}, genetic code length: {len(genetic_code)}")
            
        # Log genetic code structure
        if debug_level >= 1:
            logger.debug(f"Evaluation #{self.evaluation_count}: Analyzing genetic code structure")
            log_genetic_code(genetic_code)
        
        # Convert genetic code to loss function
        try:
            logger.debug(f"Evaluation #{self.evaluation_count}: Decoding genetic code to loss function")
            # Create interpreter config with computation budget settings
            interpreter_config = {
                'max_computation_factor': 10,  # Default: 10x program length
                'min_computation_budget': 1000  # Default: minimum 1000 instructions
            }
            loss_fn = GeneticInterpreter.decode_to_function(genetic_code, interpreter_config)
            if should_log:
                logger.debug(f"Successfully decoded genetic code to loss function")
        except Exception as e:
            error_details = traceback.format_exc()
            logger.debug(f"Failed to decode genetic code: {e}")
            logger.debug(f"Decode error details: {error_details}")
            return 0.0
        
        # Create a model using factory if provided, otherwise use default
        model_factory = self.config.get('model_factory')
        if model_factory:
            model = model_factory()
            logger.debug(f"Created model using provided factory: {model}")
        else:
            # Default model for MNIST
            model = torch.nn.Sequential(
                torch.nn.Linear(28*28, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 10),
                torch.nn.Softmax(dim=1)
            )
            logger.debug("Created default model (no factory provided)")
        
        # Log model info
        if debug_level >= 2:
            logger.debug(f"Model architecture: {model}")
            for name, param in model.named_parameters():
                logger.debug(f"Parameter {name}: shape={param.shape}, requires_grad={param.requires_grad}")
        
        # Set up optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Log dataset info
        if debug_level >= 1:
            for key, tensor in dataset.items():
                log_tensor_info(key, tensor)
        
        # Early stopping configuration
        early_stopping_patience = self.config.get('early_stopping_patience', 3)  # Default: 3 epochs
        min_improvement = self.config.get('min_improvement', 0.001)  # Minimum improvement threshold
        best_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        train_losses = []
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
                if debug_level >= 2 and i == 0:  # Only log first batch
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
                    
                    # Optimizer step
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
            
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
            train_losses.append(avg_epoch_loss)
            
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
        try:
            with torch.no_grad():
                val_outputs = model(dataset['valX'])
                
                if debug_level >= 1:
                    log_tensor_info("val_outputs", val_outputs)
                
                _, predicted = torch.max(val_outputs, 1)
                _, labels = torch.max(dataset['valY'], 1)
                correct = (predicted == labels).sum().item()
                accuracy = correct / len(dataset['valY'])
                
                if should_log:
                    logger.debug(f"Evaluation #{self.evaluation_count} complete, Accuracy: {accuracy:.6f}")
                
                return accuracy
        except Exception as e:
            error_details = traceback.format_exc()
            logger.debug(f"Error during evaluation: {e}")
            logger.debug(f"Evaluation error details: {error_details}")
            return 0.0

class OptimizerEvaluation(EvaluationStrategy):
    """Evaluation strategy that allows testing with different optimizers."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the evaluation strategy.
        
        Args:
            config: Configuration dictionary with:
                - optimizer: Name of optimizer ('Adam', 'SGD', etc.)
                - learning_rate: Learning rate
                - evaluation_rounds: Number of training epochs
                - batch_size: Batch size for training
        """
        self.config = {
            'optimizer': 'Adam',
            'learning_rate': 0.001,
            'evaluation_rounds': 3,
            'batch_size': 32,
            'log_interval': 500,  # Log every 500 evaluations
            'debug_level': 1,     # 0=minimal, 1=normal, 2=verbose
            **(config or {})
        }
        self.evaluation_count = 0
        logger.debug(f"OptimizerEvaluation initialized with config: {self.config}")
    
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
                
                if should_log or debug_level >= 1:
                    avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
                    logger.debug(f"Epoch {epoch+1}/{self.config['evaluation_rounds']}, Loss: {avg_loss:.6f}")
            
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

class CustomMetricEvaluation(EvaluationStrategy):
    """Evaluation strategy that uses a custom metric function."""
    
    def __init__(self, config: Dict):
        """
        Initialize the evaluation strategy.
        
        Args:
            config: Configuration dictionary with:
                - metric_fn: Function that calculates fitness score
                - evaluation_rounds: Number of training epochs
                - batch_size: Batch size for training
        """
        if 'metric_fn' not in config:
            raise ValueError("CustomMetricEvaluation requires a metric_fn in config")
            
        self.config = {
            'evaluation_rounds': 3,
            'batch_size': 32,
            'log_interval': 500,  # Log every 500 evaluations
            **config
        }
        self.evaluation_count = 0
        logger.debug("CustomMetricEvaluation initialized")
    
    def evaluate(self, genetic_code: list, dataset: dict) -> float:
        """
        Evaluate genetic code using custom metric function.
        
        Args:
            genetic_code: Genetic code to evaluate
            dataset: Dictionary with trainX, trainY, valX, valY tensors
            
        Returns:
            Fitness score calculated by metric_fn
        """
        self.evaluation_count += 1
        
        # Log at regular intervals
        should_log = (self.evaluation_count % self.config['log_interval'] == 0)
        if should_log:
            logger.debug(f"Starting evaluation #{self.evaluation_count} with custom metric")
        
        try:
            # Create interpreter config with computation budget settings
            interpreter_config = {
                'max_computation_factor': 10,  # Default: 10x program length
                'min_computation_budget': 1000  # Default: minimum 1000 instructions
            }
            loss_fn = GeneticInterpreter.decode_to_function(genetic_code, interpreter_config)
            model = torch.nn.Sequential(
                torch.nn.Linear(28*28, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 10),
                torch.nn.Softmax(dim=1)
            )
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Training
            for epoch in range(self.config['evaluation_rounds']):
                model.train()
                epoch_losses = []
                
                for i in range(0, len(dataset['trainX']), self.config['batch_size']):
                    batch_X = dataset['trainX'][i:i+self.config['batch_size']]
                    batch_Y = dataset['trainY'][i:i+self.config['batch_size']]
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = loss_fn(batch_Y, outputs)
                    loss.backward()
                    optimizer.step()
                    epoch_losses.append(loss.item())
                
                if should_log:
                    avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
                    logger.debug(f"Epoch {epoch+1}/{self.config['evaluation_rounds']}, Loss: {avg_loss:.6f}")
            
            # Calculate fitness using custom metric
            model.eval()
            with torch.no_grad():
                val_outputs = model(dataset['valX'])
                fitness = self.config['metric_fn']({
                    'outputs': val_outputs,
                    'labels': dataset['valY'],
                    'model': model,
                    'genetic_code': genetic_code
                })
                
                if should_log:
                    logger.debug(f"Evaluation complete, Fitness: {fitness:.6f}")
                
                return fitness
                
        except Exception as e:
            logger.debug(f"Error in evaluation: {e}")
            return 0.0
