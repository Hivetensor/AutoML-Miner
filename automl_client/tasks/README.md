# Task Strategies for AutoML

This module provides a strategy pattern for different machine learning tasks in the AutoML system. It allows for easy swapping of different problem domains without modifying the core evolution engine.

## Overview

The `TaskStrategy` interface defines the contract for all task implementations:

```python
class TaskStrategy(ABC):
    @abstractmethod
    def prepare_dataset(self) -> Dict[str, torch.Tensor]:
        """Prepare dataset specific to this task."""
        pass
        
    @abstractmethod
    def get_model_factory(self) -> Callable:
        """Get a factory function that creates models for this task."""
        pass
        
    @abstractmethod
    def create_initial_solution(self) -> List:
        """Create an initial solution for this task."""
        pass
        
    @abstractmethod
    def interpret_result(self, result: Any) -> Dict:
        """Interpret the evolved solution in task-specific terms."""
        pass
        
    @abstractmethod
    def get_task_config(self) -> Dict:
        """Get task-specific configuration."""
        pass
```

## Available Task Strategies

### MNISTClassificationTask

A task strategy for MNIST-like image classification problems. It creates a dataset with random values that mimic the MNIST dataset structure.

```python
from automl_client.tasks import MNISTClassificationTask

# Create with default configuration
mnist_task = MNISTClassificationTask()

# Or with custom configuration
mnist_task = MNISTClassificationTask({
    'num_train_samples': 500,
    'num_test_samples': 100,
    'input_shape': 784,  # 28x28 pixels
    'num_classes': 10,
    'hidden_size': 128
})
```

### SimpleRegressionTask

A task strategy for regression problems. It creates a dataset with a linear relationship and configurable noise.

```python
from automl_client.tasks import SimpleRegressionTask

# Create with default configuration
regression_task = SimpleRegressionTask()

# Or with custom configuration
regression_task = SimpleRegressionTask({
    'num_train_samples': 500,
    'num_test_samples': 100,
    'input_dim': 5,
    'output_dim': 1,
    'hidden_size': 32,
    'noise_level': 0.05
})
```

## Using Task Strategies with EvolutionEngine

The `EvolutionEngine` accepts a task strategy in its constructor:

```python
from automl_client.evolution_engine import EvolutionEngine
from automl_client.tasks import MNISTClassificationTask

# Create a task strategy
task = MNISTClassificationTask()

# Create an evolution engine with the task
engine = EvolutionEngine(task_strategy=task)

# Use the engine as normal
result = engine.evolve_function(...)
```

## Creating Custom Task Strategies

To create a custom task strategy, implement the `TaskStrategy` interface:

```python
from automl_client.tasks.strategy import TaskStrategy
import torch
import torch.nn as nn

class MyCustomTask(TaskStrategy):
    def __init__(self, config=None):
        self.config = {
            # Default configuration
            'param1': 100,
            'param2': 200,
            **(config or {})
        }
    
    def prepare_dataset(self):
        # Create and return your dataset
        return {
            'trainX': torch.tensor(...),
            'trainY': torch.tensor(...),
            'valX': torch.tensor(...),
            'valY': torch.tensor(...)
        }
    
    def get_model_factory(self):
        def create_model():
            # Create and return a PyTorch model
            return nn.Sequential(...)
        return create_model
    
    def create_initial_solution(self):
        # Create and return initial genetic code
        return [...]
    
    def interpret_result(self, result):
        # Interpret the result in task-specific terms
        return {
            'task_type': 'my_custom_task',
            'metrics': {
                'metric1': ...,
                'metric2': ...
            }
        }
    
    def get_task_config(self):
        return self.config
```

## Example

See `examples/task_strategy_example.py` for a complete example of using different task strategies.
