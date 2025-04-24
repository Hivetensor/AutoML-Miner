import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ModelFactory:
    @staticmethod
    def create_model(component_type: str, config: Optional[Dict] = None) -> nn.Module:

        config = config or {}
        
        # Extract and use seed if provided, default to 42 for deterministic behavior
        seed = config.get("model_seed", 42)
        ModelFactory.set_seed(seed)
        logger.info(f"Creating {component_type} model with seed {seed}")

        """Create model appropriate for the component type."""
        if component_type == "loss":
            return ModelFactory._create_loss_model(config)
        elif component_type == "optimizer":
            return ModelFactory._create_optimizer_model(config)
        elif component_type == "activation":
            return ModelFactory._create_activation_model(config)
        elif component_type == "regularization":
            return ModelFactory._create_regularization_model(config)
        else:
            return ModelFactory._create_default_model(config)

    @staticmethod
    def _create_loss_model(config: Optional[Dict] = None) -> nn.Module:
        """Create a simple model for loss function evaluation."""
        input_dim = config.get("input_dim", 10) if config else 10
        output_dim = config.get("output_dim", 1) if config else 1
        hidden_dim = config.get("hidden_dim", 8) if config else 8
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    @staticmethod
    def _create_optimizer_model(config: Optional[Dict] = None) -> nn.Module:
        """Create a model for optimizer evaluation."""
        input_dim = config.get("input_dim", 10) if config else 10
        output_dim = config.get("output_dim", 1) if config else 1
        hidden_dim = config.get("hidden_dim", 64) if config else 64
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    @staticmethod
    def _create_activation_model(config: Optional[Dict] = None) -> nn.Module:
        """Create a model for activation function evaluation."""
        input_dim = config.get("input_dim", 10) if config else 10
        output_dim = config.get("output_dim", 1) if config else 1
        hidden_dim = config.get("hidden_dim", 8) if config else 8
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim)
        )

    @staticmethod
    def _create_regularization_model(config: Optional[Dict] = None) -> nn.Module:
        """Create a model for regularization evaluation."""
        input_dim = config.get("input_dim", 10) if config else 10
        output_dim = config.get("output_dim", 1) if config else 1
        hidden_dim = config.get("hidden_dim", 32) if config else 32
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, output_dim)
        )

    @staticmethod
    def _create_default_model(config: Optional[Dict] = None) -> nn.Module:
        """Fallback model if component type is unknown."""
        input_dim = config.get("input_dim", 10) if config else 10
        output_dim = config.get("output_dim", 1) if config else 1
        return nn.Linear(input_dim, output_dim)
    
    @staticmethod
    def set_seed(seed=42):
        """Set deterministic seed for model initialization."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
