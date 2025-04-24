"""Centralized configuration management for AutoML components."""

import json
import os
from typing import Dict, Any, Optional

class ConfigManager:
    """Manages configuration for all AutoML components."""
    
    def __init__(self, config_path: Optional[str] = None, default_config: Optional[Dict] = None):
        """
        Initialize the ConfigManager.
        
        Args:
            config_path: Path to JSON config file
            default_config: Default configuration dictionary
        """
        self.config = {
            'evolution': {
                'population_size': 20,
                'generations': 10000,
                'mutation_rate': 0.2,
                'tournament_size': 3,
                'batch_size': 32,
                'elite_count': 2,
                'crossover_rate': 0.7,
                'evaluation_rounds': 3,
                'debug_logging': True,
                'log_interval': 100
            },
            'evaluation': {
                'evaluation_rounds': 3,
                'batch_size': 32,
                'log_interval': 500,
                'debug_level': 1
            },
            'interpreter': {
                'max_computation_factor': 10,
                'min_computation_budget': 1000
            },
            'dataset': {
                'num_train_samples': 1000,
                'num_test_samples': 200,
                'input_shape': (28*28,),
                'num_classes': 10
            },
            'model': {
                'hidden_layers': [64],
                'activation': 'relu'
            }
        }
        
        # Override with provided default config
        if default_config:
            self._deep_update(self.config, default_config)
            
        # Override with file config if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                self._deep_update(self.config, file_config)
                
    def _deep_update(self, original: Dict, update: Dict) -> None:
        """Recursively update a dictionary."""
        for key, value in update.items():
            if isinstance(value, dict) and key in original:
                self._deep_update(original[key], value)
            else:
                original[key] = value
                
    def get_evolution_config(self) -> Dict[str, Any]:
        """Get configuration for evolution engine."""
        return self.config.get('evolution', {})
        
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get configuration for evaluation strategies."""
        return self.config.get('evaluation', {})
        
    def get_interpreter_config(self) -> Dict[str, Any]:
        """Get configuration for genetic interpreter."""
        return self.config.get('interpreter', {})
        
    def get_dataset_config(self) -> Dict[str, Any]:
        """Get configuration for dataset providers."""
        return self.config.get('dataset', {})
        
    def get_model_config(self) -> Dict[str, Any]:
        """Get configuration for model factories."""
        return self.config.get('model', {})
        
    def get_all_config(self) -> Dict[str, Any]:
        """Get the complete configuration dictionary."""
        return self.config
