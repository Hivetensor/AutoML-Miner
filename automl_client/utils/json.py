"""JSON Utility functions for handling special float values and other serialization issues."""

import math
import json
import numpy as np
from typing import Any, Dict, List, Union, Optional

# Default replacement values for unsupported float types
DEFAULT_REPLACEMENT = {
    'inf': 1e6,            # Replace infinity with large positive value
    '-inf': -1e6,          # Replace negative infinity with large negative value  
    'nan': -1e6,           # Replace NaN with large negative value
}

def sanitize_float_for_json(value: float, replacement_values: Optional[Dict[str, float]] = None) -> float:
    """
    Sanitize a float value to be JSON compatible.
    
    Args:
        value: The float value to sanitize
        replacement_values: Optional dictionary mapping 'inf', '-inf', 'nan' to replacement values
        
    Returns:
        A JSON-compatible float value
    """
    replacements = replacement_values or DEFAULT_REPLACEMENT
    
    if math.isinf(value):
        if value > 0:
            return replacements.get('inf', 1e6)
        else:
            return replacements.get('-inf', -1e6)
    elif math.isnan(value):
        return replacements.get('nan', -1e6)
    return value

def sanitize_for_json(obj: Any, replacement_values: Optional[Dict[str, float]] = None) -> Any:
    """
    Recursively sanitize an object to ensure all values are JSON compatible.
    
    Args:
        obj: Any object (dict, list, float, etc.) to sanitize
        replacement_values: Optional dictionary mapping 'inf', '-inf', 'nan' to replacement values
        
    Returns:
        A JSON-compatible version of the object
    """
    replacements = replacement_values or DEFAULT_REPLACEMENT
    
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v, replacements) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item, replacements) for item in obj]
    elif isinstance(obj, (float, np.floating)):
        return sanitize_float_for_json(float(obj), replacements)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist(), replacements)
    elif obj is None or isinstance(obj, (bool, int, str)):
        return obj
    else:
        # Try to convert to a basic type
        try:
            return sanitize_for_json(float(obj), replacements)
        except (TypeError, ValueError):
            try:
                return str(obj)
            except:
                return "UNCONVERTIBLE_OBJECT"

class JSONSafeEncoder(json.JSONEncoder):
    """JSON Encoder subclass that handles special float values."""
    
    def __init__(self, *args, replacement_values=None, **kwargs):
        self.replacement_values = replacement_values or DEFAULT_REPLACEMENT
        super().__init__(*args, **kwargs)
    
    def default(self, obj):
        # Handle numpy types
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return sanitize_float_for_json(float(obj), self.replacement_values)
        elif isinstance(obj, np.ndarray):
            return sanitize_for_json(obj.tolist(), self.replacement_values)
        
        # Use default behavior for other types
        return super().default(obj)

def json_dumps(obj, replacement_values=None, **kwargs):
    """
    Safe JSON serialization function that handles special float values.
    
    Args:
        obj: Object to serialize
        replacement_values: Optional replacement map for special values
        **kwargs: Additional arguments for json.dumps
        
    Returns:
        JSON string representation of the object
    """
    replacements = replacement_values or DEFAULT_REPLACEMENT
    sanitized_obj = sanitize_for_json(obj, replacements)
    return json.dumps(sanitized_obj, **kwargs)

def json_loads(json_str, **kwargs):
    """
    Wrapper around json.loads for consistency.
    
    Args:
        json_str: JSON string to parse
        **kwargs: Additional arguments for json.loads
        
    Returns:
        Parsed Python object
    """
    return json.loads(json_str, **kwargs)