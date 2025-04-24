"""Serialization utilities for genetic programs and populations."""
import math 
import json
import base64
import torch
import numpy as np
from typing import List, Dict, Any, Union, Optional

def serialize_genetic_code(genetic_code: List[List]) -> str:
    """
    Serialize genetic code to JSON string for transmission.
    
    Args:
        genetic_code: A list of instruction lists representing the genetic code
        
    Returns:
        JSON string representation of the genetic code
    """
    if not isinstance(genetic_code, list):
        raise ValueError("Genetic code must be a list")
    
    # Simply convert to JSON - genetic code is already a list of lists with basic types
    return json.dumps(genetic_code)

def deserialize_genetic_code(serialized_code: str) -> List[List]:
    """
    Deserialize genetic code from JSON string.
    
    Args:
        serialized_code: JSON string representation of genetic code
        
    Returns:
        Genetic code as a list of instruction lists
    """
    try:
        code = json.loads(serialized_code)
        
        # Validate the structure
        if not isinstance(code, list):
            raise ValueError("Deserialized genetic code must be a list")
        
        for instruction in code:
            if not isinstance(instruction, list):
                raise ValueError("Each instruction in genetic code must be a list")
            
            # Ensure first element is an op code (integer)
            if not instruction or not isinstance(instruction[0], (int, float)):
                raise ValueError("First element of instruction must be a numeric op code")
        
        return code
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format for genetic code")

def serialize_population(population: List[List[List]]) -> str:
    """
    Serialize an entire population of genetic codes.
    
    Args:
        population: List of genetic codes
        
    Returns:
        JSON string representation of the population
    """
    if not isinstance(population, list):
        raise ValueError("Population must be a list")
    
    # Convert each genetic code in the population
    serialized_population = [genetic_code for genetic_code in population]
    
    return json.dumps(serialized_population)

def deserialize_population(serialized_population: str) -> List[List[List]]:
    """
    Deserialize a population from JSON string.
    
    Args:
        serialized_population: JSON string representation of a population
        
    Returns:
        List of genetic codes representing the population
    """
    try:
        population = json.loads(serialized_population)
        
        # Validate the structure
        if not isinstance(population, list):
            raise ValueError("Deserialized population must be a list")
        
        # Verify each member is a valid genetic code
        validated_population = []
        for genetic_code in population:
            if not isinstance(genetic_code, list):
                raise ValueError("Each member of population must be a list (genetic code)")
            
            for instruction in genetic_code:
                if not isinstance(instruction, list):
                    raise ValueError("Each instruction in genetic code must be a list")
                
                # Ensure first element is an op code (integer)
                if not instruction or not isinstance(instruction[0], (int, float)):
                    raise ValueError("First element of instruction must be a numeric op code")
            
            validated_population.append(genetic_code)
        
        return validated_population
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format for population")

def serialize_tensor(tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Serialize a PyTorch tensor for transmission.
    
    Args:
        tensor: PyTorch tensor to serialize
        
    Returns:
        Dictionary representation of the tensor
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor")
    
    # Convert tensor to numpy and then to list
    tensor_np = tensor.detach().cpu().numpy()
    
    # Handle special cases like NaN and Inf
    if not np.isfinite(tensor_np).all():
        # Replace NaN and Inf with None in serialized form
        tensor_np = np.where(np.isfinite(tensor_np), tensor_np, None)
    
    # Serialize to base64 for efficiency with binary data
    tensor_bytes = tensor_np.tobytes()
    tensor_b64 = base64.b64encode(tensor_bytes).decode('utf-8')
    
    return {
        'data_b64': tensor_b64,
        'shape': tensor_np.shape,
        'dtype': str(tensor_np.dtype),
        'type': 'torch.Tensor'
    }

def deserialize_tensor(tensor_dict: Dict[str, Any]) -> torch.Tensor:
    """
    Deserialize a tensor from its dictionary representation.
    
    Args:
        tensor_dict: Dictionary representation of the tensor
        
    Returns:
        PyTorch tensor
    """
    if not isinstance(tensor_dict, dict) or 'type' not in tensor_dict or tensor_dict['type'] != 'torch.Tensor':
        raise ValueError("Input must be a serialized tensor dictionary")
    
    # Get tensor properties
    tensor_b64 = tensor_dict['data_b64']
    shape = tensor_dict['shape']
    dtype_str = tensor_dict['dtype']
    
    # Convert dtype string to numpy dtype
    dtype = np.dtype(dtype_str)
    
    # Decode base64 to bytes, then to numpy array
    tensor_bytes = base64.b64decode(tensor_b64)
    tensor_np = np.frombuffer(tensor_bytes, dtype=dtype).reshape(shape)
    
    # Convert to PyTorch tensor
    return torch.tensor(tensor_np)


def serialize_evolution_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize the result of an evolution task.
    
    Args:
        result: Evolution result dictionary
        
    Returns:
        JSON-serializable dictionary
    """
    import math
    
    if not isinstance(result, dict):
        raise ValueError("Evolution result must be a dictionary")
    
    # Create a copy of the result to avoid modifying the original
    sanitized_result = {
        "evolved_function": serialize_genetic_code(result["genetic_code"]),
        "parent_functions": result.get("parent_functions", []),
        "metadata": {}
    }
    
    # Handle metadata, focusing especially on generation_stats
    if 'metadata' in result and isinstance(result['metadata'], dict):
        metadata_copy = {}
        
        # Process each metadata field
        for key, value in result['metadata'].items():
            if key == 'generation_stats' and isinstance(value, list):
                sanitized_stats = []
                for stat_entry in value:
                    sanitized_entry = {}
                    for stat_key, stat_value in stat_entry.items():
                        if isinstance(stat_value, float):
                            if math.isnan(stat_value):
                                sanitized_entry[stat_key] = "NaN"
                            elif math.isinf(stat_value):
                                if stat_value > 0:
                                    sanitized_entry[stat_key] = "Infinity"
                                else:
                                    sanitized_entry[stat_key] = "-Infinity"
                            else:
                                sanitized_entry[stat_key] = stat_value
                        else:
                            sanitized_entry[stat_key] = stat_value
                    sanitized_stats.append(sanitized_entry)
                metadata_copy[key] = sanitized_stats
            elif isinstance(value, float):
                if math.isnan(value):
                    metadata_copy[key] = "NaN"
                elif math.isinf(value):
                    if value > 0:
                        metadata_copy[key] = "Infinity"
                    else:
                        metadata_copy[key] = "-Infinity"
                else:
                    metadata_copy[key] = value
            else:
                metadata_copy[key] = value
                
        sanitized_result["metadata"] = metadata_copy
    
    return sanitized_result

def _process_dict_for_json(d: Dict) -> Dict:
    """Helper function to process dictionary values for JSON compatibility."""
    result = {}
    for key, value in d.items():
        if isinstance(value, float):
            # Handle special float values
            if math.isnan(value):
                result[key] = "NaN"
            elif math.isinf(value):
                if value > 0:
                    result[key] = "Infinity"
                else:
                    result[key] = "-Infinity"
            else:
                result[key] = value
        elif isinstance(value, dict):
            # Recursively process nested dictionaries
            result[key] = _process_dict_for_json(value)
        elif isinstance(value, torch.Tensor):
            result[key] = serialize_tensor(value)
        else:
            result[key] = value
    return result

def deserialize_evolution_result(serialized_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deserialize an evolution result.
    
    Args:
        serialized_result: Serialized evolution result
        
    Returns:
        Evolution result dictionary with proper structure
    """
    if not isinstance(serialized_result, dict):
        raise ValueError("Serialized evolution result must be a dictionary")
    
    result = serialized_result.copy()
    
    # Handle evolved_function - deserialize it to genetic_code
    if 'evolved_function' in serialized_result:
        if isinstance(serialized_result['evolved_function'], str):
            # If it's a serialized string, deserialize to genetic code
            result['genetic_code'] = deserialize_genetic_code(serialized_result['evolved_function'])
            
    # Handle any tensors in metadata
    if 'metadata' in serialized_result and isinstance(serialized_result['metadata'], dict):
        for key, value in serialized_result['metadata'].items():
            if isinstance(value, dict) and 'type' in value and value['type'] == 'torch.Tensor':
                result['metadata'][key] = deserialize_tensor(value)
    
    return result

# Example usage
if __name__ == "__main__":
    # Example genetic code
    example_code = [
        [100, 1, 0],     # LOAD R1, S0 (y_true)
        [100, 2, 1],     # LOAD R2, S1 (y_pred)
        [2, 1, 2, 3],    # SUB R1, R2, R3 (y_true - y_pred)
        [10, 3, 4],      # SQUARE R3, R4
        [201, 4, -1, 5], # REDUCE_MEAN R4, axis=-1, R5
        [201, 5, 0, 9],  # REDUCE_MEAN R5, axis=0, R9
        [453, 9]         # RETURN R9
    ]
    
    # Serialize
    serialized = serialize_genetic_code(example_code)
    print(f"Serialized: {serialized}")
    
    # Deserialize
    deserialized = deserialize_genetic_code(serialized)
    print(f"Deserialized: {deserialized}")
    
    # Verify
    assert example_code == deserialized, "Serialization/deserialization failed!"
    print("Serialization/deserialization successful!")