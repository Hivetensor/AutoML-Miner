# automl_client/utils/fec.py
import torch
import hashlib
from typing import Tuple, Dict

# Cache for fixed test inputs and for hash→score
_FEC_INPUTS: Dict[Tuple[str, Tuple[int,...]], torch.Tensor] = {}
_FEC_CACHE: Dict[str, float] = {}

def get_fixed_tensor(key: str, shape: Tuple[int, ...], dtype=torch.float32) -> torch.Tensor:
    """
    Returns a fixed tensor of the given shape, seeded deterministically
    by (key, shape).  Subsequent calls with the same key/shape return
    the same tensor.
    """
    cache_key = (key, shape)
    if cache_key not in _FEC_INPUTS:
        # derive a seed from the key
        seed = abs(hash(key)) % (2**32)
        g = torch.Generator().manual_seed(seed)
        _FEC_INPUTS[cache_key] = torch.randn(*shape, generator=g, dtype=dtype)
    return _FEC_INPUTS[cache_key]

def fec_lookup_or_evaluate(hash_bytes: bytes, evaluator: callable) -> float:
    """
    Given raw bytes and a function `evaluator()` that computes and returns
    the true fitness, return cached fitness if available, else call
    evaluator(), cache it under hash, and return it.
    """
    h = hashlib.sha256(hash_bytes).hexdigest()
    if h in _FEC_CACHE:
        return _FEC_CACHE[h]
    score = evaluator()
    _FEC_CACHE[h] = score
    return score
