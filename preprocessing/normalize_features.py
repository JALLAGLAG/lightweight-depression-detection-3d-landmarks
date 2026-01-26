import numpy as np

def normalize(x, eps=1e-6):
    """Standard normalization per feature dimension."""
    mean = x.mean(axis=0)
    std = x.std(axis=0) + eps
    return (x - mean) / std
