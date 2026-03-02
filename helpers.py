import numpy as np

def linear_to_dB(x):
    """Convert linear value to decibels (dB)."""
    return 10 * np.log10(x)

def dB_to_linear(x):
    """Convert decibels (dB) to linear value."""
    return 10 ** (x / 10)