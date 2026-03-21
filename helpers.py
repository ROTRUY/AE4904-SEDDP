import numpy as np

def linear_to_dB(x):
    """Convert linear value to decibels (dB)."""
    return 10 * np.log10(x)

def dB_to_linear(x):
    """Convert decibels (dB) to linear value."""
    return 10 ** (x / 10)

def dBm_to_watt(x):
    """Convert dBm to watt"""
    return 10 ** (x / 10) / 1000

def spot_size(wavelength, lens_diameter, focal_length):
    return