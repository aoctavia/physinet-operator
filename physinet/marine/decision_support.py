import numpy as np

def compute_operability_score(wave_field, threshold=0.5):
    """
    Simple operability metric:
    percentage of grid points where |η| < threshold.
    """
    return np.mean(np.abs(wave_field) < threshold)
