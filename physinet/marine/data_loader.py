import numpy as np

def load_synthetic_wave_data(path):
    data = np.load(path)
    return data["eta"], data["condition"]

def load_metocean_data(path):
    """
    Real data loader placeholder.
    Later can load ERA5, NOAA, R/V Gunnerus datasets.
    """
    raise NotImplementedError("Real metocean data loading not implemented yet.")
