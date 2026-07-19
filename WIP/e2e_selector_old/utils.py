import os
import numpy as np

def ensure_output_dir(root_path, nested_subdirs):
    """Triggers dynamic directory chain parsing."""
    full_path = os.path.join(root_path, nested_subdirs)
    os.makedirs(full_path, exist_ok=True)
    return full_path

def calculate_mags(df, flux_col, flux_err_col):
    """Applies flux-to-magnitude transformations safely on batch vectors."""
    # Floor non-physical data points to protect log10 arrays
    clean_flux = np.where(df[flux_col] <= 0, 1e-10, df[flux_col])
    
    mag = 22.5 - 2.5 * np.log10(clean_flux)
    mag_err = (2.5 / np.log(10.0)) * (df[flux_err_col] / clean_flux)
    return mag, mag_err

def evaluate_condition(series, inequality, value):
    if inequality == "<": return series < value
    if inequality == "<=": return series <= value
    if inequality == ">": return series > value
    if inequality == ">=": return series >= value
    if inequality == "==": return series == value
    return np.ones(len(series), dtype=bool)

