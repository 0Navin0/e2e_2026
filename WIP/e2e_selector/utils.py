import os
import numpy as np

def ensure_output_dir(root_path, nested_subdirs):
    if not nested_subdirs: 
        return root_path
    full_path = os.path.join(root_path, nested_subdirs)
    os.makedirs(full_path, exist_ok=True)
    return full_path

def calculate_mags_only(flux_vector):
    clean_flux = np.where(flux_vector <= 0, 1e-10, flux_vector)
    return 22.5 - 2.5 * np.log10(clean_flux)

def calculate_mag_errors(flux_vector, flux_err_vector):
    clean_flux = np.where(flux_vector <= 0, 1e-10, flux_vector)
    return (2.5 / np.log(10.0)) * (flux_err_vector / clean_flux)

def evaluate_condition(series, inequality, value):
    if inequality == "<": return series < value
    if inequality == "<=": return series <= value
    if inequality == ">": return series > value
    if inequality == ">=": return series >= value
    if inequality == "==": return series == value
    return np.ones(len(series), dtype=bool)

