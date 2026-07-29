import os
import numpy as np
from .config_parser import SURVEY_MAP

def ensure_output_dir(root_path, nested_subdirs):
    if not nested_subdirs: 
        return root_path
    full_path = os.path.join(root_path, nested_subdirs)
    os.makedirs(full_path, exist_ok=True)
    return full_path

def calculate_mags(flux_vector):
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

def generate_sampling_mask(total_objects, downsample_cfg, seed):
    """Create an array of indices that will act as a mask on the original array for downsampling.
    Parameters:
    ----------
      total_objects (int): number of objects in the parent array
      downsample_cfg (dict): config that tells how to downsample
      seed (int): a seed number for numpy random sampler

    Returns:
    -------
      An array of indices into the parent array
    """
    if not downsample_cfg: 
        return None
    rng = np.random.default_rng(seed)
    fraction, factor = downsample_cfg.get("fraction"), downsample_cfg.get("factor")
    if fraction is not None: 
        target_size = int(total_objects * fraction)
    elif factor is not None: 
        target_size = int(total_objects // factor)
    else: 
        return None
    
    sampled_indices = rng.choice(total_objects, size=target_size, replace=False)
    sampled_indices.sort()
    return sampled_indices

def compute_dynamic_batch_size(num_columns, max_cells=40_000_000):
    """Enforces dynamic row limit thresholds depending on active parameter counts [6]."""
    return max(10000, max_cells // num_columns)

def apply_filters(df, filters):
    master_mask = np.ones(len(df), dtype=bool)
    for filt in filters:
        cols = filt["col"] if isinstance(filt["col"], list) else [filt["col"]]
        qty = filt.get("quantity", "raw")
        
        for col in cols:
            if qty == "mag" or qty == "flux":
                p = filt.get("photometry_type", "gold")
                s = filt.get("survey", "roman").lower()
                suffix = SURVEY_MAP[s]["suffix"]
                fmt_b = SURVEY_MAP[s]["case"](col)
                target_key = f"{qty}_{p}{suffix}_{fmt_b}"
            else:
                target_key = col
                
            if target_key not in df.columns: 
                continue
                
            if filt["type"] == "range":
                bounds, ineqs = filt["bounds"], filt["inequality"]
                # Evaluate range limits simultaneously
                master_mask &= evaluate_condition(df[target_key], ineqs[0], bounds[0])
                master_mask &= evaluate_condition(df[target_key], ineqs[1], bounds[1])
            else:
                master_mask &= evaluate_condition(df[target_key], filt["inequality"], filt["value"])
    return df[master_mask]

