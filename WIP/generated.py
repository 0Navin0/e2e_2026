import h5py
import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pprint import pprint

CATBASE = "/work/nlc38/e2e_2026/diffsky_cat"
FLEXZBOOST_FL = f"{CATBASE}/flexzboost_run1/zscatter_data.hdf5"
NOSHEAR_FL = f"{CATBASE}/e2e_catalog_noshear.parquet"

PHOTMTRY_TYPES = {"gold", "pgauss", "all"}
SURVEYS = {
    "roman": {
        "bands": "YJH", 
        "case": str.upper, 
        "suffix": ""
        },
    "lsst": {
        "bands": "UGRIZY", 
        "case": str.lower, 
        "suffix": "_LSST"
        }
}

RNG = np.random.default_rng(2026)

def check_photmtry(ptype):
    if ptype not in PHOTMTRY_TYPES:
        raise KeyError(f"Invalid photometry type '{ptype}'. Must be one of: {list(PHOTMTRY_TYPES)}")

def check_bands(bands, survey="roman"):
    if survey.lower() not in SURVEYS:
        raise KeyError(f"Invalid survey type '{survey}'. Must be one of: {list(SURVEYS.keys())}")
    
    valid_bands = SURVEYS[survey.lower()]["bands"]
    for band in bands:
        if band not in valid_bands:
            raise KeyError(f"Invalid band '{band}' for survey '{survey}'. Must be one of: {list(valid_bands)}.")

def inspect_e2e_noshear_parquet_file(filename=NOSHEAR_FL):
    """Describe the schema of the parquet formatted no-shear file from metadetect"""
    parquet_handle = pq.ParquetFile(filename)
    pprint(parquet_handle.metadata.schema.names)

def inspect_e2e_z_h5_file(filename=FLEXZBOOST_FL, preview_rows=5):
    """Recursively prints the structure of an HDF5 file and previews its datasets."""
    print(f"=== Inspecting Structure of: {filename} ===\n")
    
    with h5py.File(filename, 'r') as f:
        # Define a visitor function to print items dynamically
        def print_item(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}")
                print(f"  Shape: {obj.shape}")
                print(f"  Dtype: {obj.dtype}")
                
                # Dynamic preview: safely grab the first few rows based on actual dataset shape
                size = min(preview_rows, obj.shape[0])
                print(f"  Preview (first {size} elements): {obj[:size]}\n")
            else:
                print(f"Group: {name}\n")

        # Recursively visit all items in the file
        f.visititems(print_item)

def get_full_data(
        bands="YJH", get_roman=True, 
        only_mags=True, with_err=True, photmtry_type="gold", 
        with_flexzB_redshift=True, max_rows=None
    ):
    """
    Assimilates data from Parquet (properties/fluxes) and HDF5 (redshifts).
    Accepts row limits (max_rows) to avoid massive disk read loads during testing.

    LSST bands are available in ugrizy.
    Roman bands are available in YJH.
    Redshifts derived from flexZBoost and spec-z catalog are available.

    photometries are available in two flavors: gold and pgauss (from metadetect)

    For list of columns, see
    >>> inspect_e2e_noshear_parquet_file()
    >>> inspect_e2e_z_h5_file()
    """
    check_photmtry(photmtry_type)
    survey_name = "roman" if get_roman else "lsst"
    
    # Standardize casing per survey requirements
    bands = SURVEYS[survey_name]["case"](bands)
    check_bands(bands, survey_name)
    
    # Determine Label Infixes
    p_label = "" if photmtry_type == "all" else f"_{photmtry_type}"
    s_label = SURVEYS[survey_name]["suffix"]
    
    # Build List of Required Columns from Parquet File
    parquet_handle = pq.ParquetFile(NOSHEAR_FL)
    all_parquet_cols = parquet_handle.metadata.schema.names
    
    # Always include baseline galaxy structural properties (non-flux fields)
    requested_cols = [col for col in all_parquet_cols if "flux" not in col.lower()]
    
    # Map desired flux/error patterns dynamically
    target_flux_cols = []
    target_err_cols = []
    for b in bands:
        target_flux_cols.append(f"flux{p_label}{s_label}_{b}")
        if with_err:
            target_err_cols.append(f"flux_err{p_label}{s_label}_{b}")
 
    # Verify presence before streaming from parquet schema
    flux_to_read = [c for c in target_flux_cols if c in all_parquet_cols]
    assert len(flux_to_read)==len(target_err_cols), "Error in flux-column name parsing!"
    err_to_read = [c for c in target_err_cols if c in all_parquet_cols]
    assert len(err_to_read)==len(target_err_cols), "Error in flux_err-column name parsing!"
    requested_cols.extend(flux_to_read + err_to_read)
    
    # Read the Parquet table with memory chunking if needed
    if max_rows:
        dataset = ds.dataset(NOSHEAR_FL, format="parquet")
        df = dataset.head(max_rows, columns=requested_cols).to_pandas()
    else:
        df = pq.read_table(NOSHEAR_FL, columns=requested_cols, use_threads=True).to_pandas()
        
    # derived quantities now
    for flux_col in flux_to_read:
        band_identifier = flux_col.split("_")[-1]
        mag_col_name = f"mag{p_label}{s_label}_{band_identifier}"
        err_col_name = f"mag_err{p_label}{s_label}_{band_identifier}"
        flux_err_col = flux_col.replace("flux", "flux_err")
        
        # Clean zeros to avoid log10 infinity errors
        df[flux_col] = df[flux_col].replace(0, 1e-10)
        
        if only_mags: 
            df[mag_col_name] = 22.5 - 2.5 * np.log10(df[flux_col])
            
        if with_err and flux_err_col in df.columns:
            # Formula: σ_mag = (2.5 / ln(10)) * (σ_flux / flux)
            df[err_col_name] = (2.5 / np.log(10.0)) * (df[flux_err_col] / df[flux_col])
            
        # Drop raw fluxes downstream if user ONLY wanted absolute magnitudes
        if only_mags:
            df.drop(columns=[flux_col], errors='ignore')
            if flux_err_col in df.columns:
                df.drop(columns=[flux_err_col], errors='ignore')

    # Now get the Redshifts
    if with_flexzB_redshift:
        num_rows_needed = len(df)
        with h5py.File(FLEXZBOOST_FL, "r") as h5_file:
            df['z_phot'] = h5_file["z_phot"][:num_rows_needed]
            df['z_spec'] = h5_file["z_spec"][:num_rows_needed]
            
    return df

def get_clean_cardinalDeep_data(bands="YJH", get_roman=True, only_mags=True, photmtry_type="gold",
                                with_flexzB_redshift=True, sampling_factor=None, sample_size=None, use_cols=None):
    """
    Calls get_full_data, executes structural quality filters, cuts outliers, 
    and handles downstream subsampling.
    """
    # Use max_rows optimization directly if a specific sample size is wanted before loading everything
    fetch_rows = sample_size * sampling_factor if (sample_size and sampling_factor) else None
    
    df = get_full_data(
        bands=bands, 
        get_roman=get_roman, 
        only_mags=only_mags, 
        photmtry_type=photmtry_type, 
        with_flexzB_redshift=with_flexzB_redshift,
        max_rows=fetch_rows
    )
    
    # Drop rows containing any missing cells
    df = df.dropna()
    
    # Filter non-physical data artifacts (e.g. magnitudes exceeding extreme limit thresholds)
    obs_mag_lim = 30.0
    mag_cols = [c for c in df.columns if "mag" in c and "err" not in c]
    if mag_cols:
        mask = (df[mag_cols] > obs_mag_lim).any(axis=1)
        df = df[~mask]
        
    # Isolate strictly columns user explicitly tracked 
    if use_cols:
        existing_use_cols = [c for c in use_cols if c in df.columns]
        df = df[existing_use_cols]
        
    # Execute random subsampling splits
    if sampling_factor or sample_size:
        if sampling_factor and not sample_size:
            n = len(df) // sampling_factor
        else:
            n = min(sample_size, len(df))
            
        return df.sample(n=n, random_state=RNG)
        
    return df

