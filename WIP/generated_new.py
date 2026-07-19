import h5py
import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq

CATBASE = "/work/nlc38/e2e_2026/diffsky_cat"
FLEXZBOOST_FL = f"{CATBASE}/flexzboost_run1/zscatter_data.hdf5"
NOSHEAR_FL = f"{CATBASE}/e2e_catalog_noshear.parquet"

PHOTMTRY_TYPES_CONFIG = {"gold", "pgauss"}
SURVEY_CONFIG = {
    "roman": {"bands": "YJH", "case": str.upper, "suffix": ""},
    "lsst": {"bands": "ugrizy", "case": str.lower, "suffix": "_LSST"}
}

RNG = np.random.default_rng(2026)

# config validation
def validate_inputs(surveys, photmtry_types):
    """Ensures input choices match predefined valid configurations."""
    for survey in surveys:
        if survey not in SURVEY_CONFIG:
            raise KeyError(f"Invalid survey '{survey}'. Must be one of: {list(SURVEY_CONFIG.keys())}")
            
    for p_type in photmtry_types:
        if p_type not in PHOTMTRY_TYPES_CONFIG:
            raise KeyError(f"Invalid photometry type '{p_type}'. Must be 'gold' or 'pgauss'.")

def parse_input_list(value, default_options):
    """Helper to convert string parameters ('all' or single values) into clean lists."""
    if isinstance(value, str):
        if value.lower() == "all":
            return list(default_options)
        return [value.lower()]
    return [v.lower() for v in value]

# column resolution
def resolve_bands(survey_name, bands_input):
    """Extracts and formats valid bands for a specific survey iteration."""
    cfg = SURVEY_CONFIG[survey_name]
    if isinstance(bands_input, str) and bands_input.lower() == "all":
        return list(cfg["bands"])
    
    # Apply survey casing constraint (e.g. upper for Roman, lower for LSST)
    formatted_bands = [cfg["case"](b) for b in bands_input]
    return [b for b in formatted_bands if b in cfg["bands"]]

def build_target_columns(surveys, bands_input, photmtry_types, with_err):
    """Generates the absolute target schema list to look for in the Parquet file."""
    flux_cols = []
    err_cols = []
    
    for survey in surveys:
        bands = resolve_bands(survey, bands_input)
        suffix = SURVEY_CONFIG[survey]["suffix"]
        
        for p_type in photmtry_types:
            for band in bands:
                f_col = f"flux_{p_type}{suffix}_{band}"
                flux_cols.append(f_col)
                if with_err:
                    err_cols.append(f"flux_err_{p_type}{suffix}_{band}")
                    
    return flux_cols, err_cols

# data loading
def load_parquet_data(file_path, flux_cols, err_cols, max_rows=None):
    """Streams metadata and loads only the filtered, necessary columns into a DataFrame."""
    all_parquet_cols = pq.ParquetFile(file_path).metadata.schema.names
    
    # Isolate global non-flux structural properties
    base_cols = [c for c in all_parquet_cols if "flux" not in c.lower() and "mag" not in c.lower()]
    
    # Keep only target columns that actually exist in file schema
    valid_flux = [c for c in flux_cols if c in all_parquet_cols]
    assert len(valid_flux)==len(flux_cols), "Error in flux column name construction."
    valid_err = [c for c in err_cols if c in all_parquet_cols]
    assert len(valid_err)==len(err_cols), "Error in flux_err column name construction."
    columns_to_read = base_cols + valid_flux + valid_err
    
    if max_rows:
        dataset = ds.dataset(file_path, format="parquet")
        return dataset.head(max_rows, columns=columns_to_read).to_pandas(), valid_flux
        
    df = pq.read_table(file_path, columns=columns_to_read, use_threads=True).to_pandas()
    return df, valid_flux

######## correctly match the row indices being extracted from h5py file if I decide to subsample or downsample from the parquet file.
def append_hdf5_redshifts(df, file_path):
    """Safely reads matching rows from HDF5 file and appends them to the DataFrame."""
    num_rows = len(df)
    with h5py.File(file_path, "r") as h5_file:
        df['z_phot'] = h5_file["z_phot"][:num_rows]
        df['z_spec'] = h5_file["z_spec"][:num_rows]
    return df

def compute_mag_err(df, flux_col, flux_err_col):
    """Compute mag err"""
    return (2.5 / np.log(10.0)) * (df[flux_err_col] / df[flux_col])

# assign derived columns
def compute_magnitudes(df, flux_columns, only_mags, with_err):
    """Transforms raw flux arrays into AB magnitudes and magnitude errors."""
    for flux_col in flux_columns:
        df[flux_col] = df[flux_col].replace(0, 1e-10)

        # mag col
        mag_col = flux_col.replace("flux_", "mag_")
        mag_err_col = flux_col.replace("flux_", "mag_err_")
        flux_err_col = flux_col.replace("flux_", "flux_err_")

        df[mag_col] = 22.5 - 2.5 * np.log10(df[flux_col])
        
        # Calculate mag error: σ_mag = (2.5 / ln(10)) * (σ_flux / flux)
        if with_err and flux_err_col in df.columns:
            df[mag_err_col] = compute_mag_err(df, flux_col, flux_err_col)
            if only_mags:
                dropcols = [flux_col, flux_err_col]
        else:
            dropcols = [flux_col]

        df = df.drop(columns=dropcols, errors='raise')
            
    return df

# quality cuts and filters
def apply_quality_filters(df, max_mag_limit=30.0):
    """Removes NaN missing values and extreme physical magnitude outliers."""
    df = df.dropna()
    
    mag_cols = [c for c in df.columns if "mag_" in c and "err" not in c]
    if mag_cols:
        good_rows_mask = ~(df[mag_cols] > max_mag_limit).any(axis=1)
        df = df[good_rows_mask]
        
    return df

def apply_subsampling(df, sampling_factor=None, sample_size=None):
    """Handles downstream randomized slicing splits."""
    if not (sampling_factor or sample_size):
        return df
        
    if sampling_factor and not sample_size:
        n = len(df) // sampling_factor
    else:
        n = min(sample_size, len(df))
        
    return df.sample(n=n, random_state=RNG)

# ==============================
# ENTRY-POINT API INTERFACES
# ==============================
def get_full_data(
    surveys="roman", 
    bands="YJH", 
    photmtry_types="gold", 
    only_mags=True, 
    with_err=False, 
    with_flexzB_redshift=True, 
    max_rows=None
):
    """Main function handling core execution logic, column queries, and loading.

    Parameters:
    ----------
    surveys (str):
      all/roman/lsst
    bands (str):
      all/YJH/ugrizy
    photmtry_types (str):
      all/pgauss/gold
    only_mags (bool):
      default=True (removes flux columns)
    with_err (bool):
      default=False (removes flux_err columns)
    with_flexzB_redshift (bool):
      default=True (appends available specz and photo)
    max_rows (None|int):
      default=None (True, will return only first max_rows rows of the noshear file)
    """
    # Standardize and validate parameters
    active_surveys = parse_input_list(surveys, SURVEY_CONFIG.keys())
    active_p_types = parse_input_list(photmtry_types, PHOTMTRY_TYPES_CONFIG)
    validate_inputs(active_surveys, active_p_types)
    
    # Build col names and query files
    flux_cols, err_cols = build_target_columns(active_surveys, bands, active_p_types, with_err)
    print("flux_cols: ", flux_cols, "err cols: ", err_cols)
    df, valid_flux_cols = load_parquet_data(NOSHEAR_FL, flux_cols, err_cols, max_rows)
    print("valid_flux_cols: ", valid_flux_cols)
    
    # Add all derived quantities
    df = compute_magnitudes(df, valid_flux_cols, only_mags, with_err)
    
    # append additional info
    if with_flexzB_redshift:
        df = append_hdf5_redshifts(df, FLEXZBOOST_FL)
        
    return df

def get_clean_sample(
    surveys="all",
    bands="all",
    photmtry_types="gold",
    only_mags=True,
    with_err=False,
    with_redshift=True,
    sampling_factor=None,
    sample_size=None,
    use_cols=None
):
    """Downstream interface that fetches data, executes filter cuts, and handles random sampling."""
    # if both given, it makes sense to read a smaller data to begin with, but
    # this is not capturing the random rows, so can we do something to get
    # those rows randomly from the full data? Potentially, you could propagate
    # use of use_cols to first query the input data itself.
    fetch_rows = sample_size * sampling_factor if (sample_size and sampling_factor) else None
    
    # Fetch raw merged framework
    df = get_full_data(
        surveys=surveys, bands=bands, photmtry_types=photmtry_types,
        only_mags=only_mags, with_err=with_err, 
        with_flexzB_redshift=with_redshift, max_rows=fetch_rows
    )
    
    # drop by column cuts/filters
    df = apply_quality_filters(df, max_mag_limit=30.0)
    
    # Handle column slicing constraints
    ####### add yaml utility to define samples and infer which source columns
    ####### need to be read in the first place, like what you did with your simulation
    ####### sampling code. This will allow you to reduce the memmory load drastically
    ####### and make function much faster even if the input file being read is 1000s of
    ####### sq deg in just one file.
    if use_cols:
        existing_cols = [c for c in use_cols if c in df.columns]
        df = df[existing_cols]
        
    # Apply randomized train/test subset splits
    df = apply_subsampling(df, sampling_factor, sample_size)
    
    return df

if __name__ == "__main__":
    # a test example
    romandf = get_full_data(
            surveys="roman", bands="YJH", photmtry_types="gold", 
            only_mags=True, with_err=False, with_flexzB_redshift=True, max_rows=1000
    )

    lsstdf = get_full_data(
            surveys="lsst", bands="riz", photmtry_types="gold", 
            only_mags=True, with_err=False, with_flexzB_redshift=True, max_rows=1000
    )
