from pyarrow import parquet
import pyarrow.parquet as pq
import numpy as np
import h5py
from pprint import pprint
rng = np.random.default_rng(2026)

catbase = "/work/nlc38/e2e_2026/diffsky_cat"
flexzboost_fl = f"{catbase}/flexzboost_run1/zscatter_data.hdf5"
noshear_fl = f"{catbase}/e2e_catalog_noshear.parquet"

photmtry_types = {"gold", "pgauss", "all"}
surveys = {
        "roman": dict(bands="YJH"),
        "lsst": dict(bands="UGRIZY")
}
noshear_pq_handle = parquet.ParquetFile(noshear_fl)

def check_photmtry(ptype):
    if ptype not in photmtry_types:
        raise KeyError(
            f"Invalid photometry type '{photmtry_type}'. "
            f"Must be one of: {list(photmtry_types)}"
        )

def check_bands(bands, survey="roman"):
    if survey.lower() not in surveys:
        raise KeyError(
                f"Invalid survey type '{survey}'. "
                f"Must be one of: {list(surveys.keys())}"
        )

    for band in bands:
        if band.lower() not in surveys[survey]["bands"].lower():
            raise KeyError(
                f"Invalid band type '{band}' for survey '{survey}'. "
                f"Must be one of: {list(surveys[survey]['bands'])}."
            )

def inspect_e2e_noshear_parquet_file():
    """Describe the schema of the parquet formatted no-shear file from metadetect"""
    pprint(noshear_pq_handle.metadata.schema.names)

def inspect_e2e_z_h5_file(filename=flexzboost_fl, preview_rows=5):
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

def get_full_data(bands="YJH", only_roman=True, only_mags=True, photmtry_type = "gold", with_flexzB_redshift=True):
    """
    LSST bands are available in ugrizy.
    Roman bands are available in YJH.
    Redshifts derived from flexZBoost and spec-z catalog are available.

    photometries are available in two flavors: gold and pgauss (from metadetect)

    For list of columns, see
    >>> inspect_e2e_noshear_parquet_file()
    >>> inspect_e2e_z_h5_file()
    """

    check_photmtry_lib(photmtry_type)

    if with_flexzB_redshift:
        with h5py.File(flexzboost_fl, "r") as f:
            zphot = f["z_phot"]
            zspec = f["z_spec"]

    def get_flux_cols(bands, only_roman=True, only_mags=True, photmtry_type="gold", with_err=False):
        plabel = "all"
        if photmtry_type=="gold":
            plabel = "_gold"
        elif photmtry_type=="pgauss":
            plabel = "_pgauss"

        if only_roman:
            survey = ""
            bands = bands.upper()
            check_bands(bands, "roman")
        else:
            survey = "_LSST"
            bands = bands.lower()
            check_bands(bands, "lsst")

        # filter what extra cols you want
        nonflux_cols = [col for col in noshear_pq_handle.metadata.schema.names if "flux" not in col]

        # desired flux cols: flux_gold_LSST_z, flux_err_gold_LSST_z, flux_gold_Y, flux_err_gold_Y
        fluxcols = [f"flux{plabel}{survey}_{band}" for band in bands] 
        if only_mags:
            magcols = [f"mag{plabel}{survey}_{band}" for band in bands] 
        if with_err:
            fluxerrcols = [f"flux_err{plabel}{survey}_{band}" for band in bands]
            magerrcols = [f"flux_err{plabel}{survey}_{band}" for band in bands]

        inpcols = fluxcols if not with_err else fluxcols+fluxerrcols
        rescols = magcols if not with_err else magcols+magerrcols

        # now read data
        fluxtable = pq.read_table(noshear_fl, columns=nonflux_cols+photcols, use_threads=True)
        proptable = pq.read_table(noshear_fl, columns=nonflux_cols, use_threads=True)

        if only_mags:
            for incol,rescol in zip(inpcols, rescols):
                #handle zeros before log10
                flux[flux==0] = 1E-10
                datadict[f"mag{plabel}{survey}_{band}"] = 22.5 - 2.5 * np.log10(flux)
                ### implement flux error to mag err later
                ### datadict[f"MAG{colname_filler}_ERR_{band}"] = 2.5 /np.log(10.) *fluxerr /flux

    outdict = get_flux_cols(bands, only_obs=only_obs, only_mags=only_mags)
    #finally add extra non-flux columns
    outdict['z'] = z #true redshift
    return outdict
