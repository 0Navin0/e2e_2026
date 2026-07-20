import sys
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pathlib import Path

catalog_base =  Path("/work/nlc38/output_base")

def magLim_parquet_filter(mag_col="mag_gold_LSST_i", z_col="phot_z", **kwargs):
    mag = ds.field(mag_col)
    photoz = ds.field(z_col)
    return (mag < ((photoz * 4) + 18)) & (mag > 17.5)

def only_redshift_parquet_filter(z_col, zmin, zmax, **kwargs):
    photoz = ds.field(z_col)
    return (photoz >= zmin) & (photoz < zmax)

def full_redshift_bin_magLim_pyArrTable(
        dataset, 
        cols=None, 
        z_col="phot_z",
        zmin=0.2,
        zmax=1.05,
        **kwargs
    ):
    filt = only_redshift_parquet_filter(z_col, zmin, zmax, **kwargs)
    return dataset.to_table(columns=cols, filter=filt)

def magLim_sample_pyArrTable(
        dataset, 
        cols=None, 
        mag_col="mag_gold_LSST_i",
        z_col="phot_z",
        **kwargs
    ):
    """return a pyarrow.dataset containing magLim selection.
    cols:
      columns required in the output, order is preserved, can be a dict pointing old->new_col_name
    mag_col: 
      The magnitude column in the dataset on which magLim magnitude cut is imposed.
    z_col:
      The redshift column used to import magLim redshift selection.
    """
    filt = magLim_parquet_filter(mag_col, z_col, **kwargs)
    return dataset.to_table(columns=cols, filter=filt)

sample_selector_pyArrTable = {
        "only_redshift": full_redshift_bin_magLim_pyArrTable,
        "magLim": magLim_sample_pyArrTable
        }

def magLim_sample_df(pyArrTable):
    """Return a pandas dataframe from the pyarrwow table object"""
    return pyArrTable.to_pandas()

def load_default_dataset():
    """Return the E2E dataset.
    This contains additional columns of magnitudes, magnitude errors, phot_z
    and spec_z added by my code.

    Note: this catalog, has a 
        cut on gold_LSST_i in (17,24]
        and z_phot [0.0, 1.4)
    """
    filedir = catalog_base / "package_validation/survey_all/photmtry_all/maglim_fluxlim_superset"
    flpath = filedir / "maglim_fluxlim_superset_17Jul2026.parquet"
    return ds.dataset(flpath, format="parquet")

def see_all_columns():
    dat = load_default_dataset()
    all_cols = dat.schema.names
    print("All available columns in my parquet file:\n", all_cols)

def ordered_cols(dat, mag_or_flux="mag"):
    nonfluxcol = [col for col in dat.schema.names if (mag_or_flux not in col) & ("mag" not in col) & ("objectid" not in col)]
    goldfluxcol = [col for col in dat.schema.names if (mag_or_flux in col) & ("err" not in col) & ("gold" in col) ]
    golderrcol = [col for col in dat.schema.names if (mag_or_flux in col) & ("err" in col) & ("gold" in col) ]
    pgausserrcol = [col for col in  dat.schema.names if (mag_or_flux in col) & ("err" in col) & ("pgauss" in col) ]
    pgaussfluxcol = [col for col in dat.schema.names if (mag_or_flux in col) & ("err" not in col) & ("pgauss" in col) ]
    return ["objectid"] + nonfluxcol + goldfluxcol + golderrcol + pgaussfluxcol + pgausserrcol

def cols_for_sompz():
    """Return the columns needed to run SOMPZ pipeline."""
    dat = load_default_dataset()
    return ordered_cols(dat, mag_or_flux="flux")

def cols_for_sample_selection():
    dat = load_default_dataset()
    return ordered_cols(dat, "mag")

col_selector = {
        "mag": cols_for_sample_selection(),
        "flux": cols_for_sompz()
}

def write_magLim_fluxLim_supersample(filename, selector_input="magLim", mag_or_flux="mag", fmt=".parquet"):
    """Write parquet/dataframe objects for the MagLim/FluxLim supersample.
    mag_or_flux: flux or mag
      The flux columns should be in flux or magnitude values. For SOMPZ pipelie use "flux".
    fmt: parquet or pandas
      Either save file in Parquet or Pandas DataFame format.
    """
    fpath = filename.with_suffix(fmt)
    dat = load_default_dataset()

    selector_func = sample_selector_pyArrTable[selector_input]
    magLim_dataset = selector_func(dat, cols=col_selector[mag_or_flux], mag_col="mag_gold_LSST_i", z_col="phot_z")
    print("Number of objects in this catalog, ", magLim_dataset.num_rows)

    if fmt==".parquet":
        existing_metadata = magLim_dataset.schema.metadata or {}
        if b"pandas" in existing_metadata.keys():
            _ = existing_metadata.pop(b"pandas")
        custom_metadata = {"filter_applied": "str(magLim_parquet_filter())", "author": "Navin Chaurasiya"}
        custom_metadata_bytes = {k.encode(): v.encode() for k, v in custom_metadata.items()}
        merged_metadata = {**existing_metadata, **custom_metadata_bytes}
        table_with_metadata = magLim_dataset.replace_schema_metadata(merged_metadata)
        pq.write_table(table_with_metadata, fpath)
        print(f"saved {fpath}")
    elif fmt==".pandas":
        magLim_sample_df(magLim_dataset).to_csv(fpath, ignore_index=True, na_rep="NaN")
        print(f"saved {fpath}")
    else:
        sys.exit("Wrong file suffix. Nothing saved!")

def get_e2e_diffsky_MagLim_FluxLim_superset_data(fname=None, cols="defaut", fmt="parquet", filt=None):
    """Read in the supersample stored on disc.
    cols (list/str=default):
      Columns to read from the file
    fmt (parquet/pandas):
      parquet => return a dataset/scanner object, this preserves all the
      metadata in the file (doesn't load data)
      pandas => return a pandas dataFrame object, metadata ignored (loads data)
    filt:
      apply additional filters to dataset.scanner object on the file
    """
    if fname is None:
        # this has magnitudes
        fname = "magLim_fluxLim_supersample_samp_sel.parquet"
        fmt="pandas"
        cols="default"

    filename = catalog_base / fname
    dat = ds.dataset(filename, format="parquet")

    if cols=="default":
        all_cols = dat.schema.names
        cols = ["objectid", "ra", "dec", "z", "snr"] + [col for col in all_cols if ("mag" in col)]

    if fmt=="parquet":
        # apply any filter you want before even reading the data
        # use it to store data with metadata
        return dat.scanner(columns=cols, filter=filt)
    elif fmt=="pandas":
        return pd.read_parquet(filename, columns=cols)

if __name__=="__main__":

    dat = load_default_dataset()
    all_cols = dat.schema.names
    print("All available columns in my parquet file: ", all_cols)

    ## used to store the data in parquet format for Boyan's SOMPZ and my sample selection work: 17 July 2026
    #write_magLim_fluxLim_supersample(
    #        catalog_base / "magLim_fluxLim_supersample_sompz", 
    #        selector_input="magLim", 
    #        mag_or_flux = "flux", 
    #        fmt = ".parquet"
    #)
    #write_magLim_fluxLim_supersample(
    #        catalog_base / "magLim_fluxLim_supersample_samp_sel", 
    #        selector_input="magLim", 
    #        mag_or_flux = "mag", 
    #        fmt = ".parquet"
    #)
    # Note that this is not the comparison I actually intended. I wanted all
    # the objects to some realistic deep tier depth but this catalog only has
    # objects upto 24th mag, in lsst_gold (which currently is a true flux according to Chun-Hao)
    write_magLim_fluxLim_supersample(
            catalog_base / "magLim_fluxLim_supersample_z_0.0_1.05", 
            selector_input="only_redshift", 
            mag_or_flux="mag", 
            fmt=".parquet"
    )
