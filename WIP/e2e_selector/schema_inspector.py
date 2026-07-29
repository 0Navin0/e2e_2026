from pathlib import Path
from typing import Dict, Any, List
import h5py
import pyarrow.parquet as pq

# Module-level cache: (resolved_path_str, mtime) -> List[str]
_FILE_SCHEMA_CACHE: Dict[tuple, List[str]] = {}


def clear_schema_cache():
    """Clears the internal file schema cache manually."""
    _FILE_SCHEMA_CACHE.clear()


def _read_columns_cached(file_path: Path, source_type: str) -> List[str]:
    """Reads and caches column names using (path, mtime) as the cache key. If
    file is modified on disk, read and cache the schema again."""
    resolved_path = file_path.resolve()
    mtime = resolved_path.stat().st_mtime
    cache_key = (str(resolved_path), mtime)

    # Return cached column names if file was previously read and unmodified
    if cache_key in _FILE_SCHEMA_CACHE:
        return _FILE_SCHEMA_CACHE[cache_key]

    cols = []
    if source_type == "parquet":
        parquet_file = pq.ParquetFile(resolved_path)
        cols = parquet_file.schema.names
    elif source_type == "hdf5":
        with h5py.File(resolved_path, "r") as h5f:
            def _extract_datasets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    cols.append(name)
            h5f.visititems(_extract_datasets)

    cols = sorted(cols)
    _FILE_SCHEMA_CACHE[cache_key] = cols
    return cols


def get_native_columns(
        parquet_path=None, 
        hdf5_path=None, 
        clear_cache: bool = False,
        **additional_sources
    ) -> Dict[str, Any]:
    """
    Inspects native columns across heterogeneous data sources without re-reading file 
    headers on subsequent calls. Categorizes columns into flux, flux_err, and non-flux groups.

    Parameters
    ----------
    parquet_path : str or Path, optional
        Path to main catalog Parquet file.
    hdf5_path : str or Path, optional
        Path to auxiliary redshift/photo-z HDF5 file.
    clear_cache : bool, default False
        If True, clears the schema cache before inspecting source files.
    additional_sources : dict
        Hook for future sources (e.g. fits_path="...").

    Returns
    -------
    dict
        Dictionary containing:
        - 'sources': Mapping of source types to column names
        - 'cached_files': List of resolved file paths currently cached/inspected
        - 'all_native': All discovered native columns
        - 'all_nat_flux_cols': Columns starting with 'flux_' (excluding flux_err_)
        - 'all_nat_flux_err_cols': Columns starting with 'flux_err_'
        - 'all_nat_nonflux_cols': Columns with no 'flux' string in their name
    """
    if clear_cache:
        clear_schema_cache()

    native_catalog = {
        "sources": {},
        "cached_files": [],
        "all_native": [],
        "all_nat_flux_cols": [],
        "all_nat_flux_err_cols": [],
        "all_nat_nonflux_cols": [],
    }

    all_native_set = set()

    # Main Parquet catalog inspection
    if parquet_path:
        p_path = Path(parquet_path)
        if p_path.exists():
            pq_cols = _read_columns_cached(p_path, source_type="parquet")
            native_catalog["sources"]["parquet"] = pq_cols
            native_catalog["cached_files"].append(str(p_path.resolve()))
            all_native_set.update(pq_cols)

    # Auxiliary HDF5 redshift file inspection
    if hdf5_path:
        h_path = Path(hdf5_path)
        if h_path.exists():
            h_cols = _read_columns_cached(h_path, source_type="hdf5")
            native_catalog["sources"]["hdf5"] = h_cols
            native_catalog["cached_files"].append(str(h_path.resolve()))
            all_native_set.update(h_cols)

    # Additional dynamic sources hook
    for source_name, source_path in additional_sources.items():
        if source_path:
            s_path = Path(source_path)
            if s_path.exists():
                s_cols = _read_columns_cached(s_path, source_type="parquet")
                native_catalog["sources"][source_name] = s_cols
                native_catalog["cached_files"].append(str(s_path.resolve()))
                all_native_set.update(s_cols)

    # Sort and categorize native columns
    all_native_sorted = sorted(list(all_native_set))
    native_catalog["all_native"] = all_native_sorted

    for col in all_native_sorted:
        if col.startswith("flux_err_"):
            native_catalog["all_nat_flux_err_cols"].append(col)
        elif col.startswith("flux_"):
            native_catalog["all_nat_flux_cols"].append(col)
        elif "flux" not in col:
            native_catalog["all_nat_nonflux_cols"].append(col)

    return native_catalog
