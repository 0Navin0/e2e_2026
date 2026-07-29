# config_parser.py
import yaml
from .schema_inspector import get_native_columns
from .derived_quantities import derived_registry

ROMAN_BANDS = list("YJH")
LSST_BANDS = list("ugrizy")
SURVEY_MAP = {
    "roman": {"bands": ROMAN_BANDS, "suffix": "", "case": str.upper},
    "lsst": {"bands": LSST_BANDS, "suffix": "_LSST", "case": str.lower}
}

ALL_PHOTOMETRY_TYPES = ["gold", "pgauss"]
ALL_SURVEYS = ["roman", "lsst"]


# =========================================================================
# 1. Config Loading Helpers
# =========================================================================

def load_global_config(global_cfg="global_paths.yaml"):
    """Loads top-level input paths and system configurations."""
    with open(global_cfg, 'r') as f:
        return yaml.safe_load(f)


def load_environment_and_profile(paths_yaml, config_yaml, profile_name):
    """Loads global environment settings and merges them into a selected profile."""
    env = load_global_config(paths_yaml)
    with open(config_yaml, 'r') as f:
        profiles = yaml.safe_load(f)

    if profile_name not in profiles:
        raise KeyError(f"Profile '{profile_name}' not found in {config_yaml}")

    profile = profiles[profile_name]
    profile["source_in"] = env["inputs"]["source_parquet"]
    profile["redshift_in"] = env["inputs"]["redshift_hdf5"]
    profile["root_output"] = env["system"]["root_output_dir"]
    profile["global_seed"] = env["system"]["default_seed"]
    return profile


def parse_list(block, fallbacks):
    """Parses config block options ('all' or 'which')."""
    if not block:
        return []
    if block.get("all"):
        return list(fallbacks)
    w = block.get("which", [])
    if isinstance(w, str):
        raise KeyError(f"You passed '{w}'. Expected a list!")
    return [x.lower() for x in w]


def build_column_name(prefix, p_type, survey, band):
    """Constructs standardized column names (e.g. flux_gold_LSST_i, mag_gold_LSST_i)."""
    s_key = survey.lower()
    suffix = SURVEY_MAP[s_key]["suffix"]
    fmt_b = SURVEY_MAP[s_key]["case"](band)
    return f"{prefix}_{p_type.lower()}{suffix}_{fmt_b}"


# =========================================================================
# 2. Modular Filter Handlers
# =========================================================================

def handle_range(cfg):
    """Processes a range filter config into lower and upper boundary rules."""
    low, high = cfg['bounds']
    l_op, h_op = cfg['inequality']
    res = [
        {"col": cfg['col'], "op": l_op, "val": low},
        {"col": cfg['col'], "op": h_op, "val": high}
    ]
    if "_original_derived_col" in cfg:
        for r in res:
            r["_original_derived_col"] = cfg["_original_derived_col"]
    return res


def handle_max_min(cfg):
    """Processes a min or max single-bound filter config."""
    f_type = cfg.get('type', '').lower()
    op = cfg.get('inequality')
    val = cfg.get('value', cfg.get('val'))
    if f_type in ['max', 'less_than']:
        if not op:
            op = "<="
        assert op in ["<", "<="], f"Error in column {cfg.get('col')}: 'max' filter requires < or <=, got '{op}'"
    elif f_type in ['min', 'greater_than']:
        if not op:
            op = ">="
        assert op in [">", ">="], f"Error in column {cfg.get('col')}: 'min' filter requires > or >=, got '{op}'"
    res = [{"col": cfg['col'], "op": op, "val": val}]
    if "_original_derived_col" in cfg:
        res[0]["_original_derived_col"] = cfg["_original_derived_col"]
    return res


def handle_bitmask(cfg):
    """Processes a bitmask filter config."""
    return [{"col": cfg['col'], "op": "&", "val": cfg.get('mask')}]


FILTER_HANDLERS = {
    "range": handle_range,
    "max": handle_max_min,
    "min": handle_max_min,
    "bitmask": handle_bitmask
}


def _flip_inequality(ineq_list):
    """Flips inequality operators when converting inverted magnitude bounds to flux limits."""
    flip_map = {">": "<", ">=": "<=", "<": ">", "<=": ">="}
    lower_op = flip_map.get(ineq_list[1], ">=")
    upper_op = flip_map.get(ineq_list[0], "<")
    return [lower_op, upper_op]


# =========================================================================
# 3. Main Resolution Engine
# =========================================================================

def resolve_columns_and_filters(config, native_schema):
    """
    Parses YAML configuration and inspects native schema to return:
      - parquet_cols: Native Parquet input columns to read.
      - hdf5_cols: Auxiliary HDF5 input columns to read.
      - output_cols: Final columns saved in output catalog (native + derived).
      - derived_cols: Derived quantities to compute via derived_registry.
      - filter_required_cols: Columns explicitly required for filtering.
      - resolved_filters: Structured list of operational active filters.
    """
    all_native = set(native_schema.get("all_native", []))
    all_nat_nonflux_cols = set(native_schema.get("all_nat_nonflux_cols", []))
    all_nat_flux_cols = set(native_schema.get("all_nat_flux_cols", []))
    all_nat_flux_err_cols = set(native_schema.get("all_nat_flux_err_cols", []))
    verbose = config.get("verbose", False)

    p_types = parse_list(config.get("photometry_type", config.get("photometry", {})), ALL_PHOTOMETRY_TYPES)
    surveys = parse_list(config.get("survey", {}), ALL_SURVEYS)

    output_cols = set()
    required_native_cols = set()
    derived_cols = set()
    filter_required_cols = set()
    resolved_filters = []

    # Helper to check if a native column matches selected surveys and photometry types
    def _matches_survey_and_ptype(col_name):
        c_lower = col_name.lower()
        has_lsst = "_lsst_" in c_lower or c_lower.endswith("_lsst")
        if "lsst" in surveys and "roman" not in surveys and not has_lsst:
            return False
        if "roman" in surveys and "lsst" not in surveys and has_lsst:
            return False
        if p_types and not any(f"_{pt.lower()}_" in c_lower or c_lower.startswith(f"flux_{pt.lower()}_") or c_lower.startswith(f"flux_err_{pt.lower()}_") or c_lower.startswith(f"mag_{pt.lower()}_") for pt in p_types):
            return False
        return True

    # ---------------------------------------------------------------------
    # Step A: Native Non-Flux Properties
    # ---------------------------------------------------------------------
    nonflux_cfg = config.get("native_prop", {}).get("nonflux_prop", {})
    if nonflux_cfg.get("all"):
        requested_nonflux = all_nat_nonflux_cols
    else:
        requested_nonflux = set(nonflux_cfg.get("which", [])).intersection(all_nat_nonflux_cols)

    output_cols.update(requested_nonflux)
    required_native_cols.update(requested_nonflux)

    # ---------------------------------------------------------------------
    # Step B: Native Flux and Flux Error Properties (Direct Schema Check)
    # ---------------------------------------------------------------------
    flux_cfg = config.get("native_prop", {}).get("flux_cols", {})
    flux_err_cfg = config.get("native_prop", {}).get("flux_err_cols", {})

    # Flux Columns
    if flux_cfg.get("all"):
        matched = {c for c in all_nat_flux_cols if _matches_survey_and_ptype(c)}
        output_cols.update(matched)
        required_native_cols.update(matched)
    elif flux_cfg.get("which"):
        which_list = [str(w).lower() for w in flux_cfg["which"]]
        for c in all_nat_flux_cols:
            c_lower = c.lower()
            if c_lower in which_list or any(c_lower.endswith(f"_{w}") for w in which_list):
                if _matches_survey_and_ptype(c):
                    output_cols.add(c)
                    required_native_cols.add(c)

    # Flux Error Columns
    if flux_err_cfg.get("all"):
        matched_err = {c for c in all_nat_flux_err_cols if _matches_survey_and_ptype(c)}
        output_cols.update(matched_err)
        required_native_cols.update(matched_err)
    elif flux_err_cfg.get("which"):
        which_err_list = [str(w).lower() for w in flux_err_cfg["which"]]
        for c in all_nat_flux_err_cols:
            c_lower = c.lower()
            if c_lower in which_err_list or any(c_lower.endswith(f"_{w}") for w in which_err_list):
                if _matches_survey_and_ptype(c):
                    output_cols.add(c)
                    required_native_cols.add(c)

    # ---------------------------------------------------------------------
    # Step C: Derived Properties (get_mags, mag_err_cols, other_cols)
    # ---------------------------------------------------------------------
    derived_block = config.get("derived_prop", {})
    mag_cfg = derived_block.get("get_mags", {})
    mag_err_cfg = derived_block.get("mag_err_cols", {})
    other_cfg = derived_block.get("other_cols", {})

    all_registered = derived_registry.list_all()

    # Mags
    if mag_cfg.get("all"):
        for name in all_registered:
            if name.startswith("mag_") and not name.startswith("mag_err_"):
                if _matches_survey_and_ptype(name):
                    derived_cols.add(name)
                    output_cols.add(name)
    elif mag_cfg.get("which"):
        which_mags = [str(w).lower() for w in mag_cfg["which"]]
        for name in all_registered:
            if name.startswith("mag_") and not name.startswith("mag_err_"):
                if name.lower() in which_mags or any(name.lower().endswith(f"_{w}") for w in which_mags):
                    if _matches_survey_and_ptype(name):
                        derived_cols.add(name)
                        output_cols.add(name)

    # Mag Errors
    if mag_err_cfg.get("all"):
        for name in all_registered:
            if name.startswith("mag_err_"):
                if _matches_survey_and_ptype(name):
                    derived_cols.add(name)
                    output_cols.add(name)
    elif mag_err_cfg.get("which"):
        which_errs = [str(w).lower() for w in mag_err_cfg["which"]]
        for name in all_registered:
            if name.startswith("mag_err_"):
                if name.lower() in which_errs or any(name.lower().endswith(f"_{w}") for w in which_errs):
                    if _matches_survey_and_ptype(name):
                        derived_cols.add(name)
                        output_cols.add(name)

    # Custom/Other Derived Columns
    if other_cfg.get("which"):
        for item in other_cfg["which"]:
            if item in all_registered:
                derived_cols.add(item)
                output_cols.add(item)

    # ---------------------------------------------------------------------
    # Step D: Filter Processing (Native + Derived Conversions)
    # ---------------------------------------------------------------------
    raw_filters = []
    if "native_prop" in config:
        raw_filters.extend(config["native_prop"].get("nonflux_prop", {}).get("filters", []) or [])
        raw_filters.extend(config["native_prop"].get("flux_cols", {}).get("filters", []) or [])
        raw_filters.extend(config["native_prop"].get("flux_err_cols", {}).get("filters", []) or [])
    if "derived_prop" in config:
        raw_filters.extend(config["derived_prop"].get("filters", []) or [])

    for filt in raw_filters:
        if not filt or not isinstance(filt, dict) or "col" not in filt:
            continue

        col_input = filt["col"]
        qty = filt.get("quantity", "raw")
        p_type = filt.get("photometry_type", p_types[0] if p_types else "gold")
        survey_val = filt.get("survey", surveys[0] if surveys else "roman")

        if qty == "mag" or (isinstance(col_input, str) and col_input.startswith("mag_")):
            # Construct target derived mag name and converted native flux name
            mag_col_name = build_column_name("mag", p_type, survey_val, col_input) if qty == "mag" else col_input
            flux_col_name = build_column_name("flux", p_type, survey_val, col_input) if qty == "mag" else col_input.replace("mag_", "flux_")

            filter_required_cols.add(mag_col_name)

            # Invert bounds if range filter
            if filt.get("type") == "range" and "bounds" in filt:
                inverted_bounds = derived_registry.invert(mag_col_name, filt["bounds"])
                f_min, f_max = sorted(inverted_bounds)
                flipped_ineq = _flip_inequality(filt.get("inequality", [">=", "<="]))

                converted_cfg = {
                    "col": flux_col_name,
                    "type": "range",
                    "bounds": [f_min, f_max],
                    "inequality": flipped_ineq,
                    "_original_derived_col": mag_col_name
                }
                filter_required_cols.add(flux_col_name)
                required_native_cols.add(flux_col_name)

                handler = FILTER_HANDLERS.get(converted_cfg["type"].lower())
                if handler:
                    resolved_filters.extend(handler(converted_cfg))
                if verbose:
                    print(f"[Filter Conversion] Magnitude cut on '{mag_col_name}' {filt['bounds']} "
                          f"converted to native flux cut on '{flux_col_name}' [{f_min:.4f}, {f_max:.4f}] nJy")
            else:
                derived_cols.add(mag_col_name)
                handler = FILTER_HANDLERS.get(filt.get("type", "range").lower())
                if handler:
                    resolved_filters.extend(handler(filt))
        else:
            # Native column filter
            target_cols = [col_input] if isinstance(col_input, str) else col_input
            for c in target_cols:
                filter_required_cols.add(c)
                required_native_cols.add(c)

            handler = FILTER_HANDLERS.get(filt.get("type", "range").lower())
            if handler:
                resolved_filters.extend(handler(filt))

    # ---------------------------------------------------------------------
    # Step E: Resolve Native Column Prerequisites for Derived Quantities
    # ---------------------------------------------------------------------
    if derived_cols:
        req_native_from_derived = derived_registry.get_required_native_cols(list(derived_cols))
        required_native_cols.update(req_native_from_derived)

    # ---------------------------------------------------------------------
    # Step F: Map Native Inputs into Parquet vs HDF5 Columns
    # ---------------------------------------------------------------------
    parquet_sources = set(native_schema.get("sources", {}).get("parquet", []))
    hdf5_sources = set(native_schema.get("sources", {}).get("hdf5", []))

    parquet_cols = sorted(list(required_native_cols.intersection(parquet_sources)))
    hdf5_cols = sorted(list(required_native_cols.intersection(hdf5_sources)))

    # Fallback for unmapped redshift or general columns
    for col in required_native_cols:
        if col not in parquet_cols and col not in hdf5_cols:
            if col in ["phot_z", "spec_z", "z_phot", "z_spec"]:
                hdf5_cols.append(col)
            elif col in all_native:
                parquet_cols.append(col)

    return {
        "parquet_cols": sorted(list(set(parquet_cols))),
        "hdf5_cols": sorted(list(set(hdf5_cols))),
        "output_cols": sorted(list(output_cols)),
        "derived_cols": sorted(list(derived_cols)),
        "filter_required_cols": sorted(list(filter_required_cols)),
        "resolved_filters": resolved_filters
    }
