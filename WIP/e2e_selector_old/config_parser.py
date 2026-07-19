import yaml

SURVEY_MAP = {
    "roman": {"bands": ["Y", "J", "H"], "suffix": "", "case": str.upper},
    "lsst": {"bands": ["u", "g", "r", "i", "z", "y"], "suffix": "_LSST", "case": str.lower}
}

def load_config(config_path, profile_name="lsst_gold_y1"):
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Merge named profile with defaults
    profile = config_data.get(profile_name, {})
    default = config_data.get("default", {})
    
    merged = {**default, **profile}
    for key in ["downSample", "main_cols", "flux_cols", "get_mags"]:
        if key in default and key in profile:
            merged[key] = {**default[key], **profile[key]}
            
    return merged

def resolve_required_columns(config, all_schema_names):
    """
    Scans the requested variables and filter configurations to identify 
    all source input columns that must be read from the Parquet disk.
    """
    input_cols = set(config.get("main_cols", {}).get("which", []))
    final_keep_cols = set(input_cols)
    
    # 1. Resolve Flux Column naming patterns
    flux_cfg = config.get("flux_cols", {})
    mag_cfg = config.get("get_mags", {})
    
    # Automatically add fluxes if specific mags are requested (since mags are derived from flux)
    for survey, s_info in SURVEY_MAP.items():
        suffix = s_info["suffix"]
        for band in s_info["bands"]:
            f_col = f"flux_gold{suffix}_{band}"
            fe_col = f"flux_err_gold{suffix}_{band}"
            m_col = f"mag_gold{suffix}_{band}"
            me_col = f"mag_err_gold{suffix}_{band}"
            
            # If all flux columns or specific bands are targeted
            if flux_cfg.get("all") or band in flux_cfg.get("which", []):
                input_cols.update([f_col, fe_col])
                final_keep_cols.update([f_col, fe_col])
                
            # If all mags or specific bands are targeted
            if mag_cfg.get("all") or band in mag_cfg.get("which", []):
                input_cols.update([f_col, fe_col]) # needed for computation
                final_keep_cols.update([m_col, me_col])

    # 2. Extract Columns utilized across filter parameters
    for filt in config.get("filters", []):
        cols = filt["col"] if isinstance(filt["col"], list) else [filt["col"]]
        qty = filt.get("quantity", "raw")
        srv = filt.get("survey", "roman")
        
        for col in cols:
            if qty == "mag":
                suffix = SURVEY_MAP[srv]["suffix"]
                band = SURVEY_MAP[srv]["case"](col)
                input_cols.update([f"flux_gold{suffix}_{band}", f"flux_err_gold{suffix}_{band}"])
            else:
                input_cols.add(col)
                
    # Filter out columns that are built natively via external HDF5 steps
    input_cols = {c for c in input_cols if c in all_schema_names}
    
    return list(input_cols), list(final_keep_cols)

