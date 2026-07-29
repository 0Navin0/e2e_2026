import yaml

ROMAN_BANDS = list("YJH")
LSST_BANDS = list("ugrizy")
SURVEY_MAP = {
    "roman": {"bands": ROMAN_BANDS, "suffix": "", "case": str.upper},
    "lsst": {"bands": LSST_BANDS, "suffix": "_LSST", "case": str.lower}
}

ALL_PHOTOMETRY_TYPES = ["gold", "pgauss"]
ALL_SURVEYS = ["roman", "lsst"]

def load_global_config(global_cf="global_paths.yaml"):
    with open(global_cfg, 'r') as f:
        return yaml.safe_load(f)

def load_environment_and_profile(paths_yaml, config_yaml, profile_name):
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
    if not block: return []
    if block.get("all"): return list(fallbacks)
    w = block.get("which", [])
    if isinstance(w, str): 
        raise KeyError(f"You passed {w}. Expected a list!")
    return [x.lower() for x in w]

def build_flux_names(p_type, survey, band):
    suffix = SURVEY_MAP[survey.lower()]["suffix"]
    fmt_b = SURVEY_MAP[survey.lower()]["case"](band)
    return f"flux_{p_type.lower()}{suffix}_{fmt_b}", f"flux_err_{p_type.lower()}{suffix}_{fmt_b}"

def resolve_columns_and_actions(config, native_schema):
    operational_cols = {
        "parquet_cols": [],
        "hdf5_cols": []
    }
    p_types = parse_list(config.get("photometry_type", config.get("photometry", {})), ALL_PHOTOMETRY_TYPES)
    surveys = parse_list(config.get("survey", {}), ALL_SURVEYS)
    
    # Extract native structural keys
    nonflux_cols = config.get("native_prop", {}).get("nonflux_prop", {})
    # here make use of native_schema from schema_inspector.py
    nonflux_cols = set(nonflux_cols.get("which", [])) if not nonflux_cols.get("all") else native_schema["all_nat_nonflux_cols"]
    
    # store the output columns needed from parquet and hdf5 files separately and put them in operational_cols dict
    input_cols = {c for c in nonflux_cols if c in native_schema and c not in ["phot_z", "spec_z"]}
    final_keep_cols = set(nonflux_cols)
    
    # I don't need actions now, because any derived quantity can now be converted in to a cut/filter on native quantity
    # So modify this code, to return a unified list of filters for parquet file.
    # Also return another variable that has all the native and derived quantities that need to be stored in the output file
    actions = {"calc_mags": False, "calc_mag_errs": False, "keep_flux": False, "keep_flux_err": False}
    flux_dependencies = set()
    all_filters = []

    verbose = config.get("verbose", False)

    # 1. Native Property Filters
    if "native_prop" in config and "filters" in config["native_prop"]:
        all_filters.extend(config["native_prop"]["filters"] or [])

    # 2. Native Flux Property Context Blocks
    flux_block = config.get("flux_cols", {})
    if flux_block.get("all") or flux_block.get("which") or flux_block.get("filters"):
        actions["keep_flux"] = True
        if flux_block.get("flux_err_cols", False): 
            actions["keep_flux_err"] = True
        if flux_block.get("filters"):
            all_filters.extend(flux_block["filters"])
        
        # FIX: Loop surveys independently and dynamically compute the bands specific to that survey configuration
        for s in surveys:
            bands_to_loop = SURVEY_MAP[s]["bands"] if flux_block.get("all") else list(flux_block.get("which", []))
            
            if verbose:
                print(f"[Flux Resolution] Survey: {s} | Photometries: {p_types} | Bands to loop: {bands_to_loop}")
                
            for p in p_types:
                for b in bands_to_loop:
                    # Verify if token mapped correctly to this specific survey's case strategy
                    fmt_b = SURVEY_MAP[s]["case"](b)
                    if fmt_b in SURVEY_MAP[s]["bands"]:
                        f_col, fe_col = build_flux_names(p, s, fmt_b)
                        flux_dependencies.add(f_col)
                        final_keep_cols.add(f_col)
                        if actions["keep_flux_err"]:
                            flux_dependencies.add(fe_col)
                            final_keep_cols.add(fe_col)

    # 3. Derived Property Magnitude Context Blocks (Modular & Encapsulated)
    derived_block = config.get("derived_prop", {})
    mag_cfg = derived_block.get("get_mags", {})
    if mag_cfg.get("all") or mag_cfg.get("which") or mag_cfg.get("filters"):
        actions["calc_mags"] = True
        if mag_cfg.get("mag_err_cols", False): 
            actions["calc_mag_errs"] = True
        if mag_cfg.get("filters"):
            all_filters.extend(mag_cfg["filters"])
        
        # FIX: Loop surveys independently and resolve bands correctly per survey
        for s in surveys:
            bands_to_loop = SURVEY_MAP[s]["bands"] if mag_cfg.get("all") else list(mag_cfg.get("which", []))
            
            if verbose:
                print(f"[Mag Resolution] Survey: {s} | Photometries: {p_types} | Bands to loop: {bands_to_loop}")
                
            for p in p_types:
                for b in bands_to_loop:
                    fmt_b = SURVEY_MAP[s]["case"](b)
                    if fmt_b in SURVEY_MAP[s]["bands"]:
                        f_col, fe_col = build_flux_names(p, s, fmt_b)
                        flux_dependencies.add(f_col)
                        
                        suffix = SURVEY_MAP[s]["suffix"]
                        final_keep_cols.add(f"mag_{p}{suffix}_{fmt_b}")
                        if actions["calc_mag_errs"]:
                            flux_dependencies.add(fe_col)
                            final_keep_cols.add(f"mag_err_{p}{suffix}_{fmt_b}")

    # Process overall filter requirements to guarantee target fluxes are cached
    for ii, filt in enumerate(all_filters):
        if verbose:
            print(f"config_parser: filter {ii}, {filt}")
            
        cols = filt["col"] if isinstance(filt["col"], list) else [filt["col"]]
        qty = filt.get("quantity", "raw")
        
        for col in cols:
            if qty in ["mag", "flux", "flux_err", "mag_err"]:
                # Error Guard: Enforce photometry_type string visibility
                if "photometry_type" not in filt:
                    raise KeyError(f"Filter specification error on '{col}': 'photometry_type' is missing for a flux/magnitude filter condition.")
                p = filt["photometry_type"]
                if p not in ALL_PHOTOMETRY_TYPES:
                    raise ValueError(f"Invalid photometry type '{p}' inside filter. Must be one of: {ALL_PHOTOMETRY_TYPES}")
                
                # Error Guard: Enforce survey configuration validation
                if "survey" not in filt:
                    raise KeyError(f"Filter specification error on '{col}': 'survey' key name must be defined for a flux/magnitude condition.")
                s = filt["survey"].lower()
                if s not in ALL_SURVEYS:
                    raise ValueError(f"Invalid survey type '{s}' inside filter. Must be one of: {ALL_SURVEYS}")
                
                f_col, fe_col = build_flux_names(p, s, col)
                flux_dependencies.add(f_col)
                
                if qty == "mag": 
                    actions["calc_mags"] = True
                if qty in ["flux_err", "mag_err"] or actions["calc_mag_errs"]:
                    flux_dependencies.add(fe_col)
                if qty == "mag_err":
                    actions["calc_mag_errs"] = True
            else:
                if col not in ["phot_z", "spec_z"]: 
                    input_cols.add(col)

    valid_flux_reads = {c for c in flux_dependencies if c in native_schema}
    input_cols.update(valid_flux_reads)
    
    return list(input_cols), list(final_keep_cols), actions, all_filters

