import yaml

ROMAN_BANDS = list("YJH")
LSST_BANDS = list("ugrizy")
SURVEY_MAP = {
    "roman": {"bands": ROMAN_BANDS, "suffix": "", "case": str.upper},
    "lsst": {"bands": LSST_BANDS, "suffix": "_LSST", "case": str.lower}
}

ALL_PHOTOMETRY_TYPES = ["gold", "pgauss"]
ALL_SURVEYS = ["roman", "lsst"]

def load_environment_and_profile(paths_yaml, config_yaml, profile_name):
    with open(paths_yaml, 'r') as f:
        env = yaml.safe_load(f)
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
    if isinstance(w, str): return [w.lower()]
    return [x.lower() for x in w]

def build_flux_names(p_type, survey, band):
    suffix = SURVEY_MAP[survey.lower()]["suffix"]
    fmt_b = SURVEY_MAP[survey.lower()]["case"](band)
    return f"flux_{p_type.lower()}{suffix}_{fmt_b}", f"flux_err_{p_type.lower()}{suffix}_{fmt_b}"

def resolve_columns_and_actions(config, schema_names):
    p_types = parse_list(config.get("photometry_type", config.get("photometry", {})), ALL_PHOTOMETRY_TYPES)
    surveys = parse_list(config.get("survey", {}), ALL_SURVEYS)
    
    # Extract native structural keys
    nat_prop = config.get("native_prop", {}).get("nonflux_prop", {})
    requested_props = set(nat_prop.get("which", [])) if not nat_prop.get("all") else {c for c in schema_names if "flux" not in c.lower()}.union({"phot_z", "spec_z"})
    requested_props.add("objectid") # Always preserve object identifier 
    
    input_cols = {c for c in requested_props if c in schema_names and c not in ["phot_z", "spec_z"]}
    final_keep_cols = set(requested_props)
    
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

    valid_flux_reads = {c for c in flux_dependencies if c in schema_names}
    input_cols.update(valid_flux_reads)
    
    return list(input_cols), list(final_keep_cols), actions, all_filters

