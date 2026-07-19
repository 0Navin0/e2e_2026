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
    if isinstance(w, str): return list(w) if len(w) > 1 and w.lower() not in fallbacks else [w.lower()]
    return [x.lower() for x in w]

def build_flux_names(p_type, survey, band):
    suffix = SURVEY_MAP[survey.lower()]["suffix"]
    fmt_b = SURVEY_MAP[survey.lower()]["case"](band)
    return f"flux_{p_type.lower()}{suffix}_{fmt_b}", f"flux_err_{p_type.lower()}{suffix}_{fmt_b}"

def resolve_columns_and_actions(config, schema_names): 
    p_types = parse_list(config.get("photometry_type", {}), ALL_PHOTOMETRY_TYPES)
    surveys = parse_list(config.get("survey", {}), ALL_SURVEYS)
    
    # Extract native structural keys
    nat_prop = config.get("native_prop", {}).get("nonflux_prop", {})
    requested_props = set(nat_prop.get("which", [])) if not nat_prop.get("all") else {c for c in schema_names if "flux" not in c.lower()}
    requested_props.add("objectid") # Always preserve object identifier 
    
    input_cols = {c for c in requested_props if c in schema_names and c not in ["phot_z", "spec_z"]}
    final_keep_cols = set(requested_props)
    
    actions = {"calc_mags": False, "calc_mag_errs": False, "keep_flux": False, "keep_flux_err": False}
    flux_dependencies = set()
    all_filters = []

    # 1. Native Property Filters
    if "native_prop" in config and "filters" in config["native_prop"]:
        all_filters.extend(config["native_prop"]["filters"] or [])

    # 2. Native Flux Property Context Blocks
    flux_block = config.get("flux_cols", {})
    if flux_block.get("all") or flux_block.get("which") or flux_block.get("filters"):
        # any of filter or out storage request, means get the flux col read from the input file.
        actions["keep_flux"] = True
        if flux_block.get("flux_err_cols", False): 
            # if you ask to store, then you definately read it from file.
            actions["keep_flux_err"] = True
        if flux_block.get("filters"):
            # make sure that if filter requires flux err that flux_dependencies
            # variable stores the name of the flux error column too. Infact,
            # any column for filtering should decide what get read from the
            # input file.
            all_filters.extend(flux_block["filters"])
        
        # ensure a mechanism to read bands from two different surveys and associate them with the survey name and column names correctly.
        bands_to_loop = SURVEY_MAP[surveys[0]]["bands"] if flux_block.get("all") else list(flux_block.get("which", []))
        # implement correct verbosity: for a survey print it's bands, photometry_type etc.
        if config[verbose]:
            print(f"bands to loop: {bands_to_loop}")
        for s in surveys:
            for p in p_types:
                for b in bands_to_loop:
                    if SURVEY_MAP[s]["case"](b) in SURVEY_MAP[s]["bands"]:
                        f_col, fe_col = build_flux_names(p, s, b)
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
        
        bands_to_loop = SURVEY_MAP[surveys[0]]["bands"] if mag_cfg.get("all") else list(mag_cfg.get("which", []))
        for s in surveys:
            for p in p_types:
                for b in bands_to_loop:
                    if SURVEY_MAP[s]["case"](b) in SURVEY_MAP[s]["bands"]:
                        f_col, fe_col = build_flux_names(p, s, b)
                        flux_dependencies.add(f_col)
                        
                        suffix = SURVEY_MAP[s]["suffix"]
                        fmt_b = SURVEY_MAP[s]["case"](b)
                        final_keep_cols.add(f"mag_{p}{suffix}_{fmt_b}")
                        if actions["calc_mag_errs"]: 
                            flux_dependencies.add(fe_col)
                            final_keep_cols.add(f"mag_err_{p}{suffix}_{fmt_b}")

    # Process overall filter requirements to guarantee target fluxes are cached
    # note filters only add column names to the dependency list
    # so that these are read from the input file 
    for ii,filt in enumerate(all_filters):
        # each filt is a dict
        print(f"config_parser: filter {ii}, {filt}")
        # manages if filter defined on more than one col
        cols = filt["col"] if isinstance(filt["col"], list) else [filt["col"]]
        qty = filt.get("quantity", "raw")
        
        for col in cols:
            # in future you can add conditions on more derived quantities here
            # you just have to know which raw quantities will need to be read
            # to put a cut/filter on the desired derived quantity.
            if qty == "mag" or qty == "flux":
                # raise an error if photometry_type is not provided for a flux filter
                p = filt.get("photometry_type", "gold")
                # raise an error if survey not known
                s = filt.get("survey", "roman")
                f_col, fe_col = build_flux_names(p, s, col)
                flux_dependencies.add(f_col)
                if qty == "mag": 
                    actions["calc_mags"] = True
                if qty in "flux_err": 
                    flux_dependencies.add(fe_col)
                if qty == "mag_err":
                    actions["calc_mag_errs"] = True
                    flux_dependencies.add(fe_col)
            else:
                if col not in ["phot_z", "spec_z"]: 
                    # this is because the redshifts are not there in the parquet file.
                    # We get the redshift from a different file.
                    input_cols.add(col)

    valid_flux_reads = {c for c in flux_dependencies if c in schema_names}
    input_cols.update(valid_flux_reads)
    
    return list(input_cols), list(final_keep_cols), actions, all_filters

