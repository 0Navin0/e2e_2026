import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow as pa
import pandas as pd
import numpy as np
import h5py
import json
from .config_parser import resolve_required_columns, SURVEY_MAP
from .utils import calculate_mags, evaluate_condition

def generate_sampling_mask(total_objects, downsample_cfg, seed=2026):
    """Pre-constructs a boolean evaluation array from the start."""
    rng = np.random.default_rng(seed)
    
    fraction = downsample_cfg.get("fraction")
    factor = downsample_cfg.get("factor")
    
    if fraction is not None:
        target_size = int(total_objects * fraction)
    elif factor is not None:
        target_size = int(total_objects // factor)
    else:
        return None # Keep everything
        
    sampled_indices = rng.choice(total_objects, size=target_size, replace=False)
    sampled_indices.sort()
    return sampled_indices

def compute_dynamic_batch_size(num_sampled_objects, num_columns, max_cells=50_000_000):
    """Calculates dynamically how many rows to process per iteration loop."""
    calculated_rows = max_cells // num_columns
    return max(5000, min(calculated_rows, num_sampled_objects))

def apply_filters(df, filters):
    """Executes dynamic vectorized configuration cuts."""
    master_mask = np.ones(len(df), dtype=bool)
    
    for filt in filters:
        cols = filt["col"] if isinstance(filt["col"], list) else [filt["col"]]
        qty = filt.get("quantity", "raw")
        srv = filt.get("survey", "roman")
        
        for col in cols:
            # Build target name string
            if qty == "mag":
                suffix = SURVEY_MAP[srv]["suffix"]
                band = SURVEY_MAP[srv]["case"](col)
                target_key = f"mag_gold{suffix}_{band}"
            else:
                target_key = col
                
            if target_key not in df.columns:
                continue
                
            if filt["type"] == "range":
                bounds = filt["bounds"]
                ineqs = filt["inequality"]
                mask1 = evaluate_condition(df[target_key], ineqs[0], bounds[0])
                mask2 = evaluate_condition(df[target_key], ineqs[1], bounds[1])
                master_mask &= (mask1 & mask2)
            else:
                master_mask &= evaluate_condition(df[target_key], filt["inequality"], filt["value"])
                
    return df[master_mask]

def run_pipeline(config, root_outdir, seed=2026):
    parquet_path = config["detection_in"]
    hdf5_path = config["flexzboost_in"]
    
    # 1. Access properties via parquet metadata
    pq_file = pq.ParquetFile(parquet_path)
    total_objects = pq_file.metadata.num_rows
    all_schema_names = pq_file.metadata.schema.names
    
    print(f"Total objects identified in catalog: {total_objects:,}")
    
    # 2. Process column structures
    input_cols, final_keep_cols = resolve_required_columns(config, all_schema_names)
    
    # 3. Handle downsampling tracking arrays from the absolute start
    sampled_indices = generate_sampling_mask(total_objects, config["downSample"], seed=seed)
    num_to_process = len(sampled_indices) if sampled_indices is not None else total_objects
    
    # Calculate batch size dynamically
    batch_size = compute_dynamic_batch_size(num_to_process, len(input_cols))
    print(f"Dynamic memory threshold set. Batch processing block size: {batch_size:,} rows")
    
    # 4. Stream and loop chunks via pyarrow dataset API
    dataset = ds.dataset(parquet_path, format="parquet")
    scanner = dataset.scanner(columns=input_cols)
    
    output_df_list = []
    global_row_offset = 0
    
    # Open context reference connection to companion dataset
    with h5py.File(hdf5_path, 'r') as h5_f:
        z_phot_ds = h5_f["z_phot"]
        z_spec_ds = h5_f["z_spec"]
        
        for record_batch in scanner.to_batches():
            batch_df = record_batch.to_pandas()
            batch_len = len(batch_df)
            
            # Map global location indices
            batch_start = global_row_offset
            batch_end = global_row_offset + batch_len
            
            # Add HDF5 metadata alignments
            batch_df["z_phot"] = z_phot_ds[batch_start:batch_end]
            batch_df["z_spec"] = z_spec_ds[batch_start:batch_end]
            
            # Intersect with the pre-constructed downsampling arrays if present
            if sampled_indices is not None:
                # Find elements within this specific batch's boundaries
                valid_in_batch = sampled_indices[(sampled_indices >= batch_start) & (sampled_indices < batch_end)]
                if len(valid_in_batch) == 0:
                    global_row_offset += batch_len
                    continue
                # Localize slice
                local_indices = valid_in_batch - batch_start
                batch_df = batch_df.iloc[local_indices].copy()
            
            # Compute Derived Magnitudes across loop targets
            for survey, s_info in SURVEY_MAP.items():
                suffix = s_info["suffix"]
                for band in s_info["bands"]:
                    f_col = f"flux_gold{suffix}_{band}"
                    fe_col = f"flux_err_gold{suffix}_{band}"
                    m_col = f"mag_gold{suffix}_{band}"
                    me_col = f"mag_err_gold{suffix}_{band}"
                    
                    if f_col in batch_df.columns:
                        m, me = calculate_mags(batch_df, f_col, fe_col)
                        batch_df[m_col] = m
                        batch_df[me_col] = me
            
            # Apply dynamic config filters
            filtered_batch = apply_filters(batch_df, config.get("filters", []))
            
            if len(filtered_batch) > 0:
                # Isolate target requested system items and store chunk references
                cols_to_extract = [c for c in final_keep_cols if c in filtered_batch.columns]
                # Keep redshifts if targeted
                if "z_phot" in filtered_batch.columns: cols_to_extract.extend(["z_phot", "z_spec"])
                
                output_df_list.append(filtered_batch[list(set(cols_to_extract))])
                
            global_row_offset += batch_len

    # Consolidate results cleanly
    if not output_df_list:
        print("No source records matched your filter criteria.")
        return
        
    final_df = pd.concat(output_df_list, ignore_index=True)
    
    # 5. Pack metadata dictionary inside output Parquet file
    from .utils import ensure_output_dir
    out_dir = ensure_output_dir(root_outdir, config["outdir"])
    final_output_filepath = os.path.join(out_dir, "selected_catalog.parquet")
    
    table = pa.Table.from_pandas(final_df)
    custom_metadata = {
        "random_seed": str(seed),
        "config_profile": json.dumps(config)
    }
    # Append to schema metadata dictionary
    existing_meta = table.schema.metadata or {}
    combined_meta = {**existing_meta, **{k.encode(): v.encode() for k, v in custom_metadata.items()}}
    table = table.replace_schema_metadata(combined_meta)
    
    pq.write_table(table, final_output_filepath)
    print(f"Processing Complete. Catalog successfully written to: {final_output_filepath}")

