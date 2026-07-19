import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow as pa
import pandas as pd
import numpy as np
import h5py
import json
import os
from .config_parser import resolve_columns_and_actions, SURVEY_MAP
from .utils import calculate_mags_only, calculate_mag_errors, evaluate_condition, ensure_output_dir
import time

def generate_sampling_mask(total_objects, downsample_cfg, seed):
    if not downsample_cfg: return None
    rng = np.random.default_rng(seed)
    fraction, factor = downsample_cfg.get("fraction"), downsample_cfg.get("factor")
    if fraction is not None: 
        target_size = int(total_objects * fraction)
    elif factor is not None: 
        target_size = int(total_objects // factor)
    else: 
        return None
    
    sampled_indices = rng.choice(total_objects, size=target_size, replace=False)
    sampled_indices.sort()
    return sampled_indices

def compute_dynamic_batch_size(num_columns, max_cells=40_000_000):
    """Enforces dynamic row limit thresholds depending on active parameter counts [6]."""
    return max(10000, max_cells // num_columns)

def apply_filters(df, filters):
    master_mask = np.ones(len(df), dtype=bool)
    for filt in filters:
        cols = filt["col"] if isinstance(filt["col"], list) else [filt["col"]]
        qty = filt.get("quantity", "raw")
        
        for col in cols:
            if qty == "mag" or qty == "flux":
                p = filt.get("photometry_type", "gold")
                s = filt.get("survey", "roman").lower()
                suffix = SURVEY_MAP[s]["suffix"]
                fmt_b = SURVEY_MAP[s]["case"](col)
                target_key = f"{qty}_{p}{suffix}_{fmt_b}"
            else:
                target_key = col
                
            if target_key not in df.columns: continue
                
            if filt["type"] == "range":
                bounds, ineqs = filt["bounds"], filt["inequality"]
                # Evaluate range limits simultaneously
                master_mask &= evaluate_condition(df[target_key], ineqs[0], bounds[0])
                master_mask &= evaluate_condition(df[target_key], ineqs[1], bounds[1])
            else:
                master_mask &= evaluate_condition(df[target_key], filt["inequality"], filt["value"])
    return df[master_mask]

def run_pipeline(config):
    # Initialize high-level global pipeline timer
    pipeline_start_time = time.perf_counter()

    verbose = config["verbose"]
    if verbose:
        print(json.dumps(config, indent=4))
    parquet_path, hdf5_path = config["source_in"], config["redshift_in"]
    root_outdir, seed = config["root_output"], config["global_seed"]
    
    pq_file = pq.ParquetFile(parquet_path)
    total_objects = pq_file.metadata.num_rows
    
    input_cols, final_keep_cols, actions, all_filters = resolve_columns_and_actions(config, pq_file.metadata.schema.names)
    if verbose:
        print("Debug: input cols\n", input_cols)
        print("Debug: keep cols\n", final_keep_cols)
        print("Debug: actions\n", actions)
    sampled_mask = generate_sampling_mask(total_objects, config.get("downsample", {}), seed)
    
    # Calculate execution batch chunk constraints explicitly
    batch_size = compute_dynamic_batch_size(len(input_cols))
    print(f"Total entries: {total_objects:,} | Active columns: {len(input_cols)} | Batch size: {batch_size:,} rows")
    
    output_df_list = []
    dataset = ds.dataset(parquet_path, format="parquet")

    batch_idx = 1
    total_processed_rows = 0
    
    with h5py.File(hdf5_path, 'r') as h5_f:
        z_phot_ds, z_spec_ds = h5_f["z_phot"], h5_f["z_spec"]
        
        for start_idx in range(0, total_objects, batch_size):
            batch_start_time = time.perf_counter()

            end_idx = min(start_idx + batch_size, total_objects)
            current_batch_size = end_idx - start_idx
            
            if sampled_mask is not None:
                valid_indices = sampled_mask[(sampled_mask >= start_idx) & (sampled_mask < end_idx)]
                if len(valid_indices) == 0:
                    continue
                local_rows = valid_indices - start_idx
            else:
                local_rows = None
                
            # Read exactly the calculated chunk slice cleanly from disk
            table_chunk = dataset.head(end_idx, columns=input_cols).slice(start_idx, current_batch_size)
            batch_df = table_chunk.to_pandas()
            
            # Map standardized redshift headers
            batch_df["phot_z"] = z_phot_ds[start_idx:end_idx]
            batch_df["spec_z"] = z_spec_ds[start_idx:end_idx]
            
            if local_rows is not None:
                batch_df = batch_df.iloc[local_rows].copy()
                
            # Trigger isolated mathematical operations when requested in configurations
            if actions["calc_mags"] or actions["calc_mag_errs"]:
                flux_cols_present = [c for c in batch_df.columns if "flux" in c and "err" not in c]
                for f_col in flux_cols_present:
                    fe_col = f_col.replace("flux_", "flux_err_")
                    m_col = f_col.replace("flux_", "mag_")
                    me_col = f_col.replace("flux_", "mag_err_")
                    
                    if actions["calc_mags"]:
                        batch_df[m_col] = calculate_mags_only(batch_df[f_col].values)
                    if actions["calc_mag_errs"] and fe_col in batch_df.columns:
                        batch_df[me_col] = calculate_mag_errors(batch_df[f_col].values, batch_df[fe_col].values)

            # Apply complete combined filter criteria lists
            # there can be two types of filters: one on raw columns which
            # should be applied at the time of reading in the data. And others,
            # to apply on derived quantities. Right here!
            filtered_batch = apply_filters(batch_df, all_filters)
            
            matched_count = len(filtered_batch)
            if matched_count > 0:
                keep_now = [c for c in final_keep_cols if c in filtered_batch.columns]
                output_df_list.append(filtered_batch[keep_now])

            batch_end_time = time.perf_counter()
            batch_duration = batch_end_time - batch_start_time
            throughput = current_batch_size / batch_duration / 1e6 # Millions of rows per second
            print(f"[Batch {batch_idx:02d}] Rows: {start_idx:,} to {end_idx:,} | "
                  f"Time: {batch_duration:.2f}s | Speed: {throughput:.2f}M rows/s | "
                  f"Kept: {matched_count:,} objects")
            
            batch_idx += 1

    print("-" * 80)


    if not output_df_list:
        print("Processing completed. Zero matching rows found!")
        return

    io_start = time.perf_counter()
    final_df = pd.concat(output_df_list, ignore_index=True)
    out_dir = ensure_output_dir(root_outdir, config.get("outdir", ""))
    final_output_filepath = os.path.join(out_dir, config["foutname"])
    
    # Save the file table along with reproducible configuration metadata strings
    table = pa.Table.from_pandas(final_df)
    custom_metadata = {"random_seed": str(seed), "config_profile": json.dumps(config)}
    combined_meta = {**{k.encode(): v.encode() for k, v in custom_metadata.items()}, **(table.schema.metadata or {})}
    # is this step creating another copy of the full dataset? If so, start an
    # empty table at the beginning with this metadata and add the full data to
    # this parquet table in the end.
    pq.write_table(table.replace_schema_metadata(combined_meta), final_output_filepath)
    print(f"Pipeline finished successfully. Catalog saved to: {final_output_filepath}")

    pipeline_end_time = time.perf_counter()
    total_duration = pipeline_end_time - pipeline_start_time
    io_duration = pipeline_end_time - io_start
    
    print(f"I/O Save Time  : {io_duration:.2f} seconds")
    print(f"TOTAL EXECUTION: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print(f"Final Selection Catalog Row Count: {len(final_df):,}")
    print("=" * 80)

