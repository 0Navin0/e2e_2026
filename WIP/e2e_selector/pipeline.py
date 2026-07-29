import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow as pa
import numpy as np
import h5py
import json
import os
from .config_parser import resolve_columns_and_actions
from .utils import (
        calculate_mags, calculate_mag_errors, ensure_output_dir, 
        generate_sampling_mask, compute_dynamic_batch_size,
        apply_filters
)
import time

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

    out_dir = ensure_output_dir(root_outdir, config.get("outdir", ""))
    final_output_filepath = os.path.join(out_dir, config["foutname"])

    # Establish an optimized output Parquet schema
    dataset = ds.dataset(parquet_path, format="parquet")
    scanner = dataset.scanner(columns=input_cols, batch_size=100)
    sample_batch = next(scanner.to_batches())
    sample_df = sample_batch.slice(0, 1).to_pandas()
    sample_df["phot_z"] = 0.0
    sample_df["spec_z"] = 0.0
    if actions["calc_mags"]:
        for c in final_keep_cols:
            if c not in sample_df.columns:
                sample_df[c] = 0.0
            
    writer_schema = pa.Schema.from_pandas(sample_df[final_keep_cols])
    
    # Pack your general reproducibility metadata *at the very beginning* into the schema
    custom_metadata = {"random_seed": str(seed), "config_profile": json.dumps(config)}
    combined_meta = {**{k.encode(): v.encode() for k, v in custom_metadata.items()}, **(writer_schema.metadata or {})}
    writer_schema = writer_schema.with_metadata(combined_meta)

    batch_idx = 1
    global_row_offset = 0
    total_saved_objects = 0    

    dataset = ds.dataset(parquet_path, format="parquet")
    fragments = list(dataset.get_fragments())

    with h5py.File(hdf5_path, 'r') as h5_f, pq.ParquetWriter(final_output_filepath, writer_schema, compression='snappy') as writer:
        z_phot_ds, z_spec_ds = h5_f["z_phot"], h5_f["z_spec"]
        
        # 2. Iterate through each physical fragment inside the Parquet storage file
        for fragment in fragments:
            # Stream the natural internal record chunks matching our requested columns list
            for record_batch in fragment.to_batches(columns=input_cols, batch_size=50_000):
                batch_start_time = time.perf_counter()
                
                current_batch_size = len(record_batch)
                start_idx = global_row_offset
                end_idx = start_idx + current_batch_size
                
                # 3. Apply the mask at the PyArrow level BEFORE converting to Pandas
                if sampled_mask is not None:
                    # Find indices belonging strictly to this batch window
                    valid_indices = sampled_mask[(sampled_mask >= start_idx) & (sampled_mask < end_idx)]
                    if len(valid_indices) == 0:
                        global_row_offset += current_batch_size
                        batch_idx += 1
                        continue
                    
                    # Convert to local batch offsets
                    local_rows = valid_indices - start_idx
                    
                    # DISK-LEVEL FILTER: Slice the Arrow RecordBatch using zero-copy memory offsets
                    # This guarantees we only send 1% of the data to the heavy Pandas converter!
                    record_batch = record_batch.take(pa.array(local_rows))
                
                # 4. Materialize only the remaining 1% of rows into a Pandas DataFrame
                batch_df = record_batch.to_pandas()
                
                # Map standard companion redshift coordinates seamlessly
                batch_df["phot_z"] = z_phot_ds[start_idx:end_idx][local_rows] if sampled_mask is not None else z_phot_ds[start_idx:end_idx]
                batch_df["spec_z"] = z_spec_ds[start_idx:end_idx][local_rows] if sampled_mask is not None else z_spec_ds[start_idx:end_idx]
                
                if len(batch_df) == 0:
                    global_row_offset += current_batch_size
                    batch_idx += 1
                    continue

                # 5. Compute astronomical transformations
                if actions["calc_mags"] or actions["calc_mag_errs"]:
                    flux_cols_present = [c for c in batch_df.columns if "flux" in c and "err" not in c]
                    for f_col in flux_cols_present:
                        fe_col = f_col.replace("flux_", "flux_err_")
                        m_col = f_col.replace("flux_", "mag_")
                        me_col = f_col.replace("flux_", "mag_err_")
                        
                        if actions["calc_mags"]:
                            batch_df[m_col] = calculate_mags(batch_df[f_col].values)
                        if actions["calc_mag_errs"] and fe_col in batch_df.columns:
                            batch_df[me_col] = calculate_mag_errors(batch_df[f_col].values, batch_df[fe_col].values)

                # 6. Apply physics property configuration filters
                filtered_batch = apply_filters(batch_df, all_filters)
                matched_count = len(filtered_batch)
                
                if matched_count > 0:
                    out_chunk = pa.Table.from_pandas(filtered_batch[final_keep_cols], schema=writer_schema)
                    writer.write_table(out_chunk)
                    total_saved_objects += matched_count

                batch_duration = time.perf_counter() - batch_start_time
                throughput = current_batch_size / batch_duration / 1e6
                print(f"[Batch {batch_idx:02d}] Rows: {start_idx:,} to {end_idx:,} | "
                      f"Time: {batch_duration:.2f}s | Speed: {throughput:.2f}M rows/s | "
                      f"Kept: {matched_count:,} objects")
                
                global_row_offset += current_batch_size
                batch_idx += 1

    print("-" * 80)
    print(f"Pipeline finished successfully. Catalog saved to: {final_output_filepath}")
    total_duration = time.perf_counter() - pipeline_start_time
    print(f"TOTAL EXECUTION: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print(f"Final Selection Catalog Row Count: {total_saved_objects:,}")
    print("=" * 80)
