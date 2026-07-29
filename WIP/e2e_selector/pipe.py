# pipeline.py
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow as pa
import numpy as np
import h5py
import json
import os
import time

from .schema_inspector import get_native_columns
from .derived_quantities import derived_registry
from .config_parser import resolve_columns_and_filters
from .utils import (
    ensure_output_dir,
    generate_sampling_mask,
    compute_dynamic_batch_size,
    apply_filters
)


def run_pipeline(config):
    pipeline_start_time = time.perf_counter()

    verbose = config.get("verbose", False)
    if verbose:
        print(json.dumps(config, indent=4))

    parquet_path, hdf5_path = config["source_in"], config["redshift_in"]
    root_outdir, seed = config["root_output"], config["global_seed"]

    # 1. Inspect source files and resolve operational schema & filters
    native_schema = get_native_columns(parquet_path=parquet_path, hdf5_path=hdf5_path)
    resolved = resolve_columns_and_filters(config, native_schema)

    parquet_cols = resolved["parquet_cols"]
    hdf5_cols = resolved["hdf5_cols"]
    final_keep_cols = resolved["final_keep_cols"]
    derived_to_compute = resolved["derived_to_compute"]
    unified_filters = resolved["unified_filters"]

    pq_file = pq.ParquetFile(parquet_path)
    total_objects = pq_file.metadata.num_rows

    if verbose:
        print("\n--- RESOLVED SCHEMA SUMMARY ---")
        print("Parquet Columns to Read:", parquet_cols)
        print("HDF5 Columns to Read:   ", hdf5_cols)
        print("Derived to Compute:     ", derived_to_compute)
        print("Final Output Columns:   ", final_keep_cols)
        print("Active Filters:         ", len(unified_filters))
        print("-------------------------------\n")

    sampled_mask = generate_sampling_mask(total_objects, config.get("downsample", {}), seed)
    batch_size = compute_dynamic_batch_size(len(parquet_cols))
    print(f"Total entries: {total_objects:,} | Active Parquet columns: {len(parquet_cols)} | Batch size: {batch_size:,} rows")

    out_dir = ensure_output_dir(root_outdir, config.get("outdir", ""))
    final_output_filepath = os.path.join(out_dir, config["foutname"] if isinstance(config["foutname"], str) else "".join(config["foutname"]))

    # 2. Build output Parquet writer schema from dummy record
    dataset = ds.dataset(parquet_path, format="parquet")
    scanner = dataset.scanner(columns=parquet_cols, batch_size=100)
    sample_batch = next(scanner.to_batches())
    sample_df = sample_batch.slice(0, 1).to_pandas()

    for h_col in ["phot_z", "spec_z"]:
        if h_col in hdf5_cols or h_col in final_keep_cols:
            sample_df[h_col] = 0.0

    for d_col in derived_to_compute:
        if d_col in derived_registry.list_all():
            sample_df[d_col] = derived_registry.compute(d_col, sample_df)

    writer_schema = pa.Schema.from_pandas(sample_df[final_keep_cols])

    # Metadata tagging
    custom_metadata = {"random_seed": str(seed), "config_profile": json.dumps(config)}
    combined_meta = {**{k.encode(): str(v).encode() for k, v in custom_metadata.items()}, **(writer_schema.metadata or {})}
    writer_schema = writer_schema.with_metadata(combined_meta)

    batch_idx = 1
    global_row_offset = 0
    total_saved_objects = 0

    fragments = list(dataset.get_fragments())

    # 3. Stream record batches through filter & derivation engine
    with h5py.File(hdf5_path, 'r') as h5_f, pq.ParquetWriter(final_output_filepath, writer_schema, compression='snappy') as writer:
        has_zphot = "z_phot" in h5_f
        has_zspec = "z_spec" in h5_f

        for fragment in fragments:
            for record_batch in fragment.to_batches(columns=parquet_cols, batch_size=50_000):
                batch_start_time = time.perf_counter()

                current_batch_size = len(record_batch)
                start_idx = global_row_offset
                end_idx = start_idx + current_batch_size

                # Downsampling filter at Arrow level
                if sampled_mask is not None:
                    valid_indices = sampled_mask[(sampled_mask >= start_idx) & (sampled_mask < end_idx)]
                    if len(valid_indices) == 0:
                        global_row_offset += current_batch_size
                        batch_idx += 1
                        continue

                    local_rows = valid_indices - start_idx
                    record_batch = record_batch.take(pa.array(local_rows))
                else:
                    local_rows = None

                # Materialize batch into Pandas
                batch_df = record_batch.to_pandas()

                # Attach HDF5 redshift coordinates
                if "phot_z" in final_keep_cols or "phot_z" in hdf5_cols:
                    if has_zphot:
                        z_arr = h5_f["z_phot"][start_idx:end_idx]
                        batch_df["phot_z"] = z_arr[local_rows] if local_rows is not None else z_arr

                if "spec_z" in final_keep_cols or "spec_z" in hdf5_cols:
                    if has_zspec:
                        z_arr = h5_f["z_spec"][start_idx:end_idx]
                        batch_df["spec_z"] = z_arr[local_rows] if local_rows is not None else z_arr

                if len(batch_df) == 0:
                    global_row_offset += current_batch_size
                    batch_idx += 1
                    continue

                # Compute derived quantities dynamically
                for d_col in derived_to_compute:
                    batch_df[d_col] = derived_registry.compute(d_col, batch_df)

                # Apply unified filters (including converted native flux cuts)
                filtered_batch = apply_filters(batch_df, unified_filters)
                matched_count = len(filtered_batch)

                # Write out matched records
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
