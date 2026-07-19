import argparse
from e2e_selector.config_parser import load_config
from e2e_selector.pipeline import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2E Cosmological Catalog Selection Engine")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config yaml")
    parser.add_argument("--profile", type=str, default="lsst_gold_y1", help="Target configuration profile name")
    parser.add_argument("--root_out", type=str, default="/work/nlc38/output_base", help="System root storage mount path")
    parser.add_argument("--seed", type=int, default=2026, help="Random number generator seed track")
    
    args = parser.parse_args()
    
    # 1. Resolve configuration maps
    config = load_config(args.config, args.profile)
    
    # 2. Run data processing pipeline
    run_pipeline(config, root_outdir=args.root_out, seed=args.seed)

