# call signature: python run_selection.py --env global_paths.yaml --config config.yaml --profile lsst_gold_test
import argparse
from e2e_selector.config_parser import load_environment_and_profile
from e2e_selector.pipeline import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2E Pipeline Selection Engine")
    parser.add_argument("--env", type=str, default="global_paths.yaml")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--profile", type=str, default="lsst_gold_test")
    parser.add_argument("--foutname", type=str, default="selected_catalog")


    args = parser.parse_args()
    config = load_environment_and_profile(args.env, args.config, args.profile)

    fout_config = config.get("foutname", args.foutname)
    if isinstance(fout_config, list):
        config["foutname"] = "".join(str(item) for item in fout_config)
    else:
        config["foutname"] = str(fout_config)

    if not config["foutname"].endswith(".parquet"):
        config["foutname"] += ".parquet"

    run_pipeline(config)

