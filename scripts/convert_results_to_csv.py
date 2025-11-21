#!/usr/bin/env python3
"""
Convert JSON experiment results to CSV for R analysis

Usage:
    python3 scripts/convert_results_to_csv.py

Converts all JSON files in outputs/results/ to CSV format,
combining all models into unified datasets.
"""

import json
import pandas as pd
from pathlib import Path
import glob

def convert_json_to_dataframe(json_file):
    """Load JSON and convert to pandas DataFrame"""
    with open(json_file) as f:
        data = json.load(f)
    return pd.DataFrame(data)

def main():
    results_dir = Path("outputs/results")
    output_dir = Path("outputs/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Converting JSON Results to CSV for R Analysis")
    print("="*70)
    print()

    # Study 1 Experiment 1
    print("[1/2] Processing Study 1 Experiment 1...")
    study1_files = glob.glob(str(results_dir / "study1_exp1_*.json"))

    if study1_files:
        study1_dfs = []
        for file in sorted(study1_files):
            print(f"  Loading {Path(file).name}...")
            df = convert_json_to_dataframe(file)
            study1_dfs.append(df)

        # Combine all models
        study1_combined = pd.concat(study1_dfs, ignore_index=True)

        # Save combined CSV
        output_file = output_dir / "study1_exp1_all_models.csv"
        study1_combined.to_csv(output_file, index=False)
        print(f"  ✅ Saved: {output_file}")
        print(f"     Total rows: {len(study1_combined)}")
        print(f"     Models: {study1_combined['model_name'].unique().tolist()}")
    else:
        print("  No Study 1 Exp 1 files found")

    print()

    # Study 2 Experiment 1
    print("[2/2] Processing Study 2 Experiment 1...")
    study2_files = glob.glob(str(results_dir / "study2_exp1_*.json"))

    if study2_files:
        study2_dfs = []
        for file in sorted(study2_files):
            print(f"  Loading {Path(file).name}...")
            df = convert_json_to_dataframe(file)
            study2_dfs.append(df)

        # Combine all models
        study2_combined = pd.concat(study2_dfs, ignore_index=True)

        # Save combined CSV
        output_file = output_dir / "study2_exp1_all_models.csv"
        study2_combined.to_csv(output_file, index=False)
        print(f"  ✅ Saved: {output_file}")
        print(f"     Total rows: {len(study2_combined)}")
        print(f"     Models: {study2_combined['model_name'].unique().tolist()}")
    else:
        print("  No Study 2 Exp 1 files found")

    print()
    print("="*70)
    print("✅ Conversion Complete!")
    print("="*70)
    print()
    print("Output files in: outputs/analysis/")
    print()
    print("Next steps:")
    print("  1. Open R/RStudio")
    print("  2. Load: outputs/analysis/study1_exp1_all_models.csv")
    print("  3. Load: outputs/analysis/study2_exp1_all_models.csv")
    print()

if __name__ == "__main__":
    main()
