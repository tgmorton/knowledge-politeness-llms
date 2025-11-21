#!/usr/bin/env python3
"""
Extract structured data from experiment responses using simple pattern matching

No LLM needed - just regex extraction for primary coding:
- Study 1: Extract numeric response (0-3)
- Study 2: Extract copula (was/wasn't) and quality word (terrible/bad/good/amazing)

Usage:
    python3 scripts/extract_structured_data.py
"""

import json
import pandas as pd
from pathlib import Path
import glob
import re

def extract_study1_number(response):
    """Extract numeric answer (0-3) from response"""
    # Try to find a digit 0-3
    match = re.search(r'\b([0-3])\b', str(response))
    if match:
        return int(match.group(1))

    # Try word forms
    word_to_num = {
        'zero': 0, 'none': 0, 'no': 0,
        'one': 1, 'single': 1,
        'two': 2, 'both': 2,
        'three': 3, 'all': 3
    }

    response_lower = str(response).lower()
    for word, num in word_to_num.items():
        if word in response_lower:
            return num

    return None  # Could not extract

def extract_study2_coding(response):
    """
    Extract copula and quality word from Study 2 politeness response

    Returns:
        dict with keys: copula, quality, coded_response
    """
    response_str = str(response).lower()

    # Extract copula (was/wasn't)
    copula = None
    if re.search(r"\bwasn't\b", response_str):
        copula = "wasn't"
    elif re.search(r"\bwas\b", response_str):
        copula = "was"

    # Extract quality word
    quality = None
    for word in ['terrible', 'bad', 'good', 'amazing']:
        if word in response_str:
            quality = word
            break

    # Construct coded response if both found
    coded_response = None
    if copula and quality:
        coded_response = f"{copula}_{quality}"

    return {
        'copula': copula,
        'quality': quality,
        'coded_response': coded_response
    }

def process_study1():
    """Process Study 1 Experiment 1 results"""
    results_dir = Path("outputs/results")
    output_dir = Path("outputs/analysis")

    print("[1/2] Processing Study 1 Experiment 1...")
    study1_files = glob.glob(str(results_dir / "study1_exp1_*.json"))

    if not study1_files:
        print("  No Study 1 files found")
        return None

    all_data = []
    for file in sorted(study1_files):
        print(f"  Loading {Path(file).name}...")
        with open(file) as f:
            data = json.load(f)
        all_data.extend(data)

    df = pd.DataFrame(all_data)

    # Extract numeric response
    print("  Extracting numeric responses...")
    df['response_numeric'] = df['response'].apply(extract_study1_number)

    # Flag extraction failures
    extraction_failed = df['response_numeric'].isna().sum()
    total = len(df)
    print(f"  ✅ Extracted: {total - extraction_failed}/{total} ({100*(total-extraction_failed)/total:.1f}%)")

    if extraction_failed > 0:
        print(f"  ⚠️  Failed to extract: {extraction_failed} responses")
        print("  Sample failures:")
        failures = df[df['response_numeric'].isna()][['model_name', 'response']].head(5)
        for _, row in failures.iterrows():
            print(f"    {row['model_name']}: '{row['response']}'")

    # Save
    output_file = output_dir / "study1_exp1_structured.csv"
    df.to_csv(output_file, index=False)
    print(f"  💾 Saved: {output_file}")

    return df

def process_study2():
    """Process Study 2 Experiment 1 results"""
    results_dir = Path("outputs/results")
    output_dir = Path("outputs/analysis")

    print("\n[2/2] Processing Study 2 Experiment 1...")
    study2_files = glob.glob(str(results_dir / "study2_exp1_*.json"))

    if not study2_files:
        print("  No Study 2 files found")
        return None

    all_data = []
    for file in sorted(study2_files):
        print(f"  Loading {Path(file).name}...")
        with open(file) as f:
            data = json.load(f)
        all_data.extend(data)

    df = pd.DataFrame(all_data)

    # Extract copula and quality
    print("  Extracting copula and quality words...")
    coding = df['response'].apply(extract_study2_coding)
    df['copula'] = coding.apply(lambda x: x['copula'])
    df['quality'] = coding.apply(lambda x: x['quality'])
    df['coded_response'] = coding.apply(lambda x: x['coded_response'])

    # Extraction statistics
    total = len(df)
    copula_extracted = df['copula'].notna().sum()
    quality_extracted = df['quality'].notna().sum()
    both_extracted = df['coded_response'].notna().sum()

    print(f"  ✅ Copula extracted: {copula_extracted}/{total} ({100*copula_extracted/total:.1f}%)")
    print(f"  ✅ Quality extracted: {quality_extracted}/{total} ({100*quality_extracted/total:.1f}%)")
    print(f"  ✅ Both extracted: {both_extracted}/{total} ({100*both_extracted/total:.1f}%)")

    # Distribution of coded responses
    print("\n  Coded response distribution:")
    coded_dist = df['coded_response'].value_counts()
    for code, count in coded_dist.head(10).items():
        print(f"    {code}: {count} ({100*count/total:.1f}%)")

    # Flag extraction failures
    extraction_failed = df['coded_response'].isna().sum()
    if extraction_failed > 0:
        print(f"\n  ⚠️  Failed to extract both: {extraction_failed} responses")
        print("  Sample failures:")
        failures = df[df['coded_response'].isna()][['model_name', 'response']].head(5)
        for _, row in failures.iterrows():
            print(f"    {row['model_name']}: '{row['response']}'")

    # Save
    output_file = output_dir / "study2_exp1_structured.csv"
    df.to_csv(output_file, index=False)
    print(f"  💾 Saved: {output_file}")

    return df

def main():
    output_dir = Path("outputs/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Extracting Structured Data from Experiment Responses")
    print("="*70)
    print()

    # Process both studies
    study1_df = process_study1()
    study2_df = process_study2()

    print()
    print("="*70)
    print("✅ Extraction Complete!")
    print("="*70)
    print()
    print("Output files:")
    print("  - outputs/analysis/study1_exp1_structured.csv")
    print("  - outputs/analysis/study2_exp1_structured.csv")
    print()
    print("New columns:")
    print("  Study 1: response_numeric (0-3)")
    print("  Study 2: copula (was/wasn't), quality (terrible/bad/good/amazing), coded_response")
    print()

if __name__ == "__main__":
    main()
