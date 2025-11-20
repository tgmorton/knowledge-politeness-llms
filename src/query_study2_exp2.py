#!/usr/bin/env python3
"""
Study 2 Experiment 2: Politeness Judgments - Probability Distributions

Extracts probability distributions for politeness judgments.

For each trial, asks the model to evaluate the appropriateness of the response
and extracts probability distributions over categorical judgments.

Example queries:
- "Was the response appropriate or inappropriate?"
- "How would you rate the response quality?" (good/neutral/bad)

Output: CSV with original columns + probability distributions
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils.model_scorer import ModelScorer
from utils.api_client import compute_distribution_stats
from utils.config import ExperimentConfig
from utils.prompts import construct_study2_exp2_prompt, get_polarity_tokens, get_quality_tokens

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Prompt construction moved to utils.prompts for consistency


def process_trial(
    trial: Dict,
    scorer: ModelScorer,
    model_name: str,
) -> Dict:
    """
    Process a single trial with probability extraction using direct model scoring

    Args:
        trial: Trial data dictionary
        scorer: ModelScorer instance (direct model access)
        model_name: Name of model being used

    Returns:
        Result dictionary with probability distributions
    """
    result = {
        # Original columns
        'Participant_ID': trial['Participant_ID'],
        'Domain': trial['Domain'],
        'Precontext': trial['Precontext'],
        'Scenario': trial['Scenario'],
        'Utterance': trial['Utterance'],
        'Goal': trial['Goal'],
        'State': trial['State'],
        'SP_Name': trial['SP_Name'],
        'LS_Name': trial['LS_Name'],
        'model_name': model_name,
    }

    # Score polarity and quality tokens directly
    # For Study 2 Exp 2, we score the model generating the response with different polarities and qualities
    prompt = construct_study2_exp2_prompt(trial, model_name)

    # Get all possible completions: polarity × quality
    polarities = get_polarity_tokens()  # ["was", "wasn't"]
    qualities = get_quality_tokens()    # ["terrible", "bad", "good", "amazing"]

    # Build all combinations: "It was terrible", "It was bad", etc.
    all_options = [f"It {pol} {qual}" for pol in polarities for qual in qualities]

    try:
        # Score all 8 options (2 polarities × 4 qualities)
        all_probs = scorer.score_options(prompt, all_options, normalize=True)

        # Extract probabilities for each combination
        for pol in polarities:
            for qual in qualities:
                option = f"It {pol} {qual}"
                result[f'prob_{pol}_{qual}'] = all_probs.get(option, 0.0)

        # Marginal probabilities
        # P(was) = sum over all qualities
        prob_was = sum(all_probs.get(f"It was {q}", 0.0) for q in qualities)
        prob_wasnt = sum(all_probs.get(f"It wasn't {q}", 0.0) for q in qualities)

        result['prob_was'] = prob_was
        result['prob_wasnt'] = prob_wasnt

        # Marginal over polarities: P(terrible), P(bad), P(good), P(amazing)
        for qual in qualities:
            prob_qual = sum(all_probs.get(f"It {p} {qual}", 0.0) for p in polarities)
            result[f'prob_{qual}'] = prob_qual

        # Compute entropy over all 8 options
        entropy = 0.0
        for prob in all_probs.values():
            if prob > 1e-10:
                entropy -= prob * np.log2(prob)
        result['entropy_response'] = entropy

    except Exception as e:
        logger.error(f"Error scoring trial: {e}")
        # Uniform distribution as fallback
        for pol in polarities:
            for qual in qualities:
                result[f'prob_{pol}_{qual}'] = 1.0 / 8
        result['prob_was'] = 0.5
        result['prob_wasnt'] = 0.5
        for qual in qualities:
            result[f'prob_{qual}'] = 0.25
        result['entropy_response'] = 3.0  # log2(8)

    return result


def run_experiment(
    input_file: Path,
    output_file: Path,
    model_path: str,
    model_name: str,
    limit: int = None,
):
    """
    Run Study 2 Experiment 2 using direct model scoring

    Args:
        input_file: Path to study2.csv
        output_file: Path to save results
        model_path: HuggingFace model path
        model_name: Name of model for output metadata
        limit: Optional limit on number of trials (for testing)
    """
    logger.info(f"Starting Study 2 Experiment 2 (Probability Distributions)")
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Model name: {model_name}")

    # Load input data
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} trials")

    # Apply limit if specified
    if limit:
        df = df.head(limit)
        logger.info(f"Limited to {len(df)} trials")

    logger.info(f"Total trials to score: {len(df)} (scoring 8 options per trial)")

    # Initialize model scorer
    # Get cache dir from env (set by Kubernetes manifest)
    cache_dir = os.getenv('HF_HOME', None)
    if cache_dir:
        logger.info(f"Using model cache: {cache_dir}")

    logger.info("Loading model for direct scoring...")
    scorer = ModelScorer(model_name=model_path, cache_dir=cache_dir)

    # Process trials
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing trials"):
        trial = row.to_dict()
        result = process_trial(trial, scorer, model_name)
        results.append(result)

        # Log progress every 100 trials
        if (idx + 1) % 100 == 0:
            logger.info(f"Completed {idx + 1}/{len(df)} trials")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    logger.info(f"Output has {len(results_df.columns)} columns")

    # Save output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)
    logger.info(f"Saved results to {output_file}")

    # Note: Validation for Study 2 Exp 2 not yet implemented in validation.py
    logger.info("Note: Formal validation not yet implemented for Study 2 Exp 2")

    # Close scorer
    scorer.close()

    logger.info("Experiment complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Study 2 Experiment 2: Politeness Judgments - Probability Distributions'
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Path to study2.csv'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Path to save output CSV'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='HuggingFace model path (e.g., google/gemma-2-2b-it)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Model name for output metadata (defaults to model-path)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of trials (for testing)'
    )

    args = parser.parse_args()

    # Default model_name to model_path if not specified
    model_name = args.model_name or args.model_path

    run_experiment(
        input_file=args.input,
        output_file=args.output,
        model_path=args.model_path,
        model_name=model_name,
        limit=args.limit,
    )


if __name__ == '__main__':
    main()
