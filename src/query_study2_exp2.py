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
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils.api_client import VLLMClient, compute_distribution_stats
from utils.config import ExperimentConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def construct_appropriateness_prompt(trial: Dict) -> str:
    """
    Construct prompt for appropriateness judgment

    Args:
        trial: Dictionary with trial data from study2.csv

    Returns:
        Formatted prompt string
    """
    precontext = trial['Precontext'].strip()
    scenario = trial['Scenario'].strip()
    utterance = trial['Utterance'].strip()
    goal = trial['Goal'].strip()
    state = trial['State'].strip()

    prompt = f"""{precontext}

{scenario}

{trial['SP_Name']} responds: "{utterance}"

Context:
- {goal}
- The quality of {trial['LS_Name']}'s work is: {state}

Was {trial['SP_Name']}'s response appropriate or inappropriate? Please respond with "appropriate" or "inappropriate"."""

    return prompt


def construct_quality_rating_prompt(trial: Dict) -> str:
    """
    Construct prompt for quality rating

    Args:
        trial: Dictionary with trial data from study2.csv

    Returns:
        Formatted prompt string
    """
    precontext = trial['Precontext'].strip()
    scenario = trial['Scenario'].strip()
    utterance = trial['Utterance'].strip()
    goal = trial['Goal'].strip()
    state = trial['State'].strip()

    prompt = f"""{precontext}

{scenario}

{trial['SP_Name']} responds: "{utterance}"

Context:
- {goal}
- The quality of {trial['LS_Name']}'s work is: {state}

How would you rate {trial['SP_Name']}'s response? Please respond with one word: "excellent", "good", "neutral", "poor", or "terrible"."""

    return prompt


def process_trial(
    trial: Dict,
    client: VLLMClient,
    model_name: str,
) -> Dict:
    """
    Process a single trial with probability extraction

    Args:
        trial: Trial data dictionary
        client: VLLMClient instance
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

    # Query 1: Appropriateness (binary)
    appropriateness_tokens = ["appropriate", "inappropriate"]
    try:
        appropriateness_prompt = construct_appropriateness_prompt(trial)
        appropriateness_probs = client.extract_binary_probabilities(
            appropriateness_prompt,
            appropriateness_tokens
        )

        result['prob_appropriate'] = appropriateness_probs.get('appropriate', 0.5)
        result['prob_inappropriate'] = appropriateness_probs.get('inappropriate', 0.5)

        # Compute entropy
        p_app = result['prob_appropriate']
        p_inapp = result['prob_inappropriate']
        entropy_app = 0.0
        if p_app > 1e-10:
            entropy_app -= p_app * np.log2(p_app)
        if p_inapp > 1e-10:
            entropy_app -= p_inapp * np.log2(p_inapp)
        result['entropy_appropriateness'] = entropy_app

    except Exception as e:
        logger.error(f"Error querying appropriateness for trial: {e}")
        result['prob_appropriate'] = 0.5
        result['prob_inappropriate'] = 0.5
        result['entropy_appropriateness'] = 1.0

    # Query 2: Quality rating (5-point scale)
    quality_tokens = ["excellent", "good", "neutral", "poor", "terrible"]
    try:
        quality_prompt = construct_quality_rating_prompt(trial)
        quality_probs, _ = client.extract_token_probabilities(
            quality_prompt,
            quality_tokens
        )

        for token in quality_tokens:
            result[f'prob_quality_{token}'] = quality_probs.get(token, 0.0)

        # Compute entropy
        entropy_quality = 0.0
        for token in quality_tokens:
            p = quality_probs.get(token, 0.0)
            if p > 1e-10:
                entropy_quality -= p * np.log2(p)
        result['entropy_quality'] = entropy_quality

        # Compute mean quality score (excellent=5, good=4, neutral=3, poor=2, terrible=1)
        quality_scores = {
            'excellent': 5,
            'good': 4,
            'neutral': 3,
            'poor': 2,
            'terrible': 1,
        }
        mean_quality = sum(
            quality_probs.get(token, 0.0) * score
            for token, score in quality_scores.items()
        )
        result['mean_quality_score'] = mean_quality

    except Exception as e:
        logger.error(f"Error querying quality for trial: {e}")
        # Uniform distribution as fallback
        for token in quality_tokens:
            result[f'prob_quality_{token}'] = 1.0 / len(quality_tokens)
        result['entropy_quality'] = np.log2(len(quality_tokens))
        result['mean_quality_score'] = 3.0

    return result


def run_experiment(
    input_file: Path,
    output_file: Path,
    endpoint: str,
    model_name: str,
    limit: int = None,
):
    """
    Run Study 2 Experiment 2

    Args:
        input_file: Path to study2.csv
        output_file: Path to save results
        endpoint: vLLM server endpoint URL
        model_name: Name of model being queried
        limit: Optional limit on number of trials (for testing)
    """
    logger.info(f"Starting Study 2 Experiment 2 (Probability Distributions)")
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Endpoint: {endpoint}")
    logger.info(f"Model: {model_name}")

    # Load input data
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} trials")

    # Apply limit if specified
    if limit:
        df = df.head(limit)
        logger.info(f"Limited to {len(df)} trials")

    total_queries = len(df) * 2
    logger.info(f"Total queries to make: {total_queries} (2 per trial)")

    # Initialize client
    config = ExperimentConfig()
    client = VLLMClient(base_url=endpoint, config=config)

    # Process trials
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing trials"):
        trial = row.to_dict()
        result = process_trial(trial, client, model_name)
        results.append(result)

        # Log progress every 100 trials
        if (idx + 1) % 100 == 0:
            queries_completed = (idx + 1) * 2
            logger.info(f"Completed {idx + 1}/{len(df)} trials ({queries_completed}/{total_queries} queries)")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    logger.info(f"Output has {len(results_df.columns)} columns")

    # Save output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)
    logger.info(f"Saved results to {output_file}")

    # Note: Validation for Study 2 Exp 2 not yet implemented in validation.py
    logger.info("Note: Formal validation not yet implemented for Study 2 Exp 2")

    # Close client
    client.close()

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
        '--endpoint',
        type=str,
        required=True,
        help='vLLM server endpoint (e.g., http://localhost:8000)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='gemma-2-2b-it',
        help='Model name for output metadata'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of trials (for testing)'
    )

    args = parser.parse_args()

    run_experiment(
        input_file=args.input,
        output_file=args.output,
        endpoint=args.endpoint,
        model_name=args.model_name,
        limit=args.limit,
    )


if __name__ == '__main__':
    main()
