#!/usr/bin/env python3
"""
Study 1 Experiment 2: Knowledge Attribution - Probability Distributions

This is the KEY INNOVATION of the Grace Project!

For each trial in study1.csv, generates 5 separate queries:
1-4. State queries: "What percentage probability do you assign that exactly {0,1,2,3} items have the property?"
5. Knowledge query: "Do you think X knows exactly how many items have the property?"

For each query, extracts probability distribution from logprobs.

Output: JSON array with:
- 9 original columns (8 from study1.csv + model_name)
- 44 probability columns (state{0-3}_prob_{0,10,20,...,100})
- 12 summary statistics (state{0-3}_{mean,std,entropy})
- 3 knowledge columns (prob_knowledge_yes, prob_knowledge_no, entropy_knowledge)
"""

import argparse
import logging
import json
import os
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils.model_scorer import ModelScorer
from utils.api_client import compute_distribution_stats
from utils.config import ExperimentConfig, Study1Config
from utils.validation import validate_study1_exp2
from utils.prompts import (
    construct_study1_exp2_state_prompt,
    construct_study1_exp2_knowledge_prompt,
    get_percentage_tokens,
    get_percentage_values,
    get_yesno_tokens,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Prompt construction moved to utils.prompts for consistency across all scripts


def process_trial(
    trial: Dict,
    scorer: ModelScorer,
    model_name: str,
) -> Dict:
    """
    Process a single trial with 5 queries using direct model scoring

    Args:
        trial: Trial data dictionary
        scorer: ModelScorer instance (direct model access)
        model_name: Name of model being used

    Returns:
        Result dictionary with all probability distributions and statistics
    """
    result = {
        # Original columns
        'participant_id': trial['participant_id'],
        'story_shortname': trial['story_shortname'],
        'story_setup': trial['story_setup'],
        'priorQ': trial['priorQ'],
        'speach': trial['speach'],
        'speachQ': trial['speachQ'],
        'knowledgeQ': trial['knowledgeQ'],
        'access': trial['access'],
        'observe': trial['observe'],
        'model_name': model_name,
    }

    # Percentage values (0, 10, 20, ..., 100)
    pct_values = get_percentage_values()
    # Extract numeric tokens (since tokenizer splits "10%" into "10" + "%")
    pct_tokens = get_percentage_tokens()

    # Query 1-4: State probability distributions
    for state in range(4):
        prompt = construct_study1_exp2_state_prompt(trial, state, model_name)

        try:
            # Score all percentage options directly with model
            probs = scorer.score_options(prompt, pct_tokens, normalize=True)

            # Store probabilities
            for pct_val, pct_token in zip(pct_values, pct_tokens):
                result[f'state{state}_prob_{pct_val}'] = probs.get(pct_token, 0.0)

            # Compute summary statistics
            prob_values = [probs.get(token, 0.0) for token in pct_tokens]
            stats = compute_distribution_stats(
                {token: prob_values[i] for i, token in enumerate(pct_tokens)},
                pct_values
            )

            result[f'state{state}_mean'] = stats['mean']
            result[f'state{state}_std'] = stats['std']
            result[f'state{state}_entropy'] = stats['entropy']

        except Exception as e:
            logger.error(f"Error scoring state {state} for trial: {e}")
            # Fill with uniform distribution as fallback
            for pct in pct_values:
                result[f'state{state}_prob_{pct}'] = 1.0 / len(pct_values)
            result[f'state{state}_mean'] = 50.0
            result[f'state{state}_std'] = 0.0
            result[f'state{state}_entropy'] = np.log2(len(pct_values))

    # Query 5: Knowledge question (yes/no)
    prompt = construct_study1_exp2_knowledge_prompt(trial, model_name)

    try:
        # Score yes/no options directly with model
        knowledge_probs = scorer.score_options(prompt, get_yesno_tokens(), normalize=True)

        result['prob_knowledge_yes'] = knowledge_probs.get('yes', 0.5)
        result['prob_knowledge_no'] = knowledge_probs.get('no', 0.5)

        # Compute entropy for knowledge
        p_yes = result['prob_knowledge_yes']
        p_no = result['prob_knowledge_no']
        entropy = 0.0
        if p_yes > 1e-10:
            entropy -= p_yes * np.log2(p_yes)
        if p_no > 1e-10:
            entropy -= p_no * np.log2(p_no)
        result['entropy_knowledge'] = entropy

    except Exception as e:
        logger.error(f"Error querying knowledge for trial: {e}")
        # Uniform distribution as fallback
        result['prob_knowledge_yes'] = 0.5
        result['prob_knowledge_no'] = 0.5
        result['entropy_knowledge'] = 1.0

    return result


def run_experiment(
    input_file: Path,
    output_file: Path,
    model_path: str,
    model_name: str,
    limit: int = None,
    is_reasoning_model: bool = False,
    reasoning_start_token: str = "<think>",
    reasoning_end_token: str = "</think>",
):
    """
    Run Study 1 Experiment 2 using direct model scoring

    Args:
        input_file: Path to study1.csv
        output_file: Path to save results
        model_path: HuggingFace model path (e.g., "google/gemma-2-2b-it")
        model_name: Name of model for output metadata
        limit: Optional limit on number of trials (for testing)
        is_reasoning_model: Whether this is a reasoning model (e.g., DeepSeek-R1)
        reasoning_start_token: Token marking start of reasoning trace
        reasoning_end_token: Token marking end of reasoning trace
    """
    logger.info(f"Starting Study 1 Experiment 2 (Probability Distributions)")
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Reasoning model: {is_reasoning_model}")

    # Load input data
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} trials")

    # Apply limit if specified
    if limit:
        df = df.head(limit)
        logger.info(f"Limited to {len(df)} trials")

    total_queries = len(df) * 5
    logger.info(f"Total queries to score: {total_queries} (5 per trial)")

    # Initialize model scorer
    # Get cache dir from env (set by Kubernetes manifest)
    cache_dir = os.getenv('HF_HOME', None)
    if cache_dir:
        logger.info(f"Using model cache: {cache_dir}")

    logger.info("Loading model for direct scoring...")
    scorer = ModelScorer(
        model_name=model_path,
        cache_dir=cache_dir,
        is_reasoning_model=is_reasoning_model,
        reasoning_start_token=reasoning_start_token,
        reasoning_end_token=reasoning_end_token,
    )

    # Process trials
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing trials"):
        trial = row.to_dict()
        result = process_trial(trial, scorer, model_name)
        results.append(result)

        # Log progress every 10 trials
        if (idx + 1) % 10 == 0:
            queries_completed = (idx + 1) * 5
            logger.info(f"Completed {idx + 1}/{len(df)} trials ({queries_completed}/{total_queries} queries)")

    # Verify we have all expected fields
    if results:
        num_keys = len(results[0].keys())
        logger.info(f"Output has {num_keys} fields (expected 68)")

    # Save output as JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved {len(results)} results to {output_file}")

    # Validate output
    logger.info("Validating output...")
    is_valid = validate_study1_exp2(output_file, expected_rows=len(df))

    if is_valid:
        logger.info("✅ Validation passed!")
    else:
        logger.warning("⚠️  Validation failed - check output file")

    # Close scorer
    scorer.close()

    logger.info("Experiment complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Study 1 Experiment 2: Knowledge Attribution - Probability Distributions'
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Path to study1.csv'
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
    parser.add_argument(
        '--reasoning-model',
        action='store_true',
        help='Whether this is a reasoning model (e.g., DeepSeek-R1)'
    )
    parser.add_argument(
        '--reasoning-start-token',
        type=str,
        default='<think>',
        help='Token marking start of reasoning trace (default: <think>)'
    )
    parser.add_argument(
        '--reasoning-end-token',
        type=str,
        default='</think>',
        help='Token marking end of reasoning trace (default: </think>)'
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
        is_reasoning_model=args.reasoning_model,
        reasoning_start_token=args.reasoning_start_token,
        reasoning_end_token=args.reasoning_end_token,
    )


if __name__ == '__main__':
    main()
