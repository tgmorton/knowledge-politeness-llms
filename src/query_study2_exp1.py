#!/usr/bin/env python3
"""
Study 2 Experiment 1: Politeness Judgments - Raw Text Responses

Queries models for politeness judgments given:
- A scenario (speaker asks listener for feedback)
- The listener's response
- The speaker's goal (make speaker feel good vs give informative feedback)
- The quality of work (0-4 hearts)

For each trial in study2.csv:
- Present the scenario and response
- Ask model to judge the response
- Save raw text judgment

Expected output: JSON array with original columns + response, model_name, timestamp
"""

import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
from tqdm import tqdm

from utils.api_client import VLLMClient
from utils.config import ExperimentConfig
from utils.validation import validate_study2_exp1
from utils.model_config import get_model_config
from utils.prompts import construct_study2_exp1_prompt
from utils.replication import (
    add_replication_args,
    initialize_replication,
    shuffle_trials,
    add_replication_metadata,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Prompt construction moved to utils.prompts for consistency across all scripts


def process_trial(
    trial: Dict,
    client: VLLMClient,
    model_name: str,
    seed: Optional[int] = None,
) -> Dict:
    """
    Process a single trial

    Args:
        trial: Trial data dictionary
        client: VLLMClient instance
        model_name: Name of model being used
        seed: Optional random seed for API calls

    Returns:
        Trial data with response added
    """
    # Get model config
    model_config = get_model_config(model_name)

    # Construct prompt using shared library
    prompt = construct_study2_exp1_prompt(trial, model_name)

    # Query model with model-specific parameters
    # Use max_tokens_structured for constrained format responses
    try:
        response = client.generate_text(
            prompt,
            temperature=model_config.temperature_text,
            max_tokens=model_config.max_tokens_structured,
            stop=model_config.stop_tokens,
            seed=seed,
        )
        response_text = response.text
    except Exception as e:
        logger.error(f"Error querying model for trial: {e}")
        response_text = f"ERROR: {str(e)}"

    # Add response and metadata
    result = trial.copy()
    result['response'] = response_text
    result['model_name'] = model_name
    result['timestamp'] = datetime.now().isoformat()

    return result


def run_experiment(
    input_file: Path,
    output_file: Path,
    endpoint: str,
    model_name: str,
    limit: int = None,
    replication_context: Optional[Dict] = None,
):
    """
    Run Study 2 Experiment 1

    Args:
        input_file: Path to study2.csv
        output_file: Path to save results
        endpoint: vLLM server endpoint URL
        model_name: Name of model being queried
        limit: Optional limit on number of trials (for testing)
        replication_context: Optional replication context from initialize_replication()
    """
    logger.info(f"Starting Study 2 Experiment 1")
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Endpoint: {endpoint}")
    logger.info(f"Model: {model_name}")

    if replication_context:
        logger.info(f"Replication ID: {replication_context['replication_id']}")
        logger.info(f"Seed: {replication_context['seed']}")
        logger.info(f"Shuffle: {replication_context['shuffle']}")

    # Load input data
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} trials")

    # Apply limit if specified
    if limit:
        df = df.head(limit)
        logger.info(f"Limited to {len(df)} trials")

    # Shuffle trials if replication context requests it
    if replication_context and replication_context['shuffle']:
        df = shuffle_trials(df, replication_context['seed'])
        logger.info(f"Shuffled trials with seed {replication_context['seed']}")

    # Initialize client
    config = ExperimentConfig()
    client = VLLMClient(base_url=endpoint, config=config)

    # Process trials
    results = []
    trial_seed = replication_context['seed'] if replication_context else None

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing trials"):
        trial = row.to_dict()
        result = process_trial(trial, client, model_name, seed=trial_seed)
        result['trial_order_in_replication'] = idx
        results.append(result)

        # Log progress every 100 trials
        if (idx + 1) % 100 == 0:
            logger.info(f"Completed {idx + 1}/{len(df)} trials")

    # Add replication metadata if context exists
    if replication_context:
        results = add_replication_metadata(
            results,
            replication_context,
            model_name,
            "study2_exp1"
        )
        logger.info(f"Added replication metadata (seed={replication_context['seed']})")

    # Save output as JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved {len(results)} results to {output_file}")

    # Validate output
    logger.info("Validating output...")
    is_valid = validate_study2_exp1(output_file, expected_rows=len(df))

    if is_valid:
        logger.info("✅ Validation passed!")
    else:
        logger.warning("⚠️  Validation failed - check output file")

    # Close client
    client.close()

    logger.info("Experiment complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Study 2 Experiment 1: Politeness Judgments - Raw Text Responses'
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

    # Add replication arguments
    add_replication_args(parser)

    args = parser.parse_args()

    # Initialize replication context (seed offset=1000 for study2_exp1)
    replication_context = initialize_replication(args, default_seed_offset=1000)

    run_experiment(
        input_file=args.input,
        output_file=args.output,
        endpoint=args.endpoint,
        model_name=args.model_name,
        limit=args.limit,
        replication_context=replication_context,
    )


if __name__ == '__main__':
    main()
