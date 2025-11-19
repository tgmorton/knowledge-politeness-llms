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

Expected output: CSV with original columns + response, model_name, timestamp
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict
import pandas as pd
from tqdm import tqdm

from utils.api_client import VLLMClient
from utils.config import ExperimentConfig
from utils.validation import validate_study2_exp1

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def construct_prompt(trial: Dict) -> str:
    """
    Construct prompt for Study 2 Experiment 1

    Format:
    Present the scenario, utterance, goal, and state
    Ask for a judgment of the response

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

How would you characterize {trial['SP_Name']}'s response? Was it appropriate given the context?

Please provide your judgment and reasoning."""

    return prompt


def process_trial(
    trial: Dict,
    client: VLLMClient,
    model_name: str,
) -> Dict:
    """
    Process a single trial

    Args:
        trial: Trial data dictionary
        client: VLLMClient instance
        model_name: Name of model being used

    Returns:
        Trial data with response added
    """
    # Construct prompt
    prompt = construct_prompt(trial)

    # Query model
    try:
        response = client.generate_text(prompt)
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
):
    """
    Run Study 2 Experiment 1

    Args:
        input_file: Path to study2.csv
        output_file: Path to save results
        endpoint: vLLM server endpoint URL
        model_name: Name of model being queried
        limit: Optional limit on number of trials (for testing)
    """
    logger.info(f"Starting Study 2 Experiment 1")
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
            logger.info(f"Completed {idx + 1}/{len(df)} trials")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)
    logger.info(f"Saved results to {output_file}")

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
