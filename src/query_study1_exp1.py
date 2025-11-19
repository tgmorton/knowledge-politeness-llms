#!/usr/bin/env python3
"""
Study 1 Experiment 1: Knowledge Attribution - Raw Text Responses

Queries models for reasoning about knowledge attribution based on observed evidence.

For each trial in study1.csv:
- Present the scenario setup
- Present the information (speech)
- Ask how many items have the property
- Save raw text response

Expected output: CSV with original columns + response, model_name, timestamp
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
from tqdm import tqdm

from utils.api_client import VLLMClient
from utils.config import ExperimentConfig
from utils.validation import validate_study1_exp1
from utils.reasoning_trace import ReasoningTraceWriter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def construct_prompt(trial: Dict) -> str:
    """
    Construct prompt for Study 1 Experiment 1

    Format:
    1. Present the scenario setup
    2. Ask the prior question
    3. Present the speech (observation)
    4. Ask the posterior question

    Args:
        trial: Dictionary with trial data from study1.csv

    Returns:
        Formatted prompt string
    """
    # Clean up HTML tags
    setup = trial['story_setup'].replace('<br>', '\n').strip()

    # Construct the prompt
    prompt = f"""{setup}

{trial['priorQ']}

{trial['speach']}

{trial['speachQ']}

Please provide your reasoning and answer."""

    return prompt


def process_trial(
    trial: Dict,
    client: VLLMClient,
    model_name: str,
):
    """
    Process a single trial

    Args:
        trial: Trial data dictionary
        client: VLLMClient instance
        model_name: Name of model being used

    Returns:
        Tuple of (result_dict, prompt, reasoning_info)
    """
    # Construct prompt
    prompt = construct_prompt(trial)

    # Query model
    try:
        response = client.generate_text(prompt)
        response_text = response.text
        result_id = response.result_id
        reasoning_trace = response.reasoning_trace
    except Exception as e:
        logger.error(f"Error querying model for trial: {e}")
        response_text = f"ERROR: {str(e)}"
        result_id = None
        reasoning_trace = None

    # Add response and metadata
    result = trial.copy()
    result['response'] = response_text
    result['result_id'] = result_id
    result['model_name'] = model_name
    result['timestamp'] = datetime.now().isoformat()

    # Reasoning trace info for separate file
    reasoning_info = {
        'result_id': result_id,
        'reasoning_trace': reasoning_trace,
        'prompt': prompt,
        'response': response_text,
    }

    return result, reasoning_info


def run_experiment(
    input_file: Path,
    output_file: Path,
    endpoint: str,
    model_name: str,
    reasoning_output: Optional[Path] = None,
    limit: int = None,
):
    """
    Run Study 1 Experiment 1

    Args:
        input_file: Path to study1.csv
        output_file: Path to save results
        endpoint: vLLM server endpoint URL
        model_name: Name of model being queried
        reasoning_output: Optional path to save reasoning traces (JSONL)
        limit: Optional limit on number of trials (for testing)
    """
    logger.info(f"Starting Study 1 Experiment 1")
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

    # Initialize reasoning trace writer if requested
    reasoning_writer = None
    if reasoning_output:
        reasoning_writer = ReasoningTraceWriter(reasoning_output)
        logger.info(f"Reasoning traces will be saved to: {reasoning_output}")

    # Process trials
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing trials"):
        trial = row.to_dict()
        result, reasoning_info = process_trial(trial, client, model_name)
        results.append(result)

        # Write reasoning trace if writer is active and trace exists
        if reasoning_writer and reasoning_info['reasoning_trace']:
            reasoning_writer.write_trace(
                result_id=reasoning_info['result_id'],
                trial_index=idx,
                prompt=reasoning_info['prompt'],
                reasoning_trace=reasoning_info['reasoning_trace'],
                response=reasoning_info['response'],
                model_name=model_name,
            )

        # Log progress every 10 trials
        if (idx + 1) % 10 == 0:
            logger.info(f"Completed {idx + 1}/{len(df)} trials")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)
    logger.info(f"Saved results to {output_file}")

    # Validate output
    logger.info("Validating output...")
    is_valid = validate_study1_exp1(output_file, expected_rows=len(df))

    if is_valid:
        logger.info("✅ Validation passed!")
    else:
        logger.warning("⚠️  Validation failed - check output file")

    # Close reasoning writer
    if reasoning_writer:
        reasoning_writer.close()

    # Close client
    client.close()

    logger.info("Experiment complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Study 1 Experiment 1: Knowledge Attribution - Raw Text Responses'
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
        '--reasoning-output',
        type=Path,
        default=None,
        help='Path to save reasoning traces (JSONL format, optional)'
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
        reasoning_output=args.reasoning_output,
        limit=args.limit,
    )


if __name__ == '__main__':
    main()
