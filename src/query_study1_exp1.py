#!/usr/bin/env python3
"""
Study 1 Experiment 1: Knowledge Attribution - Raw Text Responses

Queries models for reasoning about knowledge attribution based on observed evidence.

For each trial in study1.csv:
- Present the scenario setup
- Present the information (speech)
- Ask how many items have the property
- Save raw text response

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
from utils.validation import validate_study1_exp1
from utils.reasoning_trace import ReasoningTraceWriter
from utils.model_config import get_model_config
from utils.prompts import (
    construct_study1_exp1_quantity_prompt,
    construct_study1_exp1_knowledge_prompt,
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
):
    """
    Process a single trial with TWO sequential questions

    Args:
        trial: Trial data dictionary
        client: VLLMClient instance
        model_name: Name of model being used

    Returns:
        Tuple of (result_dict, reasoning_info)
    """
    # Get model config
    model_config = get_model_config(model_name)

    # ===== QUESTION 1: Quantity =====
    prompt_quantity = construct_study1_exp1_quantity_prompt(trial, model_name)

    try:
        response_quantity = client.generate_text(
            prompt_quantity,
            temperature=model_config.temperature_text,
            max_tokens=model_config.max_tokens_text,
            stop=model_config.stop_tokens,
        )
        quantity_text = response_quantity.text.strip()
        quantity_result_id = response_quantity.result_id
        quantity_reasoning = response_quantity.reasoning_trace
    except Exception as e:
        logger.error(f"Error querying model for quantity question: {e}")
        quantity_text = f"ERROR: {str(e)}"
        quantity_result_id = None
        quantity_reasoning = None

    # ===== QUESTION 2: Knowledge =====
    prompt_knowledge = construct_study1_exp1_knowledge_prompt(trial, model_name)

    try:
        response_knowledge = client.generate_text(
            prompt_knowledge,
            temperature=model_config.temperature_text,
            max_tokens=model_config.max_tokens_text,
            stop=model_config.stop_tokens,
        )
        knowledge_text = response_knowledge.text.strip()
        knowledge_result_id = response_knowledge.result_id
        knowledge_reasoning = response_knowledge.reasoning_trace
    except Exception as e:
        logger.error(f"Error querying model for knowledge question: {e}")
        knowledge_text = f"ERROR: {str(e)}"
        knowledge_result_id = None
        knowledge_reasoning = None

    # Add responses and metadata
    result = trial.copy()
    result['response_quantity'] = quantity_text
    result['response_knowledge'] = knowledge_text
    result['result_id_quantity'] = quantity_result_id
    result['result_id_knowledge'] = knowledge_result_id
    result['model_name'] = model_name
    result['timestamp'] = datetime.now().isoformat()

    # Reasoning trace info for separate file (both questions)
    reasoning_info = {
        'quantity': {
            'result_id': quantity_result_id,
            'reasoning_trace': quantity_reasoning,
            'prompt': prompt_quantity,
            'response': quantity_text,
        },
        'knowledge': {
            'result_id': knowledge_result_id,
            'reasoning_trace': knowledge_reasoning,
            'prompt': prompt_knowledge,
            'response': knowledge_text,
        }
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

        # Write reasoning traces if writer is active
        if reasoning_writer:
            # Write quantity question reasoning
            if reasoning_info['quantity']['reasoning_trace']:
                reasoning_writer.write_trace(
                    result_id=reasoning_info['quantity']['result_id'],
                    trial_index=idx,
                    prompt=reasoning_info['quantity']['prompt'],
                    reasoning_trace=reasoning_info['quantity']['reasoning_trace'],
                    response=reasoning_info['quantity']['response'],
                    model_name=model_name,
                    metadata={'question_type': 'quantity'},
                )

            # Write knowledge question reasoning
            if reasoning_info['knowledge']['reasoning_trace']:
                reasoning_writer.write_trace(
                    result_id=reasoning_info['knowledge']['result_id'],
                    trial_index=idx,
                    prompt=reasoning_info['knowledge']['prompt'],
                    reasoning_trace=reasoning_info['knowledge']['reasoning_trace'],
                    response=reasoning_info['knowledge']['response'],
                    model_name=model_name,
                    metadata={'question_type': 'knowledge'},
                )

        # Log progress every 10 trials
        if (idx + 1) % 10 == 0:
            logger.info(f"Completed {idx + 1}/{len(df)} trials")

    # Save output as JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved {len(results)} results to {output_file}")

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
