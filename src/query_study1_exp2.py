#!/usr/bin/env python3
"""
Study 1 Experiment 2: Knowledge Attribution - Probability Distributions

This is the KEY INNOVATION of the Grace Project!

For each trial in study1.csv, generates 5 separate queries:
1-4. State queries: "What percentage probability do you assign that exactly {0,1,2,3} items have the property?"
5. Knowledge query: "Do you think X knows exactly how many items have the property?"

For each query, extracts probability distribution from logprobs.

Output: 68-column CSV with:
- 9 original columns (8 from study1.csv + model_name)
- 44 probability columns (state{0-3}_prob_{0,10,20,...,100})
- 12 summary statistics (state{0-3}_{mean,std,entropy})
- 3 knowledge columns (prob_knowledge_yes, prob_knowledge_no, entropy_knowledge)
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils.api_client import VLLMClient, compute_distribution_stats
from utils.config import ExperimentConfig, Study1Config
from utils.validation import validate_study1_exp2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_object_name(story_shortname: str) -> str:
    """
    Extract object name from story shortname

    Examples:
    - "exams" -> "exams"
    - "letters" -> "letters"

    Args:
        story_shortname: Story identifier

    Returns:
        Object name (plural)
    """
    # For Study 1, the shortname IS the object name
    return story_shortname


def extract_property(story_setup: str, story_shortname: str) -> str:
    """
    Extract the property being measured from story setup

    Examples:
    - exams -> "passing grades"
    - letters -> "checks inside"

    Args:
        story_setup: Story setup text
        story_shortname: Story identifier

    Returns:
        Property description
    """
    # Simple heuristic: look for key phrases
    if "exam" in story_shortname.lower():
        return "passing grades"
    elif "letter" in story_shortname.lower():
        return "checks inside"
    elif "egg" in story_shortname.lower():
        return "cracks"
    elif "paper" in story_shortname.lower():
        return "citations"
    else:
        # Generic fallback
        return "the property"


def construct_state_query_prompt(trial: Dict, state: int) -> str:
    """
    Construct prompt for state probability query

    Format:
    "Given the scenario below, what percentage probability do you assign
    that exactly {state} of the 3 {objects} have {property}?

    {scenario}

    Please respond with a percentage (0%, 10%, 20%, ..., 100%)."

    Args:
        trial: Trial data dictionary
        state: State value (0, 1, 2, or 3)

    Returns:
        Formatted prompt
    """
    # Extract components
    setup = trial['story_setup'].replace('<br>', '\n').strip()
    speach = trial['speach'].strip()
    object_name = extract_object_name(trial['story_shortname'])
    property_name = extract_property(trial['story_setup'], trial['story_shortname'])

    prompt = f"""Given the scenario below, what percentage probability do you assign that exactly {state} of the 3 {object_name} have {property_name}?

Scenario:
{setup}

{speach}

Please respond with a single percentage (0%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, or 100%)."""

    return prompt


def construct_knowledge_query_prompt(trial: Dict) -> str:
    """
    Construct prompt for knowledge question

    Uses the knowledgeQ from the trial data.

    Args:
        trial: Trial data dictionary

    Returns:
        Formatted prompt
    """
    setup = trial['story_setup'].replace('<br>', '\n').strip()
    speach = trial['speach'].strip()
    knowledge_q = trial['knowledgeQ'].strip()

    prompt = f"""{setup}

{speach}

{knowledge_q}

Please respond with "yes" or "no"."""

    return prompt


def process_trial(
    trial: Dict,
    client: VLLMClient,
    model_name: str,
) -> Dict:
    """
    Process a single trial with 5 queries

    Args:
        trial: Trial data dictionary
        client: VLLMClient instance
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
    pct_values = list(range(0, 101, 10))
    pct_tokens = [f"{p}%" for p in pct_values]

    # Query 1-4: State probability distributions
    for state in range(4):
        prompt = construct_state_query_prompt(trial, state)

        try:
            probs, _ = client.extract_token_probabilities(prompt, pct_tokens)

            # Store probabilities
            for pct in pct_values:
                token = f"{pct}%"
                result[f'state{state}_prob_{pct}'] = probs.get(token, 0.0)

            # Compute summary statistics
            prob_values = [probs.get(f"{p}%", 0.0) for p in pct_values]
            stats = compute_distribution_stats(
                {f"{p}%": prob_values[i] for i, p in enumerate(pct_values)},
                pct_values
            )

            result[f'state{state}_mean'] = stats['mean']
            result[f'state{state}_std'] = stats['std']
            result[f'state{state}_entropy'] = stats['entropy']

        except Exception as e:
            logger.error(f"Error querying state {state} for trial: {e}")
            # Fill with uniform distribution as fallback
            for pct in pct_values:
                result[f'state{state}_prob_{pct}'] = 1.0 / len(pct_values)
            result[f'state{state}_mean'] = 50.0
            result[f'state{state}_std'] = 0.0
            result[f'state{state}_entropy'] = np.log2(len(pct_values))

    # Query 5: Knowledge question (yes/no)
    prompt = construct_knowledge_query_prompt(trial)

    try:
        knowledge_probs = client.extract_binary_probabilities(prompt, ["yes", "no"])

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
    endpoint: str,
    model_name: str,
    limit: int = None,
):
    """
    Run Study 1 Experiment 2

    Args:
        input_file: Path to study1.csv
        output_file: Path to save results
        endpoint: vLLM server endpoint URL
        model_name: Name of model being queried
        limit: Optional limit on number of trials (for testing)
    """
    logger.info(f"Starting Study 1 Experiment 2 (Probability Distributions)")
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

    total_queries = len(df) * 5
    logger.info(f"Total queries to make: {total_queries} (5 per trial)")

    # Initialize client
    config = ExperimentConfig()
    client = VLLMClient(base_url=endpoint, config=config)

    # Process trials
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing trials"):
        trial = row.to_dict()
        result = process_trial(trial, client, model_name)
        results.append(result)

        # Log progress every 10 trials
        if (idx + 1) % 10 == 0:
            queries_completed = (idx + 1) * 5
            logger.info(f"Completed {idx + 1}/{len(df)} trials ({queries_completed}/{total_queries} queries)")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Verify column count
    logger.info(f"Output has {len(results_df.columns)} columns (expected 68)")

    # Save output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)
    logger.info(f"Saved results to {output_file}")

    # Validate output
    logger.info("Validating output...")
    is_valid = validate_study1_exp2(output_file, expected_rows=len(df))

    if is_valid:
        logger.info("✅ Validation passed!")
    else:
        logger.warning("⚠️  Validation failed - check output file")

    # Close client
    client.close()

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
