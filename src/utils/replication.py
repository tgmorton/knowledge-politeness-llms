"""
Replication support utilities for Grace Project
Handles seeding, trial randomization, and metadata tracking
"""

import argparse
import random
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np


def add_replication_args(parser: argparse.ArgumentParser):
    """Add standard replication arguments to argument parser"""
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: use current time)'
    )
    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='Randomize trial order (recommended for multi-replication studies)'
    )
    parser.add_argument(
        '--replication-id',
        type=int,
        default=None,
        help='Replication index (0-indexed, from JOB_COMPLETION_INDEX)'
    )


def initialize_replication(args, default_seed_offset: int = 0) -> Dict[str, Any]:
    """
    Initialize replication context: set seeds, prepare metadata

    Args:
        args: Parsed command-line arguments (must have seed, shuffle, replication_id)
        default_seed_offset: Offset to add to seed (for different experiments)

    Returns:
        Replication context dictionary with seed, metadata, etc.
    """
    # Determine seed
    if args.seed is not None:
        seed = args.seed + default_seed_offset
    else:
        # Generate from timestamp if not provided
        seed = int(datetime.now().timestamp() * 1000) % (2**31)
        seed += default_seed_offset

    # Set global random seeds
    random.seed(seed)
    np.random.seed(seed)

    # Build context
    context = {
        'seed': seed,
        'base_seed': args.seed,
        'seed_offset': default_seed_offset,
        'shuffle': args.shuffle,
        'replication_id': args.replication_id,
        'timestamp': datetime.now().isoformat(),
    }

    return context


def shuffle_trials(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Shuffle trial order deterministically

    Args:
        df: DataFrame of trials
        seed: Random seed

    Returns:
        Shuffled DataFrame with reset index
    """
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def add_replication_metadata(
    results: List[Dict],
    context: Dict[str, Any],
    model_name: str,
    experiment_name: str
) -> List[Dict]:
    """
    Add replication metadata to each result record

    Args:
        results: List of result dictionaries
        context: Replication context from initialize_replication()
        model_name: Model identifier
        experiment_name: Experiment identifier (e.g., "study1_exp1")

    Returns:
        Results with added metadata fields
    """
    for result in results:
        result['replication_id'] = context['replication_id']
        result['replication_seed'] = context['seed']
        result['trial_order_shuffled'] = context['shuffle']
        result['run_timestamp'] = context['timestamp']
        result['model_name'] = model_name
        result['experiment_name'] = experiment_name

    return results
