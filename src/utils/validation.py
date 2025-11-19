"""
Output validation utilities for Grace Project

Validates experimental outputs for:
- Correct schema (row counts, column names)
- Probability distributions (sum to 1.0)
- Missing data
- Data quality
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

from .config import Study1Config, Study2Config

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails"""
    pass


def validate_output_schema(
    output_file: Path,
    expected_columns: List[str],
    expected_rows: Optional[int] = None,
) -> Tuple[bool, List[str]]:
    """
    Validate output file schema

    Args:
        output_file: Path to output CSV
        expected_columns: List of expected column names
        expected_rows: Expected number of rows (excluding header)

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Check file exists
    if not output_file.exists():
        errors.append(f"Output file does not exist: {output_file}")
        return False, errors

    # Load CSV
    try:
        df = pd.read_csv(output_file)
    except Exception as e:
        errors.append(f"Failed to read CSV: {e}")
        return False, errors

    # Check row count
    if expected_rows is not None and len(df) != expected_rows:
        errors.append(f"Expected {expected_rows} rows, got {len(df)}")

    # Check columns
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    extra_cols = set(df.columns) - set(expected_columns)
    if extra_cols:
        errors.append(f"Unexpected columns: {extra_cols}")

    # Check for missing data
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) > 0:
        errors.append(f"Columns with missing data: {cols_with_nulls.to_dict()}")

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_probabilities(
    df: pd.DataFrame,
    prob_columns: List[str],
    tolerance: float = 0.01,
) -> Tuple[bool, List[str]]:
    """
    Validate probability distributions sum to 1.0

    Args:
        df: DataFrame with probability columns
        prob_columns: List of probability column names
        tolerance: Acceptable deviation from 1.0

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Check all prob columns exist
    missing_cols = set(prob_columns) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing probability columns: {missing_cols}")
        return False, errors

    # Check each row sums to ~1.0
    prob_sums = df[prob_columns].sum(axis=1)
    invalid_rows = np.where(np.abs(prob_sums - 1.0) > tolerance)[0]

    if len(invalid_rows) > 0:
        for row_idx in invalid_rows[:5]:  # Show first 5 errors
            actual_sum = prob_sums.iloc[row_idx]
            errors.append(
                f"Row {row_idx}: probabilities sum to {actual_sum:.6f} "
                f"(expected 1.0 ± {tolerance})"
            )

        if len(invalid_rows) > 5:
            errors.append(f"... and {len(invalid_rows) - 5} more rows with invalid sums")

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_study1_exp1(output_file: Path, expected_rows: int = 300) -> Tuple[bool, List[str]]:
    """
    Validate Study 1 Experiment 1 output (raw text responses)

    Expected columns:
    - Original 8 columns from study1.csv
    - response: Generated text response
    - model_name: Name of model used
    - timestamp: When query was made

    Args:
        output_file: Path to output CSV
        expected_rows: Expected number of trials

    Returns:
        Tuple of (is_valid, error_messages)
    """
    config = Study1Config()
    expected_columns = config.input_columns + ['response', 'model_name', 'timestamp']

    return validate_output_schema(output_file, expected_columns, expected_rows)


def validate_study1_exp2(output_file: Path, expected_rows: int = 300) -> Tuple[bool, List[str]]:
    """
    Validate Study 1 Experiment 2 output (probability distributions)

    Expected: 68 columns total
    - 9 original columns (8 from study1.csv + model_name)
    - 44 probability columns (state{0-3}_prob_{0,10,20,...,100})
    - 12 summary statistics (state{0-3}_{mean,std,entropy})
    - 3 knowledge columns (prob_knowledge_yes, prob_knowledge_no, entropy_knowledge)

    Args:
        output_file: Path to output CSV
        expected_rows: Expected number of trials

    Returns:
        Tuple of (is_valid, error_messages)
    """
    config = Study1Config()
    errors = []

    # Build expected columns
    original_cols = config.input_columns + ['model_name']

    # Probability columns for 4 states (0, 1, 2, 3 exams passed)
    prob_cols = []
    for state in range(4):
        for pct in range(0, 101, 10):
            prob_cols.append(f'state{state}_prob_{pct}')

    # Summary statistics
    stats_cols = []
    for state in range(4):
        stats_cols.extend([
            f'state{state}_mean',
            f'state{state}_std',
            f'state{state}_entropy',
        ])

    # Knowledge columns
    knowledge_cols = [
        'prob_knowledge_yes',
        'prob_knowledge_no',
        'entropy_knowledge',
    ]

    expected_columns = original_cols + prob_cols + stats_cols + knowledge_cols

    # Validate schema
    is_valid, schema_errors = validate_output_schema(output_file, expected_columns, expected_rows)
    errors.extend(schema_errors)

    # If schema is valid, validate probabilities
    if is_valid:
        df = pd.read_csv(output_file)

        # Validate each state's probability distribution
        for state in range(4):
            state_prob_cols = [f'state{state}_prob_{pct}' for pct in range(0, 101, 10)]
            prob_valid, prob_errors = validate_probabilities(df, state_prob_cols)
            if not prob_valid:
                errors.append(f"State {state} probability validation failed:")
                errors.extend(prob_errors)

        # Validate knowledge probabilities
        knowledge_prob_cols = ['prob_knowledge_yes', 'prob_knowledge_no']
        prob_valid, prob_errors = validate_probabilities(df, knowledge_prob_cols)
        if not prob_valid:
            errors.append("Knowledge probability validation failed:")
            errors.extend(prob_errors)

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_study2_exp1(output_file: Path, expected_rows: int = 2424) -> Tuple[bool, List[str]]:
    """
    Validate Study 2 Experiment 1 output (raw politeness responses)

    Expected columns:
    - Original 5 columns from study2.csv
    - response: Generated text response
    - model_name: Name of model used
    - timestamp: When query was made

    Args:
        output_file: Path to output CSV
        expected_rows: Expected number of trials

    Returns:
        Tuple of (is_valid, error_messages)
    """
    config = Study2Config()
    expected_columns = config.input_columns + ['response', 'model_name', 'timestamp']

    return validate_output_schema(output_file, expected_columns, expected_rows)


def validate_study2_exp2(output_file: Path, expected_rows: int = 2424) -> Tuple[bool, List[str]]:
    """
    Validate Study 2 Experiment 2 output (politeness probability distributions)

    Args:
        output_file: Path to output CSV
        expected_rows: Expected number of trials

    Returns:
        Tuple of (is_valid, error_messages)
    """
    # TODO: Implement based on Study 2 Experiment 2 design
    # This depends on the specific probability extraction approach for politeness
    logger.warning("Study 2 Experiment 2 validation not yet implemented")
    return True, []


def print_validation_report(
    output_file: Path,
    is_valid: bool,
    errors: List[str],
):
    """
    Print validation report to console

    Args:
        output_file: Path to validated file
        is_valid: Whether validation passed
        errors: List of error messages
    """
    print(f"\n{'='*70}")
    print(f"Validation Report: {output_file.name}")
    print(f"{'='*70}")

    if is_valid:
        print("✅ VALIDATION PASSED")
        print(f"\nFile: {output_file}")
        df = pd.read_csv(output_file)
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
    else:
        print("❌ VALIDATION FAILED")
        print(f"\nFile: {output_file}")
        print(f"\nErrors ({len(errors)}):")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")

    print(f"{'='*70}\n")


def validate_file(
    output_file: Path,
    study: int,
    experiment: int,
    expected_rows: Optional[int] = None,
    verbose: bool = True,
) -> bool:
    """
    Validate output file based on study and experiment type

    Args:
        output_file: Path to output CSV
        study: Study number (1 or 2)
        experiment: Experiment number (1 or 2)
        expected_rows: Expected number of rows (None for auto-detect)
        verbose: Print validation report

    Returns:
        True if validation passed, False otherwise
    """
    # Auto-detect expected rows
    if expected_rows is None:
        expected_rows = 300 if study == 1 else 2424

    # Select validation function
    if study == 1 and experiment == 1:
        is_valid, errors = validate_study1_exp1(output_file, expected_rows)
    elif study == 1 and experiment == 2:
        is_valid, errors = validate_study1_exp2(output_file, expected_rows)
    elif study == 2 and experiment == 1:
        is_valid, errors = validate_study2_exp1(output_file, expected_rows)
    elif study == 2 and experiment == 2:
        is_valid, errors = validate_study2_exp2(output_file, expected_rows)
    else:
        raise ValueError(f"Invalid study/experiment combination: {study}/{experiment}")

    if verbose:
        print_validation_report(output_file, is_valid, errors)

    return is_valid
