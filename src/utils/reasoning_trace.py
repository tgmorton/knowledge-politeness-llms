"""
Reasoning Trace Utilities

Handles saving and loading reasoning traces from thinking models.
Traces are saved as JSONL (JSON Lines) for efficient streaming.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ReasoningTraceWriter:
    """
    Writes reasoning traces to JSONL file

    Each line is a JSON object with:
    - result_id: Links to main results CSV
    - trial_index: Index of trial in experiment
    - prompt: Input prompt
    - reasoning_trace: Model's reasoning process
    - response: Final response text
    - timestamp: When generated
    - model_name: Model that generated it
    """

    def __init__(self, output_path: Path):
        """
        Initialize reasoning trace writer

        Args:
            output_path: Path to JSONL file
        """
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open file in append mode
        self.file = open(self.output_path, 'a', encoding='utf-8')
        logger.info(f"Reasoning trace writer initialized: {output_path}")

    def write_trace(
        self,
        result_id: str,
        trial_index: int,
        prompt: str,
        reasoning_trace: Optional[str],
        response: str,
        model_name: str,
        metadata: Optional[Dict] = None,
    ):
        """
        Write a reasoning trace entry

        Args:
            result_id: Unique ID linking to results CSV
            trial_index: Index of trial (row number)
            prompt: Input prompt
            reasoning_trace: Model's reasoning (None if not available)
            response: Final response text
            model_name: Name of model
            metadata: Additional metadata (optional)
        """
        entry = {
            'result_id': result_id,
            'trial_index': trial_index,
            'prompt': prompt,
            'reasoning_trace': reasoning_trace,
            'response': response,
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
        }

        if metadata:
            entry['metadata'] = metadata

        # Write as single line JSON
        json_line = json.dumps(entry, ensure_ascii=False)
        self.file.write(json_line + '\n')
        self.file.flush()  # Ensure written immediately

    def close(self):
        """Close the file"""
        if hasattr(self, 'file') and self.file:
            self.file.close()
            logger.info(f"Reasoning trace writer closed: {self.output_path}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


def load_reasoning_traces(input_path: Path):
    """
    Load reasoning traces from JSONL file

    Args:
        input_path: Path to JSONL file

    Yields:
        Dictionary for each reasoning trace entry
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def get_trace_by_result_id(input_path: Path, result_id: str) -> Optional[Dict]:
    """
    Find reasoning trace by result ID

    Args:
        input_path: Path to JSONL file
        result_id: Result ID to search for

    Returns:
        Reasoning trace entry or None if not found
    """
    for entry in load_reasoning_traces(input_path):
        if entry.get('result_id') == result_id:
            return entry
    return None
