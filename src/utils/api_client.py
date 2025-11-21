"""
vLLM API Client for Grace Project

Provides a wrapper around vLLM's OpenAI-compatible API with:
- Retry logic and timeout handling
- Logprob extraction for probability distributions
- Temperature control for different experiment types
"""

import time
import logging
import uuid
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import httpx
import numpy as np
from scipy.special import softmax

from .config import ExperimentConfig

logger = logging.getLogger(__name__)


@dataclass
class CompletionResponse:
    """Response from vLLM completion"""
    text: str
    logprobs: Optional[Dict] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict] = None
    reasoning_trace: Optional[str] = None  # For thinking models (o1, o3, etc.)
    result_id: Optional[str] = None  # Unique ID linking response to reasoning trace


class VLLMClient:
    """
    Client for vLLM OpenAI-compatible API

    Supports both text generation and probability extraction via logprobs.
    """

    def __init__(
        self,
        base_url: str,
        config: Optional[ExperimentConfig] = None,
        timeout: int = 120,
    ):
        """
        Initialize vLLM client

        Args:
            base_url: Base URL of vLLM server (e.g., "http://localhost:8000")
            config: Experiment configuration (uses default if None)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.config = config or ExperimentConfig()
        self.timeout = timeout

        # HTTP client with timeout
        self.client = httpx.Client(timeout=timeout)

        logger.info(f"Initialized VLLMClient with base_url={base_url}")

    def _make_request(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        logprobs: Optional[int] = None,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> Dict:
        """
        Make request to vLLM API with retry logic

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            logprobs: Number of logprobs to return (None for no logprobs)
            stop: Stop sequences
            seed: Optional random seed for reproducible sampling

        Returns:
            API response dictionary
        """
        endpoint = f"{self.base_url}/v1/completions"

        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if logprobs is not None:
            payload["logprobs"] = logprobs

        if stop is not None:
            payload["stop"] = stop

        if seed is not None:
            payload["seed"] = seed

        # Infinite retry logic - never give up on connection errors
        # This allows the script to wait indefinitely for port-forward reconnection
        attempt = 0

        while True:
            try:
                response = self.client.post(endpoint, json=payload)
                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                # HTTP errors (4xx, 5xx) are not retried - these are real errors
                logger.error(f"HTTP error: {e}")
                raise

            except httpx.RequestError as e:
                # Connection errors are common with port-forward drops
                # We retry these indefinitely
                if attempt == 0:
                    logger.warning(f"Connection lost to {self.base_url}")
                    logger.warning("Waiting for port-forward reconnection...")
                    logger.warning("Script will continue retrying - it will not give up!")

                # Use exponential backoff with cap
                # Cap the exponent calculation to prevent overflow
                delay = min(self.config.retry_delay_seconds * (1.5 ** min(attempt, 20)), 60)

                logger.info(f"Connection attempt {attempt + 1} failed, retrying in {delay:.1f}s...")
                time.sleep(delay)
                attempt += 1
                # Continue forever - no break condition

    def generate_text(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
        extract_reasoning: bool = True,
        reasoning_start_token: str = "<think>",
        reasoning_end_token: str = "</think>",
    ) -> CompletionResponse:
        """
        Generate text completion (Experiment 1 - raw text responses)

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (default: 0.7 from config)
            max_tokens: Maximum tokens (default: 500 from config)
            stop: Stop sequences
            seed: Optional random seed for reproducible sampling
            extract_reasoning: Whether to extract reasoning traces from text
            reasoning_start_token: Token marking start of reasoning (default: <think>)
            reasoning_end_token: Token marking end of reasoning (default: </think>)

        Returns:
            CompletionResponse with generated text and extracted reasoning trace
        """
        temp = temperature if temperature is not None else self.config.temp_text_generation
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens_text

        response = self._make_request(
            prompt=prompt,
            temperature=temp,
            max_tokens=tokens,
            logprobs=None,
            stop=stop,
            seed=seed,
        )

        # Extract response
        choice = response['choices'][0]
        generated_text = choice['text'].strip()

        # Generate unique result ID
        result_id = str(uuid.uuid4())

        # Extract reasoning trace if available (for thinking models)
        # Try separate field first (some APIs return reasoning separately)
        reasoning_trace = None
        if 'reasoning' in choice:
            reasoning_trace = choice['reasoning']
        elif 'thought' in choice:
            reasoning_trace = choice['thought']
        elif 'chain_of_thought' in choice:
            reasoning_trace = choice['chain_of_thought']

        # If no separate field and extract_reasoning is True, parse from text
        if reasoning_trace is None and extract_reasoning:
            reasoning_trace = self._extract_reasoning_from_text(
                generated_text, reasoning_start_token, reasoning_end_token
            )

        return CompletionResponse(
            text=generated_text,
            finish_reason=choice.get('finish_reason'),
            usage=response.get('usage'),
            reasoning_trace=reasoning_trace,
            result_id=result_id,
        )

    def _extract_reasoning_from_text(
        self, text: str, start_token: str, end_token: str
    ) -> Optional[str]:
        """
        Extract reasoning trace from text with reasoning tags

        For reasoning models like DeepSeek-R1, reasoning is embedded in the
        generated text within special tags (e.g., <think>...</think>).

        Args:
            text: Generated text
            start_token: Token marking start of reasoning
            end_token: Token marking end of reasoning

        Returns:
            Extracted reasoning trace or None if not found
        """
        pattern = re.escape(start_token) + r'(.*?)' + re.escape(end_token)
        match = re.search(pattern, text, re.DOTALL)

        if match:
            reasoning = match.group(1).strip()
            logger.debug(f"Extracted reasoning trace ({len(reasoning)} chars)")
            return reasoning

        return None

    def extract_token_probabilities(
        self,
        prompt: str,
        tokens: List[str],
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[Dict[str, float], str]:
        """
        Extract probability distribution over specified token sequences

        NEW APPROACH: Get logprobs for the first token in each option by examining
        what the model would generate. We look at logprobs for the top-k tokens
        and match against our target sequences.

        Args:
            prompt: Input prompt
            tokens: List of token sequences (e.g., ["0%", "10%", ..., "100%"])
            temperature: Sampling temperature (default: 1.0)
            seed: Optional random seed for reproducible sampling

        Returns:
            Tuple of (probability_dict, generated_text)
        """
        temp = temperature if temperature is not None else self.config.temp_probabilities

        # Generate 1 token to get logprobs for what comes next
        response = self._make_request(
            prompt=prompt,
            temperature=temp,
            max_tokens=1,
            logprobs=50,  # Request more logprobs to catch all our options
            seed=seed,
        )

        choice = response['choices'][0]
        generated_text = choice['text'].strip()

        if 'logprobs' not in choice or not choice['logprobs']:
            raise ValueError("No logprobs returned from API")

        # Get first token's logprobs
        top_logprobs = choice['logprobs'].get('top_logprobs', [{}])[0]

        # For each option, find its logprob in the returned logprobs
        logprobs_dict = {}
        for option in tokens:
            logprob = self._find_option_logprob(option, top_logprobs)
            logprobs_dict[option] = logprob

        # Normalize to get probability distribution
        probs = self._normalize_logprobs(logprobs_dict)

        return probs, generated_text

    def _find_option_logprob(self, option: str, top_logprobs: Dict[str, float]) -> float:
        """
        Find logprob for an option in the top_logprobs dict

        Tries multiple variations to handle tokenization differences:
        - Exact match: "10%"
        - With space: " 10%"
        - First token only: "10" (if "10%" isn't found)
        - With space + first token: " 10"

        Args:
            option: Target sequence (e.g., "10%")
            top_logprobs: Dict of {token: logprob}

        Returns:
            Log probability for this option
        """
        # Try exact match
        if option in top_logprobs:
            return top_logprobs[option]

        # Try with leading space
        if f" {option}" in top_logprobs:
            return top_logprobs[f" {option}"]

        # If option contains %, try just the number part
        if '%' in option:
            number_part = option.replace('%', '')

            # Try number without space
            if number_part in top_logprobs:
                return top_logprobs[number_part]

            # Try number with space
            if f" {number_part}" in top_logprobs:
                return top_logprobs[f" {number_part}"]

        # Try without leading/trailing whitespace
        option_stripped = option.strip()
        if option_stripped in top_logprobs:
            return top_logprobs[option_stripped]

        if f" {option_stripped}" in top_logprobs:
            return top_logprobs[f" {option_stripped}"]

        # Not found - assign very low probability
        logger.warning(f"Option '{option}' not found in top_logprobs. Available tokens: {list(top_logprobs.keys())[:10]}")
        return -20.0

    def _normalize_logprobs(self, logprobs_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Convert logprobs to normalized probability distribution

        Args:
            logprobs_dict: Dictionary mapping tokens to log probabilities

        Returns:
            Dictionary mapping tokens to probabilities (sum = 1.0)
        """
        tokens = list(logprobs_dict.keys())
        logprobs = np.array([logprobs_dict[t] for t in tokens])

        # Apply softmax to get probabilities
        probs = softmax(logprobs)

        # Create normalized probability dictionary
        prob_dict = {token: float(prob) for token, prob in zip(tokens, probs)}

        return prob_dict

    def extract_binary_probabilities(
        self,
        prompt: str,
        tokens: List[str],
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Extract binary probability distribution (e.g., "yes"/"no")

        Used for Study 1 knowledge questions: "Does Mark know how many exams passed?"

        Args:
            prompt: Input prompt
            tokens: Two tokens to extract probabilities for (e.g., ["yes", "no"])
            temperature: Sampling temperature
            seed: Optional random seed for reproducible sampling

        Returns:
            Probability distribution over tokens
        """
        if len(tokens) != 2:
            raise ValueError("Binary extraction requires exactly 2 tokens")

        probs, _ = self.extract_token_probabilities(prompt, tokens, temperature, seed)
        return probs

    def close(self):
        """Close HTTP client"""
        self.client.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


def compute_distribution_stats(probs: Dict[str, float], values: List[float]) -> Dict[str, float]:
    """
    Compute summary statistics for probability distribution

    Args:
        probs: Probability distribution (token -> probability)
        values: Numeric values corresponding to each token

    Returns:
        Dictionary with mean, std, entropy
    """
    tokens = list(probs.keys())
    prob_values = np.array([probs[t] for t in tokens])
    numeric_values = np.array(values)

    # Mean (expected value)
    mean = np.sum(prob_values * numeric_values)

    # Standard deviation
    variance = np.sum(prob_values * (numeric_values - mean) ** 2)
    std = np.sqrt(variance)

    # Entropy (in bits)
    # H = -sum(p * log2(p)) for p > 0
    entropy = 0.0
    for p in prob_values:
        if p > 1e-10:  # Avoid log(0)
            entropy -= p * np.log2(p)

    return {
        'mean': float(mean),
        'std': float(std),
        'entropy': float(entropy),
    }
