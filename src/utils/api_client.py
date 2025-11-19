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
    ) -> Dict:
        """
        Make request to vLLM API with retry logic

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            logprobs: Number of logprobs to return (None for no logprobs)
            stop: Stop sequences

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

        # Retry logic
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.post(endpoint, json=payload)
                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error on attempt {attempt + 1}: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay_seconds)
                else:
                    raise

            except httpx.RequestError as e:
                logger.error(f"Request error on attempt {attempt + 1}: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay_seconds)
                else:
                    raise

        raise RuntimeError(f"Failed after {self.config.max_retries} retries")

    def generate_text(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> CompletionResponse:
        """
        Generate text completion (Experiment 1 - raw text responses)

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (default: 0.7 from config)
            max_tokens: Maximum tokens (default: 500 from config)
            stop: Stop sequences

        Returns:
            CompletionResponse with generated text
        """
        temp = temperature if temperature is not None else self.config.temp_text_generation
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens_text

        response = self._make_request(
            prompt=prompt,
            temperature=temp,
            max_tokens=tokens,
            logprobs=None,
            stop=stop,
        )

        # Extract response
        choice = response['choices'][0]

        # Generate unique result ID
        result_id = str(uuid.uuid4())

        # Extract reasoning trace if available (for thinking models)
        # Some models return reasoning in a separate field
        reasoning_trace = None
        if 'reasoning' in choice:
            reasoning_trace = choice['reasoning']
        elif 'thought' in choice:
            reasoning_trace = choice['thought']
        elif 'chain_of_thought' in choice:
            reasoning_trace = choice['chain_of_thought']

        return CompletionResponse(
            text=choice['text'].strip(),
            finish_reason=choice.get('finish_reason'),
            usage=response.get('usage'),
            reasoning_trace=reasoning_trace,
            result_id=result_id,
        )

    def extract_token_probabilities(
        self,
        prompt: str,
        tokens: List[str],
        temperature: Optional[float] = None,
    ) -> Tuple[Dict[str, float], str]:
        """
        Extract probability distribution over specified tokens (Experiment 2)

        This is the KEY method for Study 1 Experiment 2 probability extraction.

        Args:
            prompt: Input prompt (e.g., "What % probability do you assign that exactly 2 exams passed?")
            tokens: List of tokens to extract probabilities for (e.g., ["0%", "10%", ..., "100%"])
            temperature: Sampling temperature (default: 1.0 from config)

        Returns:
            Tuple of (probability_dict, generated_token)
            - probability_dict: Maps each token to its probability
            - generated_token: The token that was actually sampled
        """
        temp = temperature if temperature is not None else self.config.temp_probabilities

        response = self._make_request(
            prompt=prompt,
            temperature=temp,
            max_tokens=1,  # Single token for probability extraction
            logprobs=self.config.logprobs_count,
        )

        # Extract logprobs
        choice = response['choices'][0]
        generated_token = choice['text'].strip()

        # Get top logprobs for first (and only) token
        if 'logprobs' not in choice or not choice['logprobs']:
            raise ValueError("No logprobs returned from API")

        # vLLM returns logprobs as a dict: {token: logprob, ...}
        token_logprobs = choice['logprobs'].get('top_logprobs', [{}])[0]

        # Extract logprobs for requested tokens
        logprobs_dict = {}
        for token in tokens:
            # Try exact match first
            if token in token_logprobs:
                logprobs_dict[token] = token_logprobs[token]
            # Try with leading space (vLLM tokenization)
            elif f" {token}" in token_logprobs:
                logprobs_dict[token] = token_logprobs[f" {token}"]
            else:
                # Token not in top logprobs, assign very low probability
                logprobs_dict[token] = -20.0  # exp(-20) ≈ 2e-9

        # Convert logprobs to probabilities and normalize
        probs = self._normalize_logprobs(logprobs_dict)

        return probs, generated_token

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
    ) -> Dict[str, float]:
        """
        Extract binary probability distribution (e.g., "yes"/"no")

        Used for Study 1 knowledge questions: "Does Mark know how many exams passed?"

        Args:
            prompt: Input prompt
            tokens: Two tokens to extract probabilities for (e.g., ["yes", "no"])
            temperature: Sampling temperature

        Returns:
            Probability distribution over tokens
        """
        if len(tokens) != 2:
            raise ValueError("Binary extraction requires exactly 2 tokens")

        probs, _ = self.extract_token_probabilities(prompt, tokens, temperature)
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
