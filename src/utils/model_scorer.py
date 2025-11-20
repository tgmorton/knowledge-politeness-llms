"""
Direct Model Scoring for Probability Extraction

This module provides direct model access for computing log probabilities
over a constrained set of completions. Used for Experiment 2 (probability
extraction) where we need accurate P(option | prompt) for each option.

Why not use vLLM?
-----------------
vLLM is optimized for text generation, not for scoring pre-defined options.
For probability extraction, we need to:
1. Compute P("0%" | prompt), P("10%" | prompt), ..., P("100%" | prompt)
2. Handle multi-token sequences (e.g., "10%" = "10" + "%")
3. Get exact logprobs for the full sequence

Direct model access gives us:
- Accurate scoring of any completion
- Full control over tokenization
- Simple, transparent implementation

Usage:
    scorer = ModelScorer(model_name="google/gemma-2-2b-it")

    prompt = "What percentage? Answer:"
    options = ["0%", "10%", "20%", ..., "100%"]

    probs = scorer.score_options(prompt, options)
    # Returns: {"0%": 0.05, "10%": 0.15, ..., "100%": 0.03}
"""

import logging
from typing import Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

logger = logging.getLogger(__name__)


class ModelScorer:
    """
    Direct model access for scoring completions

    Loads a model and computes log probabilities for prompt+completion sequences.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: str = None,
        device: str = "auto",
        torch_dtype: str = "auto",
    ):
        """
        Initialize model scorer

        Args:
            model_name: HuggingFace model identifier (e.g., "google/gemma-2-2b-it")
            cache_dir: Directory to cache downloaded models (e.g., "/models/.cache")
            device: Device to use ("cpu", "cuda", "mps", or "auto")
            torch_dtype: Torch dtype ("float16", "float32", "bfloat16", or "auto")
        """
        self.model_name = model_name
        self.cache_dir = cache_dir

        if cache_dir:
            logger.info(f"Loading model: {model_name} (cache: {cache_dir})")
        else:
            logger.info(f"Loading model: {model_name}")

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        logger.info(f"Using device: {device}")

        # Determine dtype
        if torch_dtype == "auto":
            if device == "cuda":
                torch_dtype = torch.float16
            elif device == "mps":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
        else:
            torch_dtype = getattr(torch, torch_dtype)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir
        )
        self.model.eval()  # Set to evaluation mode

        logger.info(f"Model loaded successfully on {device} with dtype {torch_dtype}")

    def score_completion(self, prompt: str, completion: str) -> float:
        """
        Compute log probability of a completion given a prompt

        Computes: log P(completion | prompt)

        This is done by:
        1. Tokenizing prompt + completion
        2. Getting model logits for the full sequence
        3. Extracting logprobs for completion tokens only
        4. Summing logprobs across all completion tokens

        Args:
            prompt: Input prompt (e.g., "What percentage? Answer:")
            completion: Completion to score (e.g., "10%")

        Returns:
            Total log probability for the completion
        """
        # Tokenize prompt and full text
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt")
        full_inputs = self.tokenizer(prompt + completion, return_tensors="pt")

        # Move to device
        prompt_inputs = {k: v.to(self.device) for k, v in prompt_inputs.items()}
        full_inputs = {k: v.to(self.device) for k, v in full_inputs.items()}

        # Get prompt length (number of tokens)
        prompt_len = prompt_inputs['input_ids'].shape[1]

        # Get completion tokens
        completion_token_ids = full_inputs['input_ids'][0, prompt_len:]

        if len(completion_token_ids) == 0:
            logger.warning(f"Completion '{completion}' tokenizes to 0 tokens after prompt")
            return -100.0  # Very low probability

        # Forward pass to get logits
        with torch.no_grad():
            outputs = self.model(**full_inputs)
            logits = outputs.logits

        # Compute log probability for each completion token
        total_logprob = 0.0

        for i, token_id in enumerate(completion_token_ids):
            # Get logits at the position *before* this token
            # (position where model predicts this token)
            position = prompt_len + i - 1
            position_logits = logits[0, position, :]

            # Convert to log probabilities
            log_probs = torch.nn.functional.log_softmax(position_logits, dim=-1)

            # Get log prob for this specific token
            token_logprob = log_probs[token_id].item()
            total_logprob += token_logprob

        return total_logprob

    def score_options(
        self,
        prompt: str,
        options: List[str],
        normalize: bool = True,
    ) -> Dict[str, float]:
        """
        Score multiple options and return probability distribution

        Args:
            prompt: Input prompt
            options: List of possible completions to score
            normalize: If True, convert logprobs to probabilities summing to 1.0

        Returns:
            Dictionary mapping each option to its (normalized) probability
        """
        # Score each option
        logprobs = {}
        for option in options:
            logprob = self.score_completion(prompt, option)
            logprobs[option] = logprob
            logger.debug(f"Option '{option}': logprob = {logprob:.4f}")

        if not normalize:
            return logprobs

        # Convert to probabilities and normalize
        return self._normalize_logprobs(logprobs)

    def _normalize_logprobs(self, logprobs: Dict[str, float]) -> Dict[str, float]:
        """
        Convert logprobs to normalized probability distribution

        Uses softmax: P(option) = exp(logprob) / sum(exp(logprob) for all options)

        Args:
            logprobs: Dictionary of option -> logprob

        Returns:
            Dictionary of option -> probability (sums to 1.0)
        """
        options = list(logprobs.keys())
        logprob_values = np.array([logprobs[opt] for opt in options])

        # Numerical stability: subtract max before exp
        max_logprob = np.max(logprob_values)
        exp_values = np.exp(logprob_values - max_logprob)

        # Normalize
        probabilities = exp_values / np.sum(exp_values)

        return {opt: float(prob) for opt, prob in zip(options, probabilities)}

    def close(self):
        """
        Clean up model resources

        Moves model to CPU and clears GPU cache if applicable.
        """
        if hasattr(self, 'model'):
            self.model.cpu()
            del self.model

        if hasattr(self, 'tokenizer'):
            del self.tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model resources cleaned up")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
