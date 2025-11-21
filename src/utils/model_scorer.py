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
from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import re

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
        is_reasoning_model: bool = False,
        reasoning_start_token: str = "<think>",
        reasoning_end_token: str = "</think>",
    ):
        """
        Initialize model scorer

        Args:
            model_name: HuggingFace model identifier (e.g., "google/gemma-2-2b-it")
            cache_dir: Directory to cache downloaded models (e.g., "/models/.cache")
            device: Device to use ("cpu", "cuda", "mps", or "auto")
            torch_dtype: Torch dtype ("float16", "float32", "bfloat16", or "auto")
            is_reasoning_model: Whether this is a reasoning model (e.g., DeepSeek-R1)
            reasoning_start_token: Token marking start of reasoning trace
            reasoning_end_token: Token marking end of reasoning trace
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.is_reasoning_model = is_reasoning_model
        self.reasoning_start_token = reasoning_start_token
        self.reasoning_end_token = reasoning_end_token

        if cache_dir:
            logger.info(f"Loading model: {model_name} (cache: {cache_dir})")
        else:
            logger.info(f"Loading model: {model_name}")

        # Determine device and device_map
        device_map_for_model = device  # Will be used in from_pretrained
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                # For multi-GPU setups, use device_map="auto" to distribute model
                gpu_count = torch.cuda.device_count()
                if gpu_count > 1:
                    device_map_for_model = "auto"
                    logger.info(f"Detected {gpu_count} GPUs - will use automatic device mapping")
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
            device_map=device_map_for_model,
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

    def generate_with_reasoning(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        return_logprobs: bool = False,
    ) -> Dict:
        """
        Generate text and extract reasoning traces (for reasoning models)

        For reasoning models like DeepSeek-R1, this extracts:
        - Reasoning trace (text within <think>...</think>)
        - Final answer (text outside reasoning tags)
        - Token-level logprobs (if requested)

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            return_logprobs: Whether to return token-level logprobs

        Returns:
            Dictionary with:
                - full_text: Complete generated text
                - reasoning_trace: Text within reasoning tags (or None)
                - final_answer: Text outside reasoning tags
                - reasoning_logprobs: List of (token, logprob) for reasoning (if requested)
                - answer_logprobs: List of (token, logprob) for answer (if requested)
        """
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        prompt_len = inputs['input_ids'].shape[1]

        # Generate with logprobs if requested
        with torch.no_grad():
            if return_logprobs:
                # Generate tokens one by one to capture logprobs
                generated_tokens = []
                generated_logprobs = []

                input_ids = inputs['input_ids']

                for _ in range(max_new_tokens):
                    outputs = self.model(input_ids)
                    logits = outputs.logits[:, -1, :]

                    # Apply temperature
                    if temperature > 0:
                        logits = logits / temperature

                    # Get log probabilities
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                    # Sample or greedy
                    if temperature == 0:
                        next_token = torch.argmax(logits, dim=-1)
                    else:
                        probs = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).squeeze()

                    # Get logprob of selected token
                    token_logprob = log_probs[0, next_token].item()

                    # Append token and logprob
                    generated_tokens.append(next_token.item())
                    generated_logprobs.append(token_logprob)

                    # Check for EOS
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break

                    # Append to input for next iteration
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

                # Decode generated text
                full_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)

                # Decode individual tokens for logprob mapping
                token_texts = [self.tokenizer.decode([t], skip_special_tokens=False) for t in generated_tokens]

            else:
                # Simple generation without logprobs
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                # Decode only the generated part (exclude prompt)
                generated_ids = outputs[0, prompt_len:]
                full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
                token_texts = []
                generated_logprobs = []

        # Parse reasoning trace if this is a reasoning model
        if self.is_reasoning_model:
            reasoning_trace, final_answer = self._parse_reasoning(full_text)

            if return_logprobs and token_texts:
                # Split logprobs into reasoning and answer
                reasoning_logprobs, answer_logprobs = self._split_logprobs(
                    token_texts, generated_logprobs, reasoning_trace, final_answer
                )
            else:
                reasoning_logprobs = []
                answer_logprobs = []
        else:
            reasoning_trace = None
            final_answer = full_text
            reasoning_logprobs = []
            answer_logprobs = list(zip(token_texts, generated_logprobs)) if return_logprobs else []

        return {
            'full_text': full_text,
            'reasoning_trace': reasoning_trace,
            'final_answer': final_answer,
            'reasoning_logprobs': reasoning_logprobs,
            'answer_logprobs': answer_logprobs,
        }

    def _parse_reasoning(self, text: str) -> Tuple[Optional[str], str]:
        """
        Parse reasoning trace from generated text

        Extracts text within reasoning tags (e.g., <think>...</think>)
        and returns final answer (text outside tags).

        Args:
            text: Generated text potentially containing reasoning tags

        Returns:
            Tuple of (reasoning_trace, final_answer)
        """
        # Find reasoning trace
        pattern = re.escape(self.reasoning_start_token) + r'(.*?)' + re.escape(self.reasoning_end_token)
        match = re.search(pattern, text, re.DOTALL)

        if match:
            reasoning_trace = match.group(1).strip()
            # Remove reasoning tags from text to get final answer
            final_answer = re.sub(pattern, '', text, flags=re.DOTALL).strip()
        else:
            reasoning_trace = None
            final_answer = text.strip()

        return reasoning_trace, final_answer

    def _split_logprobs(
        self,
        token_texts: List[str],
        logprobs: List[float],
        reasoning_trace: Optional[str],
        final_answer: str,
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Split token-level logprobs into reasoning and answer sections

        Args:
            token_texts: List of decoded token strings
            logprobs: List of logprob values (same length as token_texts)
            reasoning_trace: Extracted reasoning text (or None)
            final_answer: Extracted final answer text

        Returns:
            Tuple of (reasoning_logprobs, answer_logprobs)
            Each is a list of (token, logprob) tuples
        """
        if not reasoning_trace:
            # No reasoning trace, all tokens are answer
            return [], list(zip(token_texts, logprobs))

        # Reconstruct full text to find boundaries
        full_text = ''.join(token_texts)

        # Find where reasoning starts and ends
        start_idx = full_text.find(self.reasoning_start_token)
        end_idx = full_text.find(self.reasoning_end_token)

        if start_idx == -1 or end_idx == -1:
            # Couldn't find tags, return everything as answer
            logger.warning("Could not find reasoning tags in token stream")
            return [], list(zip(token_texts, logprobs))

        # Accumulate character positions to determine which tokens belong where
        reasoning_logprobs = []
        answer_logprobs = []

        char_pos = 0
        for token, logprob in zip(token_texts, logprobs):
            token_end = char_pos + len(token)

            # Determine if this token is in reasoning section
            # (between start_idx and end_idx + len(end_token))
            if start_idx <= char_pos < end_idx + len(self.reasoning_end_token):
                # Skip the tag tokens themselves, only keep reasoning content
                if token not in [self.reasoning_start_token, self.reasoning_end_token]:
                    reasoning_logprobs.append((token, logprob))
            else:
                answer_logprobs.append((token, logprob))

            char_pos = token_end

        return reasoning_logprobs, answer_logprobs

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
