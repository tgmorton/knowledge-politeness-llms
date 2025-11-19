"""
Model-Specific Configuration for Grace Project

Defines which models are "reasoning models" (CoT-enabled) vs standard models,
and provides model-specific prompting strategies.
"""

from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    name: str
    is_reasoning_model: bool  # True for o1, o3, DeepSeek-R1, etc.
    supports_system_message: bool  # True for chat models
    chat_template: Optional[str] = None  # e.g., "gemma", "llama3", "chatml"
    stop_tokens: Optional[List[str]] = None

    # Prompting strategy
    use_direct_answer_prompt: bool = True  # Add "answer directly" instructions
    max_tokens_text: int = 500  # For Experiment 1
    max_tokens_structured: int = 50  # For constrained format responses
    temperature_text: float = 0.7
    temperature_probs: float = 1.0


# Model registry
MODELS: Dict[str, ModelConfig] = {
    # Gemma-2 family (Google) - Non-reasoning models
    "google/gemma-2-2b-it": ModelConfig(
        name="google/gemma-2-2b-it",
        is_reasoning_model=False,
        supports_system_message=True,
        chat_template="gemma",
        stop_tokens=["<end_of_turn>"],
        use_direct_answer_prompt=True,  # Suppress CoT
        max_tokens_text=100,  # Limit verbosity
        max_tokens_structured=50,
        temperature_text=0.3,  # More deterministic for direct answers
    ),

    "google/gemma-2-9b-it": ModelConfig(
        name="google/gemma-2-9b-it",
        is_reasoning_model=False,
        supports_system_message=True,
        chat_template="gemma",
        stop_tokens=["<end_of_turn>"],
        use_direct_answer_prompt=True,
        max_tokens_text=100,
        max_tokens_structured=50,
        temperature_text=0.3,
    ),

    "google/gemma-2-27b-it": ModelConfig(
        name="google/gemma-2-27b-it",
        is_reasoning_model=False,
        supports_system_message=True,
        chat_template="gemma",
        stop_tokens=["<end_of_turn>"],
        use_direct_answer_prompt=True,
        max_tokens_text=100,
        max_tokens_structured=50,
        temperature_text=0.3,
    ),

    # Llama-3 family (Meta) - Non-reasoning models
    "meta-llama/Meta-Llama-3-70B-Instruct": ModelConfig(
        name="meta-llama/Meta-Llama-3-70B-Instruct",
        is_reasoning_model=False,
        supports_system_message=True,
        chat_template="llama3",
        stop_tokens=["<|eot_id|>", "<|end_of_text|>"],
        use_direct_answer_prompt=True,
        max_tokens_text=100,
        max_tokens_structured=50,
        temperature_text=0.3,
    ),

    # GPT-OSS family (OpenAI) - TBD when released
    "openai/gpt-oss-20b": ModelConfig(
        name="openai/gpt-oss-20b",
        is_reasoning_model=False,  # Update if reasoning-enabled
        supports_system_message=True,  # Assumption
        chat_template="chatml",  # Assumption, update when released
        use_direct_answer_prompt=True,
        max_tokens_text=100,
        max_tokens_structured=50,
        temperature_text=0.3,
    ),

    "openai/gpt-oss-120b": ModelConfig(
        name="openai/gpt-oss-120b",
        is_reasoning_model=False,  # Update if reasoning-enabled
        supports_system_message=True,  # Assumption
        chat_template="chatml",
        use_direct_answer_prompt=True,
        max_tokens_text=100,
        max_tokens_structured=50,
        temperature_text=0.3,
    ),

    # Reasoning models (for future use)
    "deepseek/deepseek-r1": ModelConfig(
        name="deepseek/deepseek-r1",
        is_reasoning_model=True,  # This is a reasoning model
        supports_system_message=True,
        chat_template="deepseek",
        use_direct_answer_prompt=False,  # Let it reason!
        max_tokens_text=2000,  # Allow longer reasoning traces
        max_tokens_structured=1000,
        temperature_text=0.7,
    ),

    # Test models (for local development)
    "gpt2": ModelConfig(
        name="gpt2",
        is_reasoning_model=False,
        supports_system_message=False,  # GPT-2 doesn't support chat format
        use_direct_answer_prompt=False,  # GPT-2 just continues text
        max_tokens_text=50,
        max_tokens_structured=20,
        temperature_text=0.7,
    ),

    "microsoft/phi-2": ModelConfig(
        name="microsoft/phi-2",
        is_reasoning_model=False,
        supports_system_message=False,
        use_direct_answer_prompt=True,
        max_tokens_text=100,
        max_tokens_structured=50,
        temperature_text=0.3,
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """
    Get configuration for a model

    Args:
        model_name: Model identifier (e.g., "google/gemma-2-2b-it")

    Returns:
        ModelConfig for the model, or default config if not found
    """
    if model_name in MODELS:
        return MODELS[model_name]

    # Default config for unknown models
    return ModelConfig(
        name=model_name,
        is_reasoning_model=False,
        supports_system_message=True,  # Assume modern chat model
        use_direct_answer_prompt=True,
        max_tokens_text=100,
        max_tokens_structured=50,
        temperature_text=0.3,
    )


def is_reasoning_model(model_name: str) -> bool:
    """Check if model is a reasoning model (CoT-enabled)"""
    config = get_model_config(model_name)
    return config.is_reasoning_model


def get_system_message(model_name: str, experiment_type: str = "general") -> Optional[str]:
    """
    Get appropriate system message for model and experiment

    Args:
        model_name: Model identifier
        experiment_type: "study1_exp1", "study1_exp2", "study2_exp1", "study2_exp2", or "general"

    Returns:
        System message string, or None if model doesn't support system messages
    """
    config = get_model_config(model_name)

    if not config.supports_system_message:
        return None

    # For reasoning models, minimal or no system message
    if config.is_reasoning_model:
        return "You are a helpful assistant participating in a research study."

    # For non-reasoning models, constrain output format
    if experiment_type == "study1_exp1":
        return (
            "You are participating in a research study. "
            "Answer each question directly and concisely in 1-2 sentences. "
            "Do not provide explanations, reasoning, or analysis - just your direct answer."
        )
    elif experiment_type == "study1_exp2":
        return (
            "You are participating in a research study on probabilistic reasoning. "
            "Respond with ONLY the requested percentage (e.g., '30%'). "
            "Do not provide explanations or reasoning."
        )
    elif experiment_type == "study2_exp1":
        return (
            "You are participating in a research study. "
            "Respond using ONLY the exact format requested. "
            "Do not provide explanations, reasoning, or additional commentary."
        )
    elif experiment_type == "study2_exp2":
        return (
            "You are participating in a research study. "
            "Respond using ONLY the exact format requested."
        )
    else:
        return (
            "You are a helpful assistant. "
            "Answer questions directly and concisely without unnecessary explanations."
        )
