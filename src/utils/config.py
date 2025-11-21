"""
Configuration for Grace Project experiments

Based on recommended defaults from DECISION_CHECKLIST.md:
- Quantization: fp16 (full precision)
- Context length: 4096 tokens
- Temperature settings:
  - Experiment 1 (text generation): 0.7
  - Experiment 2 (probabilities): 1.0
  - Structured extraction: 0.0
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model serving configuration"""

    # Model parameters
    dtype: str = "float16"  # fp16 for Phase 0
    max_model_len: int = 4096  # Context length
    tensor_parallel_size: int = 1  # Single GPU for Gemma-2B

    # vLLM server settings
    port: int = 8000
    host: str = "0.0.0.0"

    # Model name (will be overridden per deployment)
    model_name: str = "google/gemma-2-2b-it"

    # Resource requirements (for K8s manifests)
    gpu_count: int = 1
    memory_gi: int = 64
    cpu_count: int = 32


@dataclass
class ExperimentConfig:
    """Experiment query configuration"""

    # Temperature settings
    temp_text_generation: float = 0.7  # Experiment 1 - raw responses
    temp_probabilities: float = 1.0    # Experiment 2 - probability extraction
    temp_structured: float = 0.0       # Structured output parsing

    # Token generation limits
    max_tokens_text: int = 500         # Raw text responses
    max_tokens_probability: int = 1    # Single token for probability extraction

    # Probability extraction
    logprobs_count: int = 15           # Number of logprobs to return
    percentage_tokens: list = None     # Will be initialized in __post_init__

    # API settings
    timeout_seconds: int = 120         # Request timeout
    max_retries: int = 3               # Retry on failure
    retry_delay_seconds: int = 5       # Delay between retries

    # Batch processing
    batch_size: int = 1                # Sequential processing in Phase 0

    def __post_init__(self):
        """Initialize percentage tokens"""
        if self.percentage_tokens is None:
            # Tokens for probability extraction: "0%", "10%", "20%", ..., "100%"
            self.percentage_tokens = [f"{i}%" for i in range(0, 101, 10)]


@dataclass
class Study1Config:
    """Study 1 specific configuration"""

    # Input columns expected in study1.csv
    input_columns: list = None

    # Number of state queries per trial (0, 1, 2, 3 exams passed)
    num_state_queries: int = 4

    # Total queries per trial (4 states + 1 knowledge question)
    total_queries_per_trial: int = 5

    # Expected output columns for Experiment 2
    num_output_columns: int = 68  # 9 original + 44 probs + 12 stats + 3 knowledge

    def __post_init__(self):
        """Initialize input columns"""
        if self.input_columns is None:
            self.input_columns = [
                'participant_id',
                'story_shortname',
                'story_setup',
                'priorQ',
                'speach',  # Note: 'speach' not 'speech' (original dataset spelling)
                'speachQ',
                'knowledgeQ',
                'access',
                'observe'
            ]


@dataclass
class Study2Config:
    """Study 2 specific configuration"""

    # Input columns expected in study2.csv
    input_columns: list = None

    # Number of trials
    num_trials: int = 2424

    def __post_init__(self):
        """Initialize input columns"""
        if self.input_columns is None:
            self.input_columns = [
                'participant_id',
                'Scenario',
                'Goal',
                'State',
                'Response'
            ]


# Default configurations (singleton instances)
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()
DEFAULT_STUDY1_CONFIG = Study1Config()
DEFAULT_STUDY2_CONFIG = Study2Config()


def get_model_config(model_name: Optional[str] = None, **kwargs) -> ModelConfig:
    """Get model configuration with optional overrides"""
    config = ModelConfig()

    if model_name:
        config.model_name = model_name

        # Adjust resources based on model
        if "27b" in model_name.lower() or "70b" in model_name.lower() or "120b" in model_name.lower():
            config.tensor_parallel_size = 2
            config.gpu_count = 2
            config.memory_gi = 192

        # Set correct dtype for modern models (Gemma-2 and Llama-3 require bfloat16)
        if "llama" in model_name.lower() or "gemma-2" in model_name.lower():
            config.dtype = "bfloat16"

    # Apply any additional overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


def get_experiment_config(**kwargs) -> ExperimentConfig:
    """Get experiment configuration with optional overrides"""
    config = ExperimentConfig()

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config
