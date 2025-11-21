"""
Prompt Construction Library for Grace Project

Centralized prompt construction for all experiments. This ensures:
1. Consistent prompting across production and test scripts
2. Model-aware prompting (reasoning vs non-reasoning models)
3. Single source of truth for prompt updates

Usage:
    from utils.prompts import construct_study1_exp1_prompt

    prompt = construct_study1_exp1_prompt(trial, model_name="google/gemma-2-2b-it")
"""

from typing import Dict
from .model_config import get_model_config


def construct_study1_exp1_quantity_prompt(trial: Dict, model_name: str) -> str:
    """
    Construct prompt for Study 1 Experiment 1 - Quantity Question

    Asks: "Now how many of the 3 items do you think have the property?"

    Args:
        trial: Trial data dictionary with keys:
            - story_setup: Scenario description
            - priorQ: Prior probability question
            - speach: Evidence/observation
            - speachQ: Posterior probability question
        model_name: Model identifier (e.g., "google/gemma-2-2b-it")

    Returns:
        Formatted prompt string
    """
    model_config = get_model_config(model_name)

    # Clean up HTML tags
    setup = trial['story_setup'].replace('<br>', '\n').strip()

    # Base prompt
    prompt = f"""{setup}

{trial['priorQ']}

{trial['speach']}

{trial['speachQ']}"""

    # Model-specific instruction
    if model_config.is_reasoning_model:
        prompt += "\n\nPlease provide your reasoning and answer."
    elif model_config.use_direct_answer_prompt:
        prompt += "\n\nAnswer directly with just a number (0, 1, 2, or 3).\n\nYour answer is:"
    else:
        # For base models, request concise answer
        prompt += "\n\nAnswer with just a number (0, 1, 2, or 3).\n\nYour answer is:"

    return prompt


def construct_study1_exp1_knowledge_prompt(trial: Dict, model_name: str) -> str:
    """
    Construct prompt for Study 1 Experiment 1 - Knowledge Question

    Asks: "Do you think X knows exactly how many of the 3 items have the property?"

    Args:
        trial: Trial data dictionary with knowledgeQ
        model_name: Model identifier

    Returns:
        Formatted prompt string
    """
    model_config = get_model_config(model_name)

    # Clean up HTML tags
    setup = trial['story_setup'].replace('<br>', '\n').strip()
    speach = trial['speach'].strip()
    knowledge_q = trial['knowledgeQ'].strip()

    # Base prompt
    prompt = f"""{setup}

{speach}

{knowledge_q}"""

    # Model-specific instruction
    if model_config.is_reasoning_model:
        prompt += "\n\nPlease provide your reasoning and answer."
    elif model_config.use_direct_answer_prompt:
        prompt += "\n\nAnswer directly with just \"yes\" or \"no\".\n\nYour answer is:"
    else:
        # For base models, request concise answer
        prompt += "\n\nAnswer with just \"yes\" or \"no\".\n\nYour answer is:"

    return prompt


def construct_study1_exp1_prompt(trial: Dict, model_name: str) -> str:
    """
    DEPRECATED: Use construct_study1_exp1_quantity_prompt() and
    construct_study1_exp1_knowledge_prompt() instead.

    This function now calls the quantity prompt for backward compatibility.
    """
    return construct_study1_exp1_quantity_prompt(trial, model_name)


def construct_study1_exp2_state_prompt(trial: Dict, state: int, model_name: str) -> str:
    """
    Construct prompt for Study 1 Experiment 2 - State Probability Query

    Args:
        trial: Trial data dictionary
        state: State value (0, 1, 2, or 3)
        model_name: Model identifier

    Returns:
        Formatted prompt string
    """
    model_config = get_model_config(model_name)

    # Extract components
    setup = trial['story_setup'].replace('<br>', '\n').strip()
    speach = trial['speach'].strip()

    # Determine object name and property from story
    story_shortname = trial['story_shortname']
    object_name = story_shortname  # e.g., "exams", "letters"

    # Property mapping
    property_map = {
        'exams': 'passing grades',
        'letters': 'checks inside',
        'eggs': 'cracks',
        'papers': 'citations',
    }
    property_name = property_map.get(story_shortname.lower(), 'the property')

    # Base prompt
    prompt = f"""Given the scenario below, what percentage probability do you assign that exactly {state} of the 3 {object_name} have {property_name}?

Scenario:
{setup}

{speach}

Respond with ONLY a percentage from this list: 0%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%

Your answer is:"""

    return prompt


def construct_study1_exp2_knowledge_prompt(trial: Dict, model_name: str) -> str:
    """
    Construct prompt for Study 1 Experiment 2 - Knowledge Question

    Args:
        trial: Trial data dictionary with knowledgeQ
        model_name: Model identifier

    Returns:
        Formatted prompt string
    """
    model_config = get_model_config(model_name)

    setup = trial['story_setup'].replace('<br>', '\n').strip()
    speach = trial['speach'].strip()
    knowledge_q = trial['knowledgeQ'].strip()

    prompt = f"""{setup}

{speach}

{knowledge_q}

Respond with ONLY "yes" or "no".

Your answer is:"""

    return prompt


def construct_study2_exp1_prompt(trial: Dict, model_name: str) -> str:
    """
    Construct prompt for Study 2 Experiment 1 (Politeness - Constrained Format)

    Args:
        trial: Trial data dictionary with keys:
            - Precontext: Background information
            - Scenario: Situation description
            - Goal: Speaker's goal
            - State: Quality of work (0-4 hearts)
            - SP_Name: Speaker name
            - LS_Name: Listener name
        model_name: Model identifier

    Returns:
        Formatted prompt string
    """
    model_config = get_model_config(model_name)

    precontext = trial['Precontext'].strip()
    scenario = trial['Scenario'].strip()
    goal = trial['Goal'].strip()
    state = trial['State'].strip()

    # Base prompt with role-playing
    prompt = f"""{precontext}

{scenario}

You are {trial['SP_Name']}. {goal}.

The quality of {trial['LS_Name']}'s work is: {state}"""

    # Model-specific instruction
    if model_config.is_reasoning_model:
        # Reasoning models: allow reasoning but constrain final format
        prompt += f"""

Respond to {trial['LS_Name']}'s question using the format:
"It [was/wasn't] [terrible/bad/good/amazing]"

You may explain your reasoning, but end with your response in the exact format above.

Your answer is:"""
    else:
        # Non-reasoning models: strict format constraint
        prompt += f"""

Respond to {trial['LS_Name']}'s question using ONLY this exact format (no explanation):
"It [was/wasn't] [terrible/bad/good/amazing]"

Your answer is:"""

    return prompt


def construct_study2_exp2_prompt(trial: Dict, model_name: str) -> str:
    """
    Construct prompt for Study 2 Experiment 2 (Politeness - Probability Extraction)

    Similar to Exp 1 but designed for logprob extraction

    Args:
        trial: Trial data dictionary
        model_name: Model identifier

    Returns:
        Formatted prompt string
    """
    # For now, use same prompt as Exp 1
    # Can be customized if needed for better logprob extraction
    return construct_study2_exp1_prompt(trial, model_name)


# Token lists for probability extraction

def get_percentage_tokens() -> list:
    """
    Get list of percentage tokens for Study 1 Exp 2

    Returns percentage strings like "0%", "10%", etc.
    The API client now handles multi-token sequences properly.
    """
    return [f"{p}%" for p in range(0, 101, 10)]


def get_percentage_values() -> list:
    """
    Get numeric percentage values (0, 10, 20, ..., 100)
    """
    return list(range(0, 101, 10))


def get_yesno_tokens() -> list:
    """
    Get yes/no tokens for knowledge questions

    Returns lowercase since most models prefer lowercase in prompts
    """
    return ["yes", "no"]


def get_polarity_tokens() -> list:
    """
    Get polarity tokens for Study 2 (was/wasn't)
    """
    return ["was", "wasn't"]


def get_quality_tokens() -> list:
    """
    Get quality tokens for Study 2 (terrible/bad/good/amazing)
    """
    return ["terrible", "bad", "good", "amazing"]
