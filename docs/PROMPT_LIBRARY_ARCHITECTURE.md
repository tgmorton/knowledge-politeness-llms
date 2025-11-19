# Prompt Library Architecture

## Problem

Originally, prompt construction logic was duplicated across:
- Production query scripts (`query_study*.py`)
- Test scripts (potentially)
- Manual testing / debugging

This meant:
- **Inconsistency risk**: Changes to prompts in one place might not propagate
- **Hard to maintain**: Need to update multiple files for prompt tweaks
- **Difficult to test**: Can't easily test prompts in isolation

## Solution: Centralized Prompt Library

All prompt construction now lives in **`src/utils/prompts.py`**

### Benefits

1. **Single source of truth**: One place to update prompts
2. **Consistency**: Production, testing, and debugging all use same prompts
3. **Model-aware**: Automatically adapts prompts based on model type
4. **Testable**: Can unit test prompt construction separately
5. **Reusable**: Any script can import and use standard prompts

### Architecture

```
src/utils/prompts.py
├── construct_study1_exp1_prompt()       # Knowledge attribution - text
├── construct_study1_exp2_state_prompt()  # Knowledge attribution - probabilities
├── construct_study1_exp2_knowledge_prompt()  # Knowledge - yes/no
├── construct_study2_exp1_prompt()       # Politeness - constrained format
├── construct_study2_exp2_prompt()       # Politeness - probabilities
├── get_percentage_tokens()              # ["0", "10", ..., "100"]
├── get_percentage_values()              # [0, 10, ..., 100]
├── get_yesno_tokens()                   # ["yes", "no"]
├── get_polarity_tokens()                # ["was", "wasn't"]
└── get_quality_tokens()                 # ["terrible", "bad", "good", "amazing"]
```

### Integration with Model Config

Prompts automatically adapt based on model type:

```python
from utils.prompts import construct_study1_exp1_prompt

# For non-reasoning model (Gemma-2-2B)
prompt = construct_study1_exp1_prompt(trial, "google/gemma-2-2b-it")
# → Adds "Answer directly without explanation:"

# For reasoning model (DeepSeek R1)
prompt = construct_study1_exp1_prompt(trial, "deepseek/deepseek-r1")
# → Adds "Please provide your reasoning and answer."
```

### Usage in Query Scripts

**Before** (duplicated logic):
```python
# In query_study1_exp1.py
def construct_prompt(trial: Dict, model_name: str) -> str:
    model_config = get_model_config(model_name)
    setup = trial['story_setup'].replace('<br>', '\n').strip()
    # ... 30 lines of prompt construction ...
    return prompt

# In query_study1_exp2.py
def construct_state_query_prompt(trial: Dict, state: int) -> str:
    # ... another 40 lines ...
```

**After** (shared library):
```python
# In ALL query scripts
from utils.prompts import construct_study1_exp1_prompt

prompt = construct_study1_exp1_prompt(trial, model_name)
```

### Example: Updating Prompts Project-Wide

To change Study 1 Exp 1 prompt format:

1. Edit **one function** in `src/utils/prompts.py`:
```python
def construct_study1_exp1_prompt(trial: Dict, model_name: str) -> str:
    # Update prompt here
    prompt = f"""NEW FORMAT..."""
    return prompt
```

2. **All scripts immediately use new format**:
   - `src/query_study1_exp1.py` (production)
   - `tests/quick_local_test.sh` (local testing)
   - Any debugging scripts
   - Future scripts

No need to update multiple files!

### Token Extraction Functions

Helper functions provide consistent token lists:

```python
from utils.prompts import get_percentage_tokens, get_yesno_tokens

# For probability extraction
pct_tokens = get_percentage_tokens()  # ["0", "10", "20", ..., "100"]
probs = client.extract_token_probabilities(prompt, pct_tokens)

# For binary questions
yn_tokens = get_yesno_tokens()  # ["yes", "no"]
probs = client.extract_binary_probabilities(prompt, yn_tokens)
```

**Why numeric tokens, not percentage tokens?**

Tokenizers split `"10%"` into `"10"` + `"%"` as separate tokens. When extracting logprobs with `max_tokens=1`, we only get the first token (`"10"`), so we must search for numeric tokens in the vocabulary.

### Files Updated

**Created**:
- `src/utils/prompts.py` - New prompt library

**Modified** (to use library):
- `src/query_study1_exp1.py`
- `src/query_study1_exp2.py`
- `src/query_study2_exp1.py`
- `tests/quick_local_test.sh` (uses same prompts via imports)

### Testing

Test prompts work correctly:

```bash
# Local test (uses shared library)
./tests/quick_local_test.sh

# Unit test prompt construction
python3 -c "
from src.utils.prompts import construct_study1_exp1_prompt

trial = {
    'story_setup': 'Test scenario',
    'priorQ': 'Prior question?',
    'speach': 'Evidence',
    'speachQ': 'Posterior question?'
}

prompt = construct_study1_exp1_prompt(trial, 'google/gemma-2-2b-it')
print(prompt)
"
```

### Future Extensions

Easy to add new prompt variants:

```python
# For few-shot prompting
def construct_study1_exp1_fewshot_prompt(trial: Dict, model_name: str, examples: List) -> str:
    base_prompt = construct_study1_exp1_prompt(trial, model_name)
    # Add examples
    return fewshot_prompt

# For chain-of-thought prompting
def construct_study1_exp1_cot_prompt(trial: Dict, model_name: str) -> str:
    # Add "Let's think step by step..."
    pass
```

All scripts can immediately use new variants by updating imports.

### Best Practices

1. **Always use library functions**: Never construct prompts inline
2. **Test changes locally first**: Run `./tests/quick_local_test.sh` after prompt updates
3. **Document prompt rationale**: Add comments explaining prompt design choices
4. **Version control**: Git tracks all prompt changes in one file
5. **Model-aware by default**: Always pass `model_name` parameter

### Related Documentation

- **Model Configuration**: `src/utils/model_config.py`
- **Reasoning Suppression**: `docs/REASONING_SUPPRESSION.md`
- **Model Prompting Guide**: `docs/MODEL_PROMPTING_GUIDE.md`

---

**Status**: Implemented
**Last Updated**: 2025-11-19
**Files**: `src/utils/prompts.py` + all query scripts
