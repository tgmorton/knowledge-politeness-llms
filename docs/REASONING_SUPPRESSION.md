# Reasoning Suppression Strategy

## Problem Statement

Instruction-tuned models (Gemma-2-2B-IT, Llama-3-70B-Instruct) were generating verbose Chain-of-Thought (CoT) reasoning even when not requested, leading to responses like:

```
**Understanding the Problem**
* We're dealing with probability and a bit of logic.
* The key is to figure out how Laura's statement changes our initial assumptions.

**Initial Assumptions**
* **Assumption 1:** We assume that all 3 letters have the same probability...

**Answer:**
The probability that one of the letters has checks inside is higher...
```

This is problematic because:
1. **Unnecessarily verbose** for simple questions
2. **Wastes tokens** and increases latency
3. **Mixes reasoning with answer**, making extraction harder
4. **Not desired** for non-reasoning models (we want direct answers)

## Solution: Model-Aware Prompting

### Implementation Overview

Created `src/utils/model_config.py` that:
1. **Classifies models** as reasoning vs non-reasoning
2. **Configures prompting** strategy per model type
3. **Sets parameters** (temperature, max_tokens, stop_tokens)
4. **Provides system messages** for each experiment type

### Key Components

#### 1. Model Classification

```python
# Non-reasoning model (instruction-tuned but not CoT-trained)
"google/gemma-2-2b-it": ModelConfig(
    is_reasoning_model=False,
    use_direct_answer_prompt=True,  # Add "Answer directly" instruction
    max_tokens_text=100,  # Limit output length
    temperature_text=0.3,  # More deterministic
)

# Reasoning model (CoT-enabled)
"deepseek/deepseek-r1": ModelConfig(
    is_reasoning_model=True,
    use_direct_answer_prompt=False,  # Let it reason
    max_tokens_text=2000,  # Allow full reasoning traces
    temperature_text=0.7,
)

# Base model (no instruction-following)
"gpt2": ModelConfig(
    is_reasoning_model=False,
    use_direct_answer_prompt=False,  # Just continues text
    max_tokens_text=50,
    temperature_text=0.7,
)
```

#### 2. Prompt Adaptation

**Study 1 Experiment 1** (Knowledge Attribution):

**Before** (all models):
```
{scenario}
{prior_question}
{speech}
{posterior_question}

Please provide your reasoning and answer.
```

**After** (non-reasoning models):
```
{scenario}
{prior_question}
{speech}
{posterior_question}

Answer directly without explanation:
```

**After** (reasoning models):
```
{scenario}
{prior_question}
{speech}
{posterior_question}

Please provide your reasoning and answer.
```

**Study 2 Experiment 1** (Politeness):

**Before**:
```
You are Emma. {goal}.
Quality: {state}

Respond using ONLY the format:
"It [was/wasn't] [terrible/bad/good/amazing]"

Your response:
```

**After** (non-reasoning models):
```
You are Emma. {goal}.
Quality: {state}

Respond to Wendy's question using ONLY this exact format (no explanation):
"It [was/wasn't] [terrible/bad/good/amazing]"

Your response:
```

**After** (reasoning models):
```
You are Emma. {goal}.
Quality: {state}

Respond using the format:
"It [was/wasn't] [terrible/bad/good/amazing]"

You may explain your reasoning, but end with your response in the exact format above.

Your response:
```

#### 3. Parameter Tuning

**Non-reasoning models** (Gemma-2, Llama-3):
- `temperature=0.3` - More deterministic to suppress creativity
- `max_tokens=100` - Limit verbosity for text responses
- `max_tokens=50` - Even more constrained for structured formats
- `stop_tokens=["<end_of_turn>"]` - Stop at model's natural boundary

**Reasoning models** (DeepSeek R1):
- `temperature=0.7` - Allow exploration during reasoning
- `max_tokens=2000` - Allow full reasoning traces
- Capture reasoning separately in JSONL file with `result_id` linking

**Base models** (GPT-2, Phi-2):
- No special prompting (just continue text)
- Lower max_tokens since they're less coherent

### Files Modified

1. **`src/utils/model_config.py`** (NEW)
   - Model registry with all 6 production models + test models
   - Configuration class with prompting strategy
   - Helper functions: `get_model_config()`, `is_reasoning_model()`, `get_system_message()`

2. **`src/query_study1_exp1.py`**
   - Import `get_model_config`
   - Update `construct_prompt()` to take `model_name` parameter
   - Add conditional prompting based on `is_reasoning_model`
   - Update `process_trial()` to use model-specific parameters
   - Pass `temperature`, `max_tokens`, `stop` from config

3. **`src/query_study2_exp1.py`**
   - Import `get_model_config`
   - Update `construct_prompt()` to take `model_name` parameter
   - Add conditional prompting for reasoning vs non-reasoning
   - Update `process_trial()` to use `max_tokens_structured`
   - Pass model-specific parameters to `client.generate_text()`

4. **`docs/MODEL_PROMPTING_GUIDE.md`**
   - Added new section: "Reasoning Model vs Non-Reasoning Model Strategy"
   - Documented problem, solution, configuration examples
   - Updated open questions to include reasoning suppression testing

## Testing Checklist

### Local Testing (Phase 0)

- [ ] Test with GPT-2 (base model, should show minimal change)
- [ ] Test with Gemma-2-2B-IT (instruction model, should suppress CoT)
- [ ] Verify responses are shorter and more direct
- [ ] Verify empty response issue is fixed (min_new_tokens=10)
- [ ] Check Study 2 Exp 1 outputs proper format

### Production Testing (Phase 4)

For each model (Gemma-2B, 9B, 27B, Llama-70B):
- [ ] Run 10 trials from Study 1 Exp 1
- [ ] Verify responses are concise (< 100 tokens)
- [ ] Verify no verbose CoT reasoning
- [ ] Check response quality (still coherent and sensible)
- [ ] Run 10 trials from Study 2 Exp 1
- [ ] Verify proper format: "It [was/wasn't] [terrible/bad/good/amazing]"
- [ ] No extraneous explanation or reasoning

## Expected Behavior

### Study 1 Experiment 1

**Desired output** (non-reasoning model):
```
All 3 exams have passing grades.
```

**NOT**:
```
**Reasoning:**
* We know that 3 students took the exam.
* Mark has looked at 2 exams and found 2 with passing grades.

**Answer:**
All 3 exams have passing grades.
```

### Study 2 Experiment 1

**Desired output** (non-reasoning model):
```
It wasn't terrible
```

**NOT**:
```
**My Judgment:**
Robert's response is not particularly helpful...

**Alternative Responses:**
A more appropriate response from Robert could have been...
```

## Reasoning Trace Capture (Future)

For reasoning models (DeepSeek R1, o1, o3):
- Reasoning trace automatically captured in `CompletionResponse.reasoning_trace`
- Saved to separate JSONL file via `ReasoningTraceWriter`
- Linked to main results via `result_id` UUID
- Allows post-hoc analysis of reasoning process

Format:
```json
{
  "result_id": "f30eb8a7-55de-4c15-9fae-ca3817c24b31",
  "trial_index": 0,
  "prompt": "Students in the intro bio class...",
  "reasoning_trace": "<thinking>Let me analyze this step by step...</thinking>",
  "response": "All 3 exams have passing grades.",
  "model_name": "deepseek/deepseek-r1",
  "timestamp": "2025-11-19T12:00:00"
}
```

## Troubleshooting

### Issue: Model still generates verbose reasoning

**Possible causes**:
1. Prompt instruction not strong enough
2. Temperature too high (try lowering to 0.1)
3. Max tokens too high (try 50 instead of 100)
4. Model heavily biased toward CoT during training

**Solutions**:
1. Add even more explicit instruction: "ANSWER ONLY. NO EXPLANATION."
2. Lower temperature to 0.0 (fully deterministic)
3. Reduce max_tokens to 30
4. Use stop sequences to cut off after first sentence
5. Post-process to extract just the answer (last resort)

### Issue: Responses too short or cut off

**Possible causes**:
1. Max tokens too low
2. Stop tokens firing too early
3. Temperature too low (model uncertain, stops early)

**Solutions**:
1. Increase max_tokens (try 150)
2. Remove or adjust stop_tokens
3. Increase temperature slightly (0.5)

### Issue: Reasoning models not capturing traces

**Possible causes**:
1. vLLM not returning reasoning field
2. Model not in reasoning mode
3. Reasoning not in expected format

**Solutions**:
1. Check vLLM server config (enable reasoning output)
2. Verify model is configured as reasoning model
3. Parse response text for `<thinking>` tags or other markers
4. Manually extract reasoning from response if needed

## References

- **Model Configuration**: `src/utils/model_config.py`
- **Study 1 Exp 1**: `src/query_study1_exp1.py`
- **Study 2 Exp 1**: `src/query_study2_exp1.py`
- **Prompting Guide**: `docs/MODEL_PROMPTING_GUIDE.md`
- **CLAUDE.md**: Project overview and navigation guide

---

**Status**: Implemented, pending testing
**Last Updated**: 2025-11-19
**Author**: Claude (AI Assistant)
