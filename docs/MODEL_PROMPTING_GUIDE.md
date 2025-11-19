# Model-Specific Prompting Guide

This document outlines the prompting requirements, chat templates, and API specifications for each model in the Grace Project. Different models have different prompting schemas, and using the correct format is critical for getting valid responses.

---

## Overview

**Models in Grace Project**:
1. Gemma-2 2B (Google)
2. Gemma-2 9B (Google)
3. Gemma-2 27B (Google)
4. Llama-3 70B (Meta)
5. GPT-OSS 20B (OpenAI) - pending release
6. GPT-OSS 120B (OpenAI) - pending release

**Key Differences**:
- Chat vs completion API
- System prompt support
- Chat template format
- Special tokens
- API endpoint compatibility

---

## Decision Points Per Model

### Universal Decisions (Apply to All Models)

- [ ] **Temperature Settings**:
  - Experiment 1 (text generation): T = _____ (recommended: 0.7)
  - Experiment 2 (probabilities): T = _____ (recommended: 1.0)
  - Structured extraction: T = _____ (recommended: 0.0)

- [ ] **API Type**:
  - [ ] Chat Completion API (`/v1/chat/completions`)
  - [ ] Completion API (`/v1/completions`)
  - Decision: _____________________

- [ ] **System Prompt Strategy**:
  - [ ] Use same system prompt across all models
  - [ ] Customize per model based on training
  - [ ] No system prompt
  - Decision: _____________________

---

## Model 1: Gemma-2 2B / 9B / 27B

### Model Information

- **Family**: Google Gemma-2
- **Training**: Instruction-tuned (`-it` variants)
- **License**: Apache 2.0
- **Repository**: 
  - `google/gemma-2-2b-it`
  - `google/gemma-2-9b-it`
  - `google/gemma-2-27b-it`

### Chat Template

Gemma-2 uses a specific chat template format:

```
<start_of_turn>user
{user_message}<end_of_turn>
<start_of_turn>model
{assistant_response}<end_of_turn>
```

**With system prompt**:
```
<start_of_turn>system
{system_message}<end_of_turn>
<start_of_turn>user
{user_message}<end_of_turn>
<start_of_turn>model
```

### vLLM API Usage

**Option A: Chat Completion API** (Recommended)
```python
response = requests.post(
    f"{model_endpoint}/v1/chat/completions",
    json={
        "model": "google/gemma-2-2b-it",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ],
        "temperature": 0.7,
        "max_tokens": 100,
        "logprobs": True,
        "top_logprobs": 15
    }
)
```

**Option B: Completion API** (Manual template)
```python
prompt = """<start_of_turn>user
What is 2+2?<end_of_turn>
<start_of_turn>model
"""

response = requests.post(
    f"{model_endpoint}/v1/completions",
    json={
        "model": "google/gemma-2-2b-it",
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 100,
        "logprobs": 15
    }
)
```

### Decision Points for Gemma-2

- [ ] **System Prompt Content**: _____________________
- [ ] **Use Chat API or Completion API**: _____________________
- [ ] **Stop Tokens**: `["<end_of_turn>"]` or vLLM default?
- [ ] **Max Tokens**: _____ (recommended: 100 for Exp 1, 1 for Exp 2)

### Testing Checklist

- [ ] Test chat template renders correctly
- [ ] Verify system prompt is included (if using)
- [ ] Check that responses don't include template tokens
- [ ] Validate logprobs extraction works
- [ ] Test with Study 1 sample prompt

---

## Model 2: Llama-3 70B

### Model Information

- **Family**: Meta Llama-3
- **Training**: Instruction-tuned (`-Instruct` variant)
- **License**: Llama 3 License (requires acceptance)
- **Repository**: `meta-llama/Meta-Llama-3-70B-Instruct`
- **HuggingFace Token**: Required (gated model)

### Chat Template

Llama-3 uses a different chat template:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

**Special Tokens**:
- `<|begin_of_text|>` - Start of conversation
- `<|start_header_id|>` - Start of role header
- `<|end_header_id|>` - End of role header
- `<|eot_id|>` - End of turn

### vLLM API Usage

**Option A: Chat Completion API** (Recommended)
```python
response = requests.post(
    f"{model_endpoint}/v1/chat/completions",
    json={
        "model": "meta-llama/Meta-Llama-3-70B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ],
        "temperature": 0.7,
        "max_tokens": 100,
        "logprobs": True,
        "top_logprobs": 15
    }
)
```

**Option B: Completion API** (Manual template)
```python
prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

What is 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

response = requests.post(
    f"{model_endpoint}/v1/completions",
    json={
        "model": "meta-llama/Meta-Llama-3-70B-Instruct",
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 100,
        "logprobs": 15
    }
)
```

### Decision Points for Llama-3 70B

- [ ] **System Prompt Content**: _____________________
- [ ] **Use Chat API or Completion API**: _____________________
- [ ] **Stop Tokens**: `["<|eot_id|>", "<|end_of_text|>"]` or vLLM default?
- [ ] **Max Tokens**: _____ (recommended: 100 for Exp 1, 1 for Exp 2)
- [ ] **HuggingFace Token**: Obtained and configured in Kubernetes Secret?

### Testing Checklist

- [ ] HuggingFace token works (test model download)
- [ ] Test chat template renders correctly
- [ ] Verify system prompt is included (if using)
- [ ] Check that responses don't include special tokens
- [ ] Validate logprobs extraction works
- [ ] Test with Study 1 sample prompt

---

## Model 3 & 4: GPT-OSS 20B / 120B

### Model Information

- **Family**: OpenAI GPT-OSS (MoE transformers)
- **Training**: Unknown until release (likely instruction-tuned)
- **License**: OpenAI OSS License (TBD)
- **Repository**: TBD (pending OpenAI release)
- **Release Date**: Unknown (monitoring OpenAI announcements)

### Chat Template

**Status**: Unknown until release

**Likely scenarios**:
1. **GPT-2/GPT-3 style** (no special chat template, just text continuation)
2. **ChatGPT style** (similar to OpenAI API format)
3. **Custom format** (new template specific to GPT-OSS)

### Decision Points for GPT-OSS Models

**PENDING OPENAI RELEASE**

Once released, determine:
- [ ] **Chat Template Format**: _____________________
- [ ] **Special Tokens**: _____________________
- [ ] **System Prompt Support**: YES / NO
- [ ] **Recommended API Type**: Chat / Completion
- [ ] **vLLM Compatibility**: Check if vLLM supports MoE for this model
- [ ] **Trust Remote Code**: May need `--trust-remote-code` flag
- [ ] **Stop Tokens**: _____________________

### Pre-Release Checklist

- [ ] Monitor OpenAI blog/GitHub for GPT-OSS release announcement
- [ ] Check HuggingFace for model availability
- [ ] Review model card for prompting instructions
- [ ] Test local download and vLLM compatibility
- [ ] Verify MoE routing works correctly
- [ ] Document actual chat template once available

### Placeholder vLLM Command (Update After Release)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-20b \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 1 \
  --trust-remote-code  # May be needed for MoE
```

---

## Prompt Engineering Strategy

### Experiment 1: Raw Text Generation

**Goal**: Get natural language responses to knowledge/politeness questions

**Prompt Structure**:
```
Context: {story_setup}

{speech}

Question: {question}

Answer:
```

**Temperature**: 0.7 (allow some variation, but mostly deterministic)

**Max Tokens**: 50-100 (allow full sentence responses)

**Stop Sequences**: Model-specific (e.g., `<end_of_turn>`, `<|eot_id|>`)

### Experiment 2: Probability Extraction

**Goal**: Get logprobs for specific tokens to build probability distributions

**For State Probabilities** (Study 1):
```
Context: {story_setup}

{speech}

Question: What percentage probability do you assign that exactly {state} of the 3 {items} {property}?

Express your answer as a percentage from 0% to 100%, in increments of 10%.

Answer:
```

**Expected Tokens**: `["0%", "10%", "20%", ..., "100%"]`

**Temperature**: 1.0 (unbiased sampling for probability distribution)

**Max Tokens**: 1 (force single token prediction)

**Logprobs**: 15 (capture all percentage tokens)

**For Knowledge Questions** (Study 1):
```
Context: {story_setup}

Question: {knowledge_question}

Answer (yes or no):
```

**Expected Tokens**: `["yes", "no", "Yes", "No"]`

**Temperature**: 1.0

**Max Tokens**: 1

**Logprobs**: 5

### Structured Extraction with DeepSeek

**Goal**: Parse raw responses into structured JSON

**Prompt Structure**:
```
You are a data extraction assistant. Extract information from the following response into JSON format.

Raw Response: "{raw_response}"

Expected Schema:
{json_schema}

Return ONLY valid JSON, no explanation.
```

**Temperature**: 0.0 (deterministic extraction)

**Max Tokens**: 200

---

## Tokenization Concerns

### Percentage Tokens

Different models may tokenize percentages differently:

**Test Required**: For each model, verify how these are tokenized:
- `0%` - Single token or two tokens (`0` + `%`)?
- `10%` - Single token or two tokens?
- `100%` - Single token, two tokens, or three?

**Implication**: If percentages are multi-token, logprob extraction becomes more complex.

**Solution Options**:
1. Request longer logprob list (e.g., 50 instead of 15)
2. Parse multi-token sequences
3. Use alternative format (e.g., "zero percent" vs "0%")

**Decision per model**:
- Gemma-2: _____________________
- Llama-3: _____________________
- GPT-OSS: _____________________ (TBD)

### Yes/No Tokens

Similarly, test capitalization:
- `yes` vs `Yes` vs `YES`
- `no` vs `No` vs `NO`

**Decision**: Combine probabilities for all variants? Or enforce lowercase in prompt?

---

## System Prompt Recommendations

### Option 1: Minimal System Prompt

```
You are a helpful assistant. Answer questions accurately and concisely.
```

**Pros**: Simple, unlikely to bias responses
**Cons**: May not encourage desired format

### Option 2: Task-Specific System Prompt

**For Experiment 1**:
```
You are participating in a research study. Read the scenario carefully and answer the question based on the information provided. Be thoughtful and natural in your responses.
```

**For Experiment 2**:
```
You are participating in a research study on probabilistic reasoning. For each question, provide your probability estimate as a percentage. Be honest about your uncertainty.
```

**Pros**: Sets appropriate context
**Cons**: May introduce unintended biases

### Option 3: No System Prompt

Use only user messages.

**Pros**: Eliminates one source of variation
**Cons**: Some models perform better with system context

### Decision

- [ ] **System Prompt Strategy**: _____________________
- [ ] **System Prompt Text** (if using): _____________________
- [ ] **Test with/without system prompt**: _____________________

---

## API Compatibility Matrix

| Model | vLLM Chat API | vLLM Completion API | OpenAI Python SDK | Custom Template Needed |
|-------|--------------|-------------------|------------------|----------------------|
| Gemma-2 2B | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No (vLLM handles it) |
| Gemma-2 9B | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No (vLLM handles it) |
| Gemma-2 27B | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No (vLLM handles it) |
| Llama-3 70B | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No (vLLM handles it) |
| GPT-OSS 20B | ❓ TBD | ❓ TBD | ❓ TBD | ❓ TBD (pending release) |
| GPT-OSS 120B | ❓ TBD | ❓ TBD | ❓ TBD | ❓ TBD (pending release) |

**Recommendation**: Use **Chat Completion API** for all models (vLLM applies correct template automatically)

---

## Testing Protocol

### Phase 0: Local Testing (Per Model)

**Test 1: Basic Generation**
```python
# Test that model loads and generates text
response = query_model(
    prompt="What is 2+2?",
    temperature=0.7,
    max_tokens=10
)
assert response is not None
assert len(response) > 0
```

**Test 2: Logprobs Extraction**
```python
# Test that logprobs are returned
response = query_model(
    prompt="Answer yes or no: Is the sky blue?",
    temperature=1.0,
    max_tokens=1,
    logprobs=10
)
assert 'logprobs' in response
assert 'yes' in response['logprobs'] or 'Yes' in response['logprobs']
```

**Test 3: Percentage Token Recognition**
```python
# Test that percentage tokens are recognized
response = query_model(
    prompt="What percentage? Answer with 0%, 10%, 20%, etc.",
    temperature=1.0,
    max_tokens=1,
    logprobs=15
)
# Check that tokens like "0%", "10%" appear in logprobs
percentage_tokens = [t for t in response['logprobs'] if '%' in t]
assert len(percentage_tokens) >= 5
```

**Test 4: Study 1 Sample Prompt**
```python
# Test with actual Study 1 prompt structure
prompt = construct_study1_prompt(sample_row)
response = query_model(
    prompt=prompt,
    temperature=0.7,
    max_tokens=100
)
# Verify response is reasonable
assert len(response) > 10
assert len(response) < 200
```

**Test 5: Chat Template (If Using Chat API)**
```python
# Test with messages format
response = query_model_chat(
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"}
    ],
    temperature=0.7,
    max_tokens=10
)
assert response is not None
```

### Phase 1: NRP Pilot Testing

**Deploy one model (Gemma-2B) and run**:
- [ ] 10 trials from Study 1 Experiment 1
- [ ] 10 trials from Study 1 Experiment 2
- [ ] 10 trials from Study 2 Experiment 1
- [ ] 10 trials from Study 2 Experiment 2

**Validate**:
- [ ] All queries complete successfully (>90% success rate)
- [ ] Logprobs extracted correctly
- [ ] Output format is correct
- [ ] No unexpected errors or crashes

---

## Prompt Validation Checklist

Before running full experiments, validate each model:

### Gemma-2 (2B/9B/27B)
- [ ] Chat template tested and working
- [ ] System prompt (if using) applied correctly
- [ ] Percentage tokens (`0%`-`100%`) recognized in logprobs
- [ ] Yes/no tokens recognized in logprobs
- [ ] Stop tokens configured correctly
- [ ] Sample Study 1 responses look reasonable
- [ ] Sample Study 2 responses look reasonable

### Llama-3 70B
- [ ] HuggingFace token configured
- [ ] Model downloads successfully
- [ ] Chat template tested and working
- [ ] System prompt (if using) applied correctly
- [ ] Percentage tokens recognized in logprobs
- [ ] Yes/no tokens recognized in logprobs
- [ ] Stop tokens configured correctly
- [ ] Sample Study 1 responses look reasonable
- [ ] Sample Study 2 responses look reasonable

### GPT-OSS 20B
- [ ] Model available from OpenAI
- [ ] vLLM supports MoE architecture
- [ ] Chat template documented and tested
- [ ] `--trust-remote-code` flag needed? (test)
- [ ] Percentage tokens recognized in logprobs
- [ ] Yes/no tokens recognized in logprobs
- [ ] Stop tokens configured correctly
- [ ] Sample responses look reasonable

### GPT-OSS 120B
- [ ] Model available from OpenAI
- [ ] vLLM supports MoE architecture
- [ ] Chat template documented and tested
- [ ] `--trust-remote-code` flag needed? (test)
- [ ] Percentage tokens recognized in logprobs
- [ ] Yes/no tokens recognized in logprobs
- [ ] Stop tokens configured correctly
- [ ] Sample responses look reasonable

---

## Common Issues & Solutions

### Issue: Logprobs don't include expected tokens

**Symptom**: Requesting logprobs for `["0%", "10%", ..., "100%"]` but they don't appear

**Possible Causes**:
1. Tokens are multi-token (e.g., `0` + `%`)
2. Need higher `top_logprobs` value
3. Model's vocabulary doesn't include these exact tokens

**Solutions**:
1. Increase `top_logprobs` to 50
2. Parse multi-token sequences
3. Use alternative format: "0 percent" instead of "0%"
4. Check model's tokenizer vocabulary

### Issue: Chat template not applied

**Symptom**: Responses include template tokens like `<end_of_turn>`

**Possible Causes**:
1. Using Completion API without manual template
2. vLLM not recognizing model's chat template
3. Stop tokens not configured

**Solutions**:
1. Use Chat Completion API instead
2. Manually apply template in Completion API
3. Configure stop tokens: `stop=["<end_of_turn>", "<|eot_id|>"]`

### Issue: Responses are too long/short

**Symptom**: Getting single words when expecting sentences, or essays when expecting words

**Possible Causes**:
1. `max_tokens` set incorrectly
2. Temperature too high/low
3. No stop tokens configured

**Solutions**:
1. Adjust `max_tokens`: 1 for Exp 2, 50-100 for Exp 1
2. Use T=0.7 for Exp 1, T=1.0 for Exp 2
3. Configure model-specific stop tokens

### Issue: System prompt ignored

**Symptom**: Model doesn't follow system instructions

**Possible Causes**:
1. Model not trained with system prompts
2. Chat template doesn't support system role
3. Using Completion API without including system in template

**Solutions**:
1. Test with/without system prompt
2. Include system message in user prompt if unsupported
3. Use Chat Completion API which handles system role

---

## Final Recommendations

### For Phase 0 (Local Testing)

1. **Start with Gemma-2B**: Smallest, fastest to iterate
2. **Use Chat Completion API**: Let vLLM handle templates
3. **Test both experiments**: Validate both Exp 1 and Exp 2 work
4. **Document token findings**: Record how each model tokenizes percentages

### For Phase 1 (NRP Pilot)

1. **Deploy Gemma-2B first**: Test on NRP infrastructure
2. **Run 10-trial pilot**: Small sample to catch issues
3. **Validate output quality**: Manually review responses
4. **Adjust prompts if needed**: Based on pilot results

### For Phase 4 (Full Production)

1. **Use consistent prompting**: Same format across all models for comparability
2. **Log all configurations**: Temperature, max_tokens, stop_tokens per model
3. **Monitor success rates**: Track API errors and retries
4. **Save raw API responses**: For debugging and reprocessing

---

## Reasoning Model vs Non-Reasoning Model Strategy

### Problem

Instruction-tuned models (Gemma-2, Llama-3) tend to produce verbose Chain-of-Thought (CoT) reasoning even when not requested, leading to lengthy responses with explanations before the actual answer.

Example unwanted output:
```
**Reasoning:**
* We know that 3 students took the exam.
* Mark has looked at 2 exams and found 2 with passing grades.
...
**Answer:**
The answer is **all 3** exams have passing grades.
```

### Solution: Model-Aware Prompting

**Implemented via `src/utils/model_config.py`**:

1. **Model Classification**:
   - **Reasoning models** (DeepSeek R1, o1, o3): `is_reasoning_model=True`
   - **Non-reasoning models** (Gemma-2, Llama-3): `is_reasoning_model=False`

2. **Prompting Strategy**:
   - **Non-reasoning models**: Add explicit instruction "Answer directly without explanation:"
   - **Reasoning models**: Request reasoning "Please provide your reasoning and answer."

3. **Parameter Tuning**:
   - **Non-reasoning models**:
     - `temperature=0.3` (more deterministic)
     - `max_tokens=100` for text, `50` for structured
   - **Reasoning models**:
     - `temperature=0.7` (allow exploration)
     - `max_tokens=500+` for text (allow full reasoning traces)

4. **Reasoning Trace Capture**:
   - For reasoning models, capture full trace in separate JSONL file
   - Link to main results via `result_id`

### Configuration Example

From `model_config.py`:
```python
"google/gemma-2-2b-it": ModelConfig(
    is_reasoning_model=False,
    use_direct_answer_prompt=True,  # Suppress CoT
    max_tokens_text=100,  # Limit verbosity
    temperature_text=0.3,  # More deterministic
)

"deepseek/deepseek-r1": ModelConfig(
    is_reasoning_model=True,
    use_direct_answer_prompt=False,  # Let it reason!
    max_tokens_text=2000,  # Allow full traces
    temperature_text=0.7,
)
```

### Prompt Examples

**Study 1 Exp 1 - Non-reasoning model**:
```
{scenario}

{prior_question}

{speech}

{posterior_question}

Answer directly without explanation:
```

**Study 1 Exp 1 - Reasoning model**:
```
{scenario}

{prior_question}

{speech}

{posterior_question}

Please provide your reasoning and answer.
```

**Study 2 Exp 1 - Non-reasoning model**:
```
You are Emma. If Emma wanted to make Wendy feel good, rather than give informative feedback.

The quality of Wendy's work is: 0 hearts

Respond to Wendy's question using ONLY this exact format (no explanation):
"It [was/wasn't] [terrible/bad/good/amazing]"

Your response:
```

## Open Questions to Resolve

1. **Chat API vs Completion API**: Which to use? (Recommend: Chat API for better system message support)
2. **System prompt text**: Exact wording? (Implemented in model_config.py, test effectiveness)
3. **Temperature values**: Using 0.3 for non-reasoning text, 1.0 for probabilities - validate effectiveness
4. **Max tokens**: 100 for Exp 1, 1 for Exp 2, 50 for structured - adjust based on test results
5. **Stop tokens**: Configured per model in model_config.py
6. **Percentage tokenization**: Test each model, document findings
7. **GPT-OSS specifics**: Wait for release, update this doc
8. **Reasoning suppression effectiveness**: Test with Gemma-2B to validate direct answer prompting works

---

## Document Status

**Last Updated**: 2025-01-16  
**Status**: Draft - Pending testing and decisions  
**Next Steps**: 
1. Test Gemma-2B locally (Phase 0)
2. Document percentage tokenization findings
3. Make final decisions on system prompts, API type
4. Update after GPT-OSS release
5. Add actual test results to this document

---

**See Also**:
- `plans/STUDY1_EXP2_CLARIFICATION.md` - Probability extraction methodology
- `plans/DECISION_CHECKLIST.md` - Operational decisions tracker
- `CLAUDE.md` - AI assistant guide for project navigation
