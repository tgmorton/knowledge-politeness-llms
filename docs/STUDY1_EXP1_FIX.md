# Study 1 Experiment 1 Fix - Knowledge Question Collection

**Status:** 🔴 NOT IMPLEMENTED
**Priority:** HIGH
**Created:** 2025-11-20

---

## Problem Statement

Study 1 Experiment 1 currently collects only ONE response per trial (the quantity question), but the input data contains a `knowledgeQ` field that should also be asked.

**Current behavior:**
- Asks: "Now how many of the 3 exams do you think have passing grades?" (`speachQ`)
- Collects: Single response (e.g., "2")

**Expected behavior:**
- Ask BOTH questions:
  1. "Now how many of the 3 exams do you think have passing grades?" (`speachQ`)
  2. "Do you think Mark knows exactly how many of the 3 exams have passing grades?" (`knowledgeQ`)
- Collect: Two separate responses

---

## Why This Matters

The knowledge attribution question is central to Study 1's research questions. Without it, we cannot analyze:
- Whether models distinguish between their own beliefs and agent knowledge
- How evidence presentation affects knowledge attribution
- Comparison between text responses (Exp 1) and probability distributions (Exp 2)

---

## Required Code Changes

### 1. Update Prompt Construction

**File:** `src/utils/prompts.py`

**Current code** (lines 40-56):
```python
def construct_study1_exp1_prompt(trial: Dict, model_name: str = None) -> str:
    prompt = f"""{setup}

{trial['priorQ']}

{trial['speach']}

{trial['speachQ']}"""

    return prompt
```

**New code:**
```python
def construct_study1_exp1_prompt(trial: Dict, model_name: str = None) -> str:
    # Ask both quantity and knowledge questions
    prompt = f"""{setup}

{trial['priorQ']}

{trial['speach']}

{trial['speachQ']}

Please answer with just a number (0, 1, 2, or 3).

After that, answer this question:

{trial['knowledgeQ']}

Please answer with just "yes" or "no"."""

    return prompt
```

### 2. Update Response Parsing

**File:** `src/query_study1_exp1.py`

**Current code** (lines ~60-80):
```python
result = {
    'participant_id': trial['participant_id'],
    'story_shortname': trial['story_shortname'],
    # ... other fields ...
    'response': response.text.strip(),  # Single response
    'result_id': response.result_id,
    'model_name': model_name,
    'timestamp': datetime.now().isoformat(),
}
```

**New code:**
```python
# Parse response into two parts
response_text = response.text.strip()

# Split on newline or common separators
# Expected format: "2\nyes" or "2. Yes" or similar
response_parts = parse_two_responses(response_text)

result = {
    'participant_id': trial['participant_id'],
    'story_shortname': trial['story_shortname'],
    # ... other fields ...
    'response_quantity': response_parts['quantity'],  # e.g., "2"
    'response_knowledge': response_parts['knowledge'],  # e.g., "yes"
    'response_raw': response_text,  # Keep raw for debugging
    'result_id': response.result_id,
    'model_name': model_name,
    'timestamp': datetime.now().isoformat(),
}
```

### 3. Add Response Parser Function

**File:** `src/query_study1_exp1.py` (new function)

```python
def parse_two_responses(response_text: str) -> Dict[str, str]:
    """
    Parse model response containing two answers:
    1. Quantity answer (0, 1, 2, or 3)
    2. Knowledge answer (yes or no)

    Handles various formats:
    - "2\nyes"
    - "2. Yes"
    - "Answer: 2\nKnowledge: yes"
    - etc.
    """
    lines = response_text.strip().split('\n')

    # Extract quantity (look for digit 0-3)
    quantity = None
    for line in lines:
        match = re.search(r'\b([0-3])\b', line)
        if match:
            quantity = match.group(1)
            break

    # Extract knowledge (look for yes/no)
    knowledge = None
    for line in lines:
        if re.search(r'\byes\b', line, re.IGNORECASE):
            knowledge = "yes"
            break
        elif re.search(r'\bno\b', line, re.IGNORECASE):
            knowledge = "no"
            break

    return {
        'quantity': quantity or "PARSE_ERROR",
        'knowledge': knowledge or "PARSE_ERROR"
    }
```

### 4. Update Output Schema

**Current JSON structure:**
```json
{
  "participant_id": 1,
  "story_shortname": "exams",
  "response": "2",
  "model_name": "gemma-2b-rtx3090",
  ...
}
```

**New JSON structure:**
```json
{
  "participant_id": 1,
  "story_shortname": "exams",
  "response_quantity": "2",
  "response_knowledge": "yes",
  "response_raw": "2\nyes",
  "model_name": "gemma-2b-rtx3090",
  ...
}
```

### 5. Update Validation

**File:** `src/utils/validation.py` (if it exists)

Add validation for both responses:
- `response_quantity`: Must be "0", "1", "2", "3", or "PARSE_ERROR"
- `response_knowledge`: Must be "yes", "no", or "PARSE_ERROR"
- Track parse error rates

---

## Implementation Steps

1. **Backup current data**
   ```bash
   cp -r outputs/results outputs/results_BACKUP_$(date +%Y%m%d)
   ```

2. **Update code files**
   - [ ] `src/utils/prompts.py` - Add knowledge question to prompt
   - [ ] `src/query_study1_exp1.py` - Add response parser
   - [ ] `src/query_study1_exp1.py` - Update result dictionary
   - [ ] Add tests for response parser

3. **Test with one model** (5 trials)
   ```bash
   ./scripts/run_local_exp1.sh gemma-2b-rtx3090 5
   ```

4. **Validate output**
   - Check that both `response_quantity` and `response_knowledge` fields exist
   - Review parse error rate
   - Manually inspect a few responses

5. **Re-run all 4 models** (full 300 trials each)
   - Gemma-2B
   - Gemma-9B
   - Llama-3B
   - Llama-8B

6. **Update CSV converter**
   - Update `scripts/convert_results_to_csv.py` to handle new schema

---

## Testing Plan

### Test Cases

**Input trial:**
```python
{
  'speachQ': 'Now how many of the 3 exams do you think have passing grades?',
  'knowledgeQ': 'Do you think Mark knows exactly how many of the 3 exams have passing grades?',
  'access': 2,
  'observe': 2
}
```

**Expected responses to parse correctly:**

| Model Response | Parsed Quantity | Parsed Knowledge |
|----------------|----------------|------------------|
| "2\nyes" | "2" | "yes" |
| "I think 2\nYes, Mark knows" | "2" | "yes" |
| "Answer: 2\nKnowledge: No" | "2" | "no" |
| "2 exams\nNo" | "2" | "no" |
| "The answer is 2. Yes." | "2" | "yes" |

**Edge cases to handle:**
- Model answers in wrong order
- Model provides only one answer
- Model rambles without clear yes/no
- Model uses "probably", "maybe", "uncertain" instead of yes/no

---

## Impact Assessment

### Data Re-Collection Required

**Models already tested (need to re-run):**
- ✅ Gemma-2B (300 trials)
- ✅ Gemma-9B (300 trials)
- ✅ Llama-3B (300 trials)
- ✅ Llama-8B (300 trials)

**Time estimate:** ~4 hours total (1 hour per model)

**Models not yet tested (can use new version):**
- Gemma-27B
- Llama-70B
- DeepSeek-R1 70B

### Analysis Impact

**Current analysis scripts will break** because they expect:
- `response` field → need to update to use `response_quantity` and `response_knowledge`

**Update needed:**
- `scripts/convert_results_to_csv.py`
- `analysis/analyze_exp1_results.R`

---

## Alternative Approaches

### Option 1: Sequential Prompts (Recommended)

Ask questions one at a time:

**Prompt 1:**
```
{story_setup}
{priorQ}
{speach}
{speachQ}

Answer with just a number (0, 1, 2, or 3):
```

**Prompt 2:**
```
{knowledgeQ}

Answer with just "yes" or "no":
```

**Pros:**
- Cleaner parsing (one answer per response)
- Less ambiguity
- 2 API calls per trial = 600 total calls (still reasonable)

**Cons:**
- 2x API calls (but Study 1 is only 300 trials, so 600 calls is fine)

### Option 2: Combined Prompt with Structured Output

Use JSON mode or structured output:

```
{story_setup}
{priorQ}
{speach}
{speachQ}
{knowledgeQ}

Respond in JSON format:
{
  "quantity": <number 0-3>,
  "knowledge": <"yes" or "no">
}
```

**Pros:**
- Single API call
- Easy parsing

**Cons:**
- Not all models support JSON mode well
- May not reflect natural reasoning

### Option 3: Post-hoc Extraction (NOT RECOMMENDED)

Use DeepSeek API to extract knowledge answer from existing quantity responses.

**Cons:**
- Less reliable than asking directly
- Adds complexity
- Loses valuable data about how models jointly reason about both questions

---

## Recommendation

**Use Option 1 (Sequential Prompts)** for:
- Clean separation of concerns
- Reliable parsing
- Maintains natural language response quality
- Only 2x API calls (600 total for 300 trials is acceptable)

---

## Related Documents

- `docs/experiment1-grace.md` - Original Study 1 design
- `plans/STUDY1_EXP2_CLARIFICATION.md` - Experiment 2 design (knowledge Q as probability)
- `src/utils/prompts.py` - Current prompt construction
- `src/query_study1_exp1.py` - Current Experiment 1 script

---

## Timeline

**If implemented:**
- Code changes: 2-3 hours
- Testing: 1 hour
- Re-run 4 models: 4 hours
- Analysis updates: 1 hour

**Total: ~8 hours of work + 4 hours compute time**

---

## Notes

- This issue was discovered 2025-11-20 during analysis pipeline development
- Current data (quantity responses only) is still valuable but incomplete
- Consider implementing before running large models (27B, 70B) to avoid re-running expensive experiments

