# Study 1 Experiment 2 - Probability Extraction Approach

## Key Design Decision: Uncertainty Quantification via Probability Distributions

### Overview

For Study 1 Experiment 2, we capture the **model's uncertainty about each state** by extracting probability distributions over percentage values (0%, 10%, 20%, ..., 100%) for each of the 4 possible states (0, 1, 2, 3).

This approach goes beyond simple discrete state probabilities and enables rich uncertainty analysis including density plots, variance measures, and model calibration studies.

---

## The Question

For a trial like:
> "Students in the introductory bio class almost always have passing grades on the exam. Mark's 3 intro bio students took an exam yesterday. Mark tells you: 'I have looked at 2 of the 3 exams. 2 of the exams have passing grades.'"

Instead of just asking:
- ❌ "What's the most likely number?" → Get P(0), P(1), P(2), P(3)

We ask **four separate questions**:
1. ✅ "What percentage probability do you assign that **exactly 0** of the 3 exams have passing grades?" → Get P(0%), P(10%), ..., P(100%)
2. ✅ "What percentage probability do you assign that **exactly 1** of the 3 exams have passing grades?" → Get P(0%), P(10%), ..., P(100%)
3. ✅ "What percentage probability do you assign that **exactly 2** of the 3 exams have passing grades?" → Get P(0%), P(10%), ..., P(100%)
4. ✅ "What percentage probability do you assign that **exactly 3** of the 3 exams have passing grades?" → Get P(0%), P(10%), ..., P(100%)

---

## What This Captures

### Example Output for One Trial

```
State 0 Distribution:
  P(0%)  = 0.05  → Model is 5% confident state 0 has 0% probability
  P(10%) = 0.15  → Model is 15% confident state 0 has 10% probability
  P(20%) = 0.25  → Model is 25% confident state 0 has 20% probability (mode)
  P(30%) = 0.20
  ...
  P(100%) = 0.01

State 1 Distribution:
  P(0%)  = 0.01
  P(10%) = 0.05
  ...
  P(60%) = 0.30  → Model thinks state 1 is most likely around 60%
  ...

State 2 Distribution:
  P(0%)  = 0.02
  ...
  P(30%) = 0.40  → High confidence state 2 has ~30% probability
  ...

State 3 Distribution:
  P(0%)  = 0.40  → Very confident state 3 has near-zero probability
  P(10%) = 0.30
  ...
  P(100%) = 0.01
```

### Visualizations Enabled

1. **Density Heatmap**: 4×11 grid showing probability density for each state
```
        0%   10%   20%   30%   40%   50%   60%   70%   80%   90%  100%
State 0 [0.05][0.15][0.25][0.20][0.15][0.10][0.05][0.03][0.01][0.00][0.01]
State 1 [0.01][0.05][0.10][0.15][0.20][0.20][0.15][0.10][0.03][0.01][0.00]
State 2 [0.02][0.08][0.20][0.40][0.20][0.07][0.02][0.01][0.00][0.00][0.00]
State 3 [0.40][0.30][0.15][0.08][0.04][0.02][0.01][0.00][0.00][0.00][0.00]
```

2. **Uncertainty Profile**: For each state, plot probability mass over percentage bins
```
State 2 Uncertainty:
     ▁▃█▆▂▁
  0% 10% 20% 30% 40% 50% 60%
  
  → Shows model is most confident state 2 has ~30% probability
  → But has non-trivial uncertainty (spread from 10%-50%)
```

3. **Model Calibration**: Compare model's confidence to actual human responses

---

## Summary Statistics Computed

For each state (0, 1, 2, 3), we compute:

### 1. Mean Expected Percentage
```python
state0_mean = sum(P(i%) * i for i in [0, 10, 20, ..., 100])
```
Example: `state2_mean = 32.5` → Model expects state 2 to have ~32.5% probability

### 2. Standard Deviation
```python
state0_std = sqrt(sum(P(i%) * (i - mean)^2 for i in [0, 10, ..., 100]))
```
Example: `state2_std = 12.3` → High uncertainty (wide distribution)

### 3. Entropy
```python
state0_entropy = -sum(P(i%) * log2(P(i%)) for i in [0, 10, ..., 100])
```
Example: `state3_entropy = 0.8` → Low entropy (confident, peaked distribution)

---

## Data Schema

### Output CSV Structure

**File**: `{model}_study1_probabilities.csv`

**68 columns total**:
- 9 original columns (participant_id, story_shortname, access, observe, etc.)
- 44 probability columns (4 states × 11 percentage bins)
  - `state0_prob_0`, `state0_prob_10`, ..., `state0_prob_100`
  - `state1_prob_0`, `state1_prob_10`, ..., `state1_prob_100`
  - `state2_prob_0`, `state2_prob_10`, ..., `state2_prob_100`
  - `state3_prob_0`, `state3_prob_10`, ..., `state3_prob_100`
- 12 summary statistics columns
  - `state0_mean`, `state0_std`, `state0_entropy`
  - `state1_mean`, `state1_std`, `state1_entropy`
  - `state2_mean`, `state2_std`, `state2_entropy`
  - `state3_mean`, `state3_std`, `state3_entropy`
- 3 knowledge columns
  - `prob_knowledge_yes`, `prob_knowledge_no`, `entropy_knowledge`

---

## Implementation Details

### Prompt Template

```python
def construct_state_percentage_prompt(row, state):
    return f"""Context: {row['story_setup']}

{row['speach']}

Question: What percentage probability do you assign that exactly {state} of the 3 {items} {property}?

Express your answer as a percentage from 0% to 100%, in increments of 10% (e.g., 0%, 10%, 20%, etc.).

Answer with just the percentage:"""
```

### vLLM API Call

```python
for state in [0, 1, 2, 3]:
    prompt = construct_state_percentage_prompt(row, state)
    
    response = client.post(
        f"{model_endpoint}/v1/completions",
        json={
            "model": "model_name",
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 1.0,  # Unbiased sampling
            "logprobs": 15       # Capture all percentage tokens
        }
    )
    
    # Extract probabilities for ["0%", "10%", "20%", ..., "100%"]
    probs = extract_token_probabilities(response, percentage_tokens)
    
    # Normalize using softmax
    normalized_probs = softmax(probs)
```

### Processing Flow

```
For each trial (300 trials):
  For each model (N models):
    For each state (4 states):
      1. Construct prompt asking about that state's probability
      2. Query model with temperature=1.0
      3. Extract logprobs for 11 percentage tokens
      4. Normalize to valid probability distribution
      5. Store in output CSV
    
    For knowledge question (1 query):
      1. Extract P(yes) and P(no)
      2. Store in output CSV
```

**Total queries per trial**: 4 (states) + 1 (knowledge) = 5 queries

**Total queries for Study 1**: 300 trials × 5 queries × N models = **1,500×N queries**

---

## Computational Cost

### Comparison to Original Plan

**Original approach (discrete states)**:
- 1 query per trial for states: "Which state (0,1,2,3)?"
- 1 query per trial for knowledge: "Yes or no?"
- Total: 2 queries × 300 trials × N models = **600×N queries**

**New approach (probability distributions)**:
- 4 queries per trial for states (one per state)
- 1 query per trial for knowledge
- Total: 5 queries × 300 trials × N models = **1,500×N queries**

**Increase**: **2.5× more queries** than originally planned

### Time Estimates

Assuming:
- Average inference time: 2 seconds per query
- Sequential processing within trial

**Per model**:
- 1,500 queries × 2 sec = 3,000 seconds = **50 minutes**

**All models in parallel** (5 models):
- **50 minutes per model** (running simultaneously)

**Acceptable given the value**: Rich uncertainty data worth the extra compute

---

## Benefits of This Approach

### 1. Uncertainty Quantification
- Not just "What's most likely?" but "How certain are you?"
- Captures full probability density, not just argmax

### 2. Model Calibration Analysis
- Compare model confidence to human responses
- Identify over-confident vs under-confident models
- Assess whether model uncertainty correlates with task difficulty

### 3. Variance Across Conditions
- Analyze how uncertainty changes with access/observe conditions
- Test hypotheses about when models are more/less confident

### 4. Comparison to Human Data
- Human subjects likely have their own uncertainty distributions
- Can compare shape of distributions, not just point estimates

### 5. Novel Research Contribution
- Most LLM studies only report argmax responses
- Full probability distributions are rarely analyzed
- Enables new insights into model reasoning under uncertainty

---

## Alternative Approaches Considered

### Option A: Single Query with Joint Distribution
**Prompt**: "What are P(0), P(1), P(2), P(3)?"

**Issues**:
- Hard to extract logprobs for structured output
- Model may not follow format
- Doesn't capture uncertainty about each probability

### Option B: Logprobs Over State Tokens
**Prompt**: "How many items have the property?"
**Extract**: P("0"), P("1"), P("2"), P("3")

**Issues**:
- Only gives one number per state (e.g., P(state=2) = 0.35)
- No information about model's uncertainty in that estimate
- Can't distinguish "confidently 35%" from "unsure, but ~35% on average"

### Option C: Current Approach ✓
**4 separate prompts, one per state**
**Extract**: Full distribution for each state

**Advantages**:
- Rich uncertainty information
- Clean extraction (just percentage tokens)
- Enables density plots and calibration analysis
- Straightforward to implement

---

## Validation Checks

After extraction, validate:

1. **Each state distribution sums to 1.0** (within tolerance)
   ```python
   assert 0.99 <= sum(state0_probs) <= 1.01
   ```

2. **All probabilities in [0, 1]**
   ```python
   assert all(0 <= p <= 1 for p in all_probs)
   ```

3. **Summary statistics are consistent**
   ```python
   calculated_mean = sum(p * val for p, val in zip(probs, [0,10,...,100]))
   assert abs(calculated_mean - stored_mean) < 0.1
   ```

4. **No NaN or missing values**
   ```python
   assert not any(pd.isna(row[prob_cols]))
   ```

---

## R Analysis Integration

### Load Data
```r
library(tidyverse)

# Load probability data
probs_df <- read_csv("gemma_2b_study1_probabilities.csv")

# Reshape to long format for visualization
probs_long <- probs_df %>%
  pivot_longer(
    cols = starts_with("state"),
    names_to = c("state", "metric"),
    names_pattern = "state(\\d)_(.*)",
    values_to = "value"
  )
```

### Plot Uncertainty Heatmap
```r
# Filter to probability columns only
prob_matrix <- probs_df %>%
  select(participant_id, access, observe, 
         matches("state\\d_prob_\\d+")) %>%
  pivot_longer(
    cols = matches("state\\d_prob_\\d+"),
    names_to = c("state", "percentage"),
    names_pattern = "state(\\d)_prob_(\\d+)",
    values_to = "probability"
  )

# Heatmap
ggplot(prob_matrix, aes(x = percentage, y = state, fill = probability)) +
  geom_tile() +
  facet_grid(access ~ observe) +
  scale_fill_viridis_c() +
  labs(title = "Model Uncertainty by State and Condition")
```

### Analyze Uncertainty Patterns
```r
# Compare entropy across conditions
entropy_df <- probs_df %>%
  select(access, observe, 
         state0_entropy, state1_entropy, 
         state2_entropy, state3_entropy) %>%
  pivot_longer(
    cols = contains("entropy"),
    names_to = "state",
    values_to = "entropy"
  )

# ANOVA: Does uncertainty vary by access/observe?
model <- aov(entropy ~ access * observe * state, data = entropy_df)
summary(model)
```

---

## Summary

This approach provides **rich uncertainty quantification** at the cost of **2.5× more API calls**. The computational cost is justified by the research value:

- ✅ Full probability density over model beliefs
- ✅ Uncertainty measures (mean, std, entropy)
- ✅ Model calibration analysis
- ✅ Novel contribution to LLM uncertainty literature
- ✅ Direct comparison to human uncertainty patterns

The implementation is straightforward and the data structure (68 columns) is manageable for analysis in R/Python.

**Recommendation**: Proceed with this approach. The extra compute time (~50 min per model) is acceptable given the research insights enabled.
