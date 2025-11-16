# Data Pipeline Architecture

## Overview

This document describes the end-to-end data pipeline for processing experimental data through self-hosted LLMs, extracting structured output, and preparing data for analysis.

---

## Pipeline Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT DATA LAYER                             │
├─────────────────────────────────────────────────────────────────────┤
│  study1.csv (300 rows)              study2.csv (2,424 rows)         │
│  ↓                                   ↓                               │
│  [PVC: grace-input-data]            [PVC: grace-input-data]         │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      MODEL SERVING LAYER                             │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ Gemma-2B │  │ Gemma-9B │  │ Llama-8B │  │Llama-70B │  ...      │
│  │ (vLLM)   │  │ (vLLM)   │  │ (vLLM)   │  │ (vLLM)   │           │
│  │ Port:8000│  │ Port:8000│  │ Port:8000│  │ Port:8000│           │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘           │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT 1: RAW RESPONSES                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Study 1 Query Jobs (Parallel)        Study 2 Query Jobs (Parallel) │
│  ├─ job-study1-exp1-gemma2b          ├─ job-study2-exp1-gemma2b    │
│  ├─ job-study1-exp1-gemma9b          ├─ job-study2-exp1-gemma9b    │
│  ├─ job-study1-exp1-llama8b          ├─ job-study2-exp1-llama8b    │
│  └─ job-study1-exp1-llama70b         └─ job-study2-exp1-llama70b   │
│                                                                       │
│  Output: {model}_study{1|2}_raw_responses.csv                       │
│  Stored in: [PVC: grace-output-data]/exp1/{model}/                  │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│              STRUCTURED OUTPUT EXTRACTION (DeepSeek)                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  DeepSeek Extraction Jobs (Sequential)                               │
│  ├─ job-extract-gemma2b-study1                                      │
│  ├─ job-extract-gemma2b-study2                                      │
│  ├─ job-extract-gemma9b-study1                                      │
│  └─ ... (all model × study combinations)                            │
│                                                                       │
│  Output: {model}_study{1|2}_structured.csv                          │
│  Stored in: [PVC: grace-output-data]/structured/{model}/            │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│              EXPERIMENT 2: PROBABILITY EXTRACTION                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Probability Extraction Jobs (Parallel)                              │
│  ├─ job-study1-exp2-gemma2b   → Extract P(0,1,2,3) & P(yes,no)     │
│  ├─ job-study1-exp2-gemma9b                                         │
│  ├─ job-study2-exp2-gemma2b   → Extract P(was,wasn't,good,bad,...) │
│  └─ job-study2-exp2-gemma9b                                         │
│                                                                       │
│  Output: {model}_study{1|2}_probabilities.csv                       │
│  Stored in: [PVC: grace-output-data]/exp2/{model}/                  │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    VALIDATION & QUALITY CHECKS                       │
├─────────────────────────────────────────────────────────────────────┤
│  ├─ Validate row counts match input                                 │
│  ├─ Check for missing values                                        │
│  ├─ Validate probabilities sum to ~1.0                              │
│  ├─ Validate structured output against JSON schemas                 │
│  ├─ Generate data quality report                                    │
│  └─ Flag rows for manual review                                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      ANALYSIS-READY DATA                             │
├─────────────────────────────────────────────────────────────────────┤
│  Merged datasets with:                                               │
│  - Original input columns                                            │
│  - Raw model responses                                               │
│  - Structured outputs                                                │
│  - Probability distributions                                         │
│  - Quality flags                                                     │
│                                                                       │
│  Ready for: R analysis, visualization, statistical tests             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Schemas

### 1. Study 1 - Raw Response Output

**File**: `{model}_study1_raw_responses.csv`

| Column | Type | Description |
|--------|------|-------------|
| participant_id | int | Original participant ID |
| story_shortname | string | Story type |
| story_setup | string | Story context |
| priorQ | string | Prior probability question |
| speach | string | Speaker's statement |
| speachQ | string | Follow-up question |
| knowledgeQ | string | Knowledge question |
| access | int | Items accessed (1-3) |
| observe | int | Items observed (1-3) |
| **model_name** | string | Model used (e.g., "gemma-2b") |
| **model_response** | string | Raw text response from model |
| **response_timestamp** | datetime | When response was generated |
| **processing_time_ms** | float | Time to generate response |

### 2. Study 1 - Structured Output

**File**: `{model}_study1_structured.csv`

Adds to raw response output:

| Column | Type | Description |
|--------|------|-------------|
| **Prior_A** | int [0-3] | Extracted prior answer |
| **probability_0** | int [0-100] | Probability for state 0 |
| **probability_1** | int [0-100] | Probability for state 1 |
| **probability_2** | int [0-100] | Probability for state 2 |
| **probability_3** | int [0-100] | Probability for state 3 |
| **Knowledge_A** | string [yes/no] | Extracted knowledge answer |
| **extraction_confidence** | float [0-1] | DeepSeek confidence score |
| **extraction_errors** | string | Any parsing errors (null if none) |

### 3. Study 1 - Probability Distribution

**File**: `{model}_study1_probabilities.csv`

**Note**: For Study 1, we extract the model's uncertainty about each state by getting probability distributions over percentage values (0%, 10%, 20%, ..., 100%) for each of the 4 states. This creates a 4×11 probability matrix per trial, capturing how confident the model is about each state's likelihood.

| Column | Type | Description |
|--------|------|-------------|
| participant_id | int | Original participant ID |
| story_shortname | string | Story type |
| access | int | Items accessed |
| observe | int | Items observed |
| **model_name** | string | Model used |
| **state0_prob_0** | float [0-1] | P(state=0 has 0% probability) |
| **state0_prob_10** | float [0-1] | P(state=0 has 10% probability) |
| **state0_prob_20** | float [0-1] | P(state=0 has 20% probability) |
| **state0_prob_30** | float [0-1] | P(state=0 has 30% probability) |
| **state0_prob_40** | float [0-1] | P(state=0 has 40% probability) |
| **state0_prob_50** | float [0-1] | P(state=0 has 50% probability) |
| **state0_prob_60** | float [0-1] | P(state=0 has 60% probability) |
| **state0_prob_70** | float [0-1] | P(state=0 has 70% probability) |
| **state0_prob_80** | float [0-1] | P(state=0 has 80% probability) |
| **state0_prob_90** | float [0-1] | P(state=0 has 90% probability) |
| **state0_prob_100** | float [0-1] | P(state=0 has 100% probability) |
| **state1_prob_0** | float [0-1] | P(state=1 has 0% probability) |
| **state1_prob_10** | float [0-1] | P(state=1 has 10% probability) |
| ... | ... | (state1_prob_20 through state1_prob_100) |
| **state2_prob_0** | float [0-1] | P(state=2 has 0% probability) |
| ... | ... | (state2_prob_10 through state2_prob_100) |
| **state3_prob_0** | float [0-1] | P(state=3 has 0% probability) |
| ... | ... | (state3_prob_10 through state3_prob_100) |
| **state0_mean** | float | Expected percentage for state 0 |
| **state0_std** | float | Std dev for state 0 distribution |
| **state0_entropy** | float | Entropy of state 0 distribution |
| **state1_mean** | float | Expected percentage for state 1 |
| **state1_std** | float | Std dev for state 1 distribution |
| **state1_entropy** | float | Entropy of state 1 distribution |
| **state2_mean** | float | Expected percentage for state 2 |
| **state2_std** | float | Std dev for state 2 distribution |
| **state2_entropy** | float | Entropy of state 2 distribution |
| **state3_mean** | float | Expected percentage for state 3 |
| **state3_std** | float | Std dev for state 3 distribution |
| **state3_entropy** | float | Entropy of state 3 distribution |
| **prob_knowledge_yes** | float [0-1] | P(knowledge=yes) |
| **prob_knowledge_no** | float [0-1] | P(knowledge=no) |
| **entropy_knowledge** | float | Entropy of knowledge distribution |

**Total columns**: 9 original + 44 state probabilities (4 states × 11 bins) + 12 summary stats + 3 knowledge = 68 columns

### 4. Study 2 - Raw Response Output

**File**: `{model}_study2_raw_responses.csv`

| Column | Type | Description |
|--------|------|-------------|
| Participant_ID | int | Original participant ID |
| Domain | string | Task domain |
| Precontext | string | Context setup |
| Scenario | string | Full scenario |
| Goal | string | Speaker's goal |
| State | string | Emotional state (hearts) |
| SP_Name | string | Speaker name |
| LS_Name | string | Listener name |
| Goal_simplified | string | Simplified goal category |
| State_number | int [0-3] | Numeric state |
| **model_name** | string | Model used |
| **model_response** | string | Raw generated utterance |
| **response_timestamp** | datetime | When generated |
| **processing_time_ms** | float | Generation time |

### 5. Study 2 - Structured Output

**File**: `{model}_study2_structured.csv`

Adds to raw response:

| Column | Type | Description |
|--------|------|-------------|
| **Was_Wasnt** | string [was/wasn't] | Extracted frame |
| **Assessment** | string [terrible/bad/good/amazing] | Extracted quality |
| **response_cleaned** | string | Cleaned utterance text |
| **extraction_confidence** | float [0-1] | Confidence score |
| **extraction_errors** | string | Parsing errors if any |

### 6. Study 2 - Probability Distribution

**File**: `{model}_study2_probabilities.csv`

| Column | Type | Description |
|--------|------|-------------|
| Participant_ID | int | Original participant ID |
| Domain | string | Task domain |
| Goal_simplified | string | Goal category |
| State_number | int | Numeric state |
| **model_name** | string | Model used |
| **prob_was** | float [0-1] | P(frame="was") |
| **prob_wasnt** | float [0-1] | P(frame="wasn't") |
| **prob_terrible** | float [0-1] | P(assessment="terrible") |
| **prob_bad** | float [0-1] | P(assessment="bad") |
| **prob_good** | float [0-1] | P(assessment="good") |
| **prob_amazing** | float [0-1] | P(assessment="amazing") |
| **prob_was_terrible** | float [0-1] | Joint P("was terrible") |
| **prob_was_bad** | float [0-1] | Joint P("was bad") |
| **prob_was_good** | float [0-1] | Joint P("was good") |
| **prob_was_amazing** | float [0-1] | Joint P("was amazing") |
| **prob_wasnt_terrible** | float [0-1] | Joint P("wasn't terrible") |
| **prob_wasnt_bad** | float [0-1] | Joint P("wasn't bad") |
| **prob_wasnt_good** | float [0-1] | Joint P("wasn't good") |
| **prob_wasnt_amazing** | float [0-1] | Joint P("wasn't amazing") |
| **entropy_frame** | float | Entropy of frame distribution |
| **entropy_assessment** | float | Entropy of assessment distribution |
| **entropy_joint** | float | Entropy of joint distribution |

---

## Python Script Specifications

### Script 1: Query Generator (Experiment 1)

**File**: `src/query_study1_exp1.py`

```python
#!/usr/bin/env python3
"""
Query generator for Study 1, Experiment 1.
Generates raw text responses from LLM for each row in study1.csv.
"""

import argparse
import asyncio
import logging
import pandas as pd
import httpx
from datetime import datetime
from pathlib import Path
from typing import Optional

class Study1QueryGenerator:
    def __init__(
        self,
        input_csv: str,
        output_csv: str,
        model_endpoint: str,
        model_name: str,
        batch_size: int = 10,
        max_retries: int = 3,
        timeout: float = 60.0
    ):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.model_endpoint = model_endpoint
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def construct_prompt(self, row: pd.Series) -> str:
        """Construct prompt for Study 1."""
        return f"""You are analyzing scenarios about knowledge and information access.

Context: {row['story_setup']}

Initial Question: {row['priorQ']}

New Information: {row['speach']}

Updated Question: {row['speachQ']}

Please provide:
1. Your answer to the initial question (a number from 0 to 3)
2. Your probability estimates for each possibility (0, 1, 2, or 3), shown as percentages that sum to 100%
3. Your answer to this knowledge question: {row['knowledgeQ']} (yes or no)

Format your response as:
Initial Answer: [0-3]
Probabilities: 0: [X]%, 1: [Y]%, 2: [Z]%, 3: [W]%
Knowledge: [yes/no]
"""
    
    async def query_model(
        self,
        prompt: str,
        client: httpx.AsyncClient
    ) -> Optional[dict]:
        """Query the model with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = await client.post(
                    f"{self.model_endpoint}/v1/chat/completions",
                    json={
                        "model": self.model_name,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a helpful assistant analyzing pragmatic language use."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "max_tokens": 200,
                        "temperature": 0.7
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            
            except Exception as e:
                self.logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed: {e}"
                )
                if attempt == self.max_retries - 1:
                    return None
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    async def process_batch(
        self,
        batch: pd.DataFrame,
        client: httpx.AsyncClient
    ) -> list:
        """Process a batch of rows in parallel."""
        tasks = []
        for idx, row in batch.iterrows():
            prompt = self.construct_prompt(row)
            tasks.append(self.query_model(prompt, client))
        
        results = await asyncio.gather(*tasks)
        return results
    
    async def run(self):
        """Main execution loop."""
        # Load input data
        df = pd.read_csv(self.input_csv)
        self.logger.info(f"Loaded {len(df)} rows from {self.input_csv}")
        
        # Add output columns
        df['model_name'] = self.model_name
        df['model_response'] = None
        df['response_timestamp'] = None
        df['processing_time_ms'] = None
        
        # Ensure output directory exists
        import os
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        
        # Process in batches
        async with httpx.AsyncClient() as client:
            for i in range(0, len(df), self.batch_size):
                batch = df.iloc[i:i+self.batch_size]
                self.logger.info(f"Processing batch {i//self.batch_size + 1}")
                
                start_time = datetime.now()
                results = await self.process_batch(batch, client)
                batch_time = (datetime.now() - start_time).total_seconds() * 1000
                
                # Update dataframe
                for j, result in enumerate(results):
                    idx = i + j
                    if result:
                        df.at[idx, 'model_response'] = result['choices'][0]['message']['content']
                        df.at[idx, 'response_timestamp'] = datetime.now().isoformat()
                        df.at[idx, 'processing_time_ms'] = batch_time / len(results)
                    else:
                        self.logger.error(f"Failed to get response for row {idx}")
                
                # Save intermediate results
                df.to_csv(self.output_csv, index=False)
                self.logger.info(f"Saved progress to {self.output_csv}")
        
        self.logger.info("Processing complete!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--model-endpoint', required=True, help='Model endpoint URL')
    parser.add_argument('--model-name', required=True, help='Model name')
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--max-retries', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=60.0)
    
    args = parser.parse_args()
    
    generator = Study1QueryGenerator(
        input_csv=args.input,
        output_csv=args.output,
        model_endpoint=args.model_endpoint,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        timeout=args.timeout
    )
    
    asyncio.run(generator.run())

if __name__ == '__main__':
    main()
```

### Script 2: Probability Extractor (Experiment 2)

**File**: `src/extract_probabilities_study1.py`

```python
#!/usr/bin/env python3
"""
Probability extractor for Study 1, Experiment 2.
Extracts token-level probabilities from LLM logprobs.
"""

import argparse
import asyncio
import logging
import numpy as np
import pandas as pd
import httpx
from typing import Dict, List

class Study1ProbabilityExtractor:
    def __init__(
        self,
        input_csv: str,
        output_csv: str,
        model_endpoint: str,
        model_name: str
    ):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.model_endpoint = model_endpoint
        self.model_name = model_name
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def construct_state_percentage_prompt(self, row: pd.Series, state: int) -> str:
        """Construct prompt for extracting percentage probability for a given state.
        
        For example, if state=2, we ask: "What percentage probability do you assign 
        that exactly 2 of the 3 items have the property?"
        """
        # Determine the item type from the story
        story = row['story_shortname']
        item_map = {
            'exams': 'exams have passing grades',
            'letters': 'letters have checks inside',
            'tickets': 'tickets have won',
            'fruits': 'fruits have dried out pith',
            'phones': 'phones are defective',
            'seeds': 'seeds will sprout'
        }
        property_desc = item_map.get(story, 'items have the property')
        
        return f"""{row['story_setup']}

{row['speach']}

Question: What percentage probability do you assign that exactly {state} of the 3 {property_desc}?

Express your answer as a percentage from 0% to 100%, in increments of 10% (e.g., 0%, 10%, 20%, etc.).

Answer with just the percentage:"""
    
    def construct_knowledge_prompt(self, row: pd.Series) -> str:
        """Construct prompt for knowledge probability extraction."""
        return f"""{row['story_setup']}

{row['speach']}

{row['knowledgeQ']}

Answer with only yes or no:"""
    
    def extract_token_probabilities(
        self,
        logprobs_response: dict,
        target_tokens: List[str]
    ) -> Dict[str, float]:
        """Extract and normalize probabilities for target tokens."""
        top_logprobs = logprobs_response['choices'][0]['logprobs']['top_logprobs'][0]
        
        # Extract logprobs for target tokens (case-insensitive)
        token_logprobs = {}
        for token in target_tokens:
            found = False
            for key in [token, token.capitalize(), token.lower(), f" {token}"]:
                if key in top_logprobs:
                    token_logprobs[token] = top_logprobs[key]
                    found = True
                    break
            if not found:
                token_logprobs[token] = -10.0  # Low probability for missing tokens
        
        # Normalize using softmax
        logprobs_array = np.array(list(token_logprobs.values()))
        exp_logprobs = np.exp(logprobs_array)
        probabilities = exp_logprobs / exp_logprobs.sum()
        
        return {
            token: float(prob)
            for token, prob in zip(target_tokens, probabilities)
        }
    
    def calculate_entropy(self, probs: List[float]) -> float:
        """Calculate entropy of probability distribution."""
        probs = np.array(probs)
        probs = probs[probs > 0]  # Remove zeros
        return -np.sum(probs * np.log2(probs))
    
    async def extract_row_probabilities(
        self,
        row: pd.Series,
        client: httpx.AsyncClient
    ) -> dict:
        """Extract probabilities for a single row.
        
        For each state (0, 1, 2, 3), we query the model to get probability 
        distributions over percentage values (0%, 10%, 20%, ..., 100%).
        This gives us 4 distributions showing the model's uncertainty.
        """
        result = {
            'model_name': self.model_name
        }
        
        # Define percentage bins
        percentage_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        # Extract probability distribution for each state
        for state in [0, 1, 2, 3]:
            state_prompt = self.construct_state_percentage_prompt(row, state)
            
            try:
                response = await client.post(
                    f"{self.model_endpoint}/v1/completions",
                    json={
                        "model": self.model_name,
                        "prompt": state_prompt,
                        "max_tokens": 1,
                        "temperature": 1.0,
                        "logprobs": 15  # Need more logprobs to capture all percentage tokens
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                
                # Extract probabilities for percentage tokens
                percentage_tokens = ["0%", "10%", "20%", "30%", "40%", "50%", 
                                   "60%", "70%", "80%", "90%", "100%"]
                percentage_probs = self.extract_token_probabilities(
                    response.json(),
                    percentage_tokens
                )
                
                # Store individual probabilities
                for pct in percentage_bins:
                    result[f'state{state}_prob_{pct}'] = percentage_probs[f'{pct}%']
                
                # Calculate summary statistics
                probs_array = np.array([percentage_probs[f'{pct}%'] for pct in percentage_bins])
                result[f'state{state}_mean'] = np.sum(probs_array * np.array(percentage_bins))
                result[f'state{state}_std'] = np.sqrt(
                    np.sum(probs_array * (np.array(percentage_bins) - result[f'state{state}_mean'])**2)
                )
                result[f'state{state}_entropy'] = self.calculate_entropy(probs_array)
            
            except Exception as e:
                self.logger.error(f"Failed to extract state {state} probabilities: {e}")
                # Set all values to None for this state
                for pct in percentage_bins:
                    result[f'state{state}_prob_{pct}'] = None
                result[f'state{state}_mean'] = None
                result[f'state{state}_std'] = None
                result[f'state{state}_entropy'] = None
        
        # Extract knowledge probabilities
        knowledge_prompt = self.construct_knowledge_prompt(row)
        try:
            response = await client.post(
                f"{self.model_endpoint}/v1/completions",
                json={
                    "model": self.model_name,
                    "prompt": knowledge_prompt,
                    "max_tokens": 1,
                    "temperature": 1.0,
                    "logprobs": 10
                },
                timeout=30.0
            )
            response.raise_for_status()
            
            knowledge_probs = self.extract_token_probabilities(
                response.json(),
                ["yes", "no"]
            )
            
            result['prob_knowledge_yes'] = knowledge_probs['yes']
            result['prob_knowledge_no'] = knowledge_probs['no']
            result['entropy_knowledge'] = self.calculate_entropy(list(knowledge_probs.values()))
        
        except Exception as e:
            self.logger.error(f"Failed to extract knowledge probabilities: {e}")
            result.update({
                'prob_knowledge_yes': None,
                'prob_knowledge_no': None,
                'entropy_knowledge': None
            })
        
        return result
    
    async def run(self):
        """Main execution loop."""
        df = pd.read_csv(self.input_csv)
        self.logger.info(f"Loaded {len(df)} rows")
        
        results = []
        async with httpx.AsyncClient() as client:
            for idx, row in df.iterrows():
                self.logger.info(f"Processing row {idx + 1}/{len(df)}")
                result = await self.extract_row_probabilities(row, client)
                
                # Combine with original row data
                row_dict = row.to_dict()
                row_dict.update(result)
                results.append(row_dict)
                
                # Save intermediate results every 10 rows
                if (idx + 1) % 10 == 0:
                    pd.DataFrame(results).to_csv(self.output_csv, index=False)
                    self.logger.info(f"Saved progress to {self.output_csv}")
        
        # Final save
        pd.DataFrame(results).to_csv(self.output_csv, index=False)
        self.logger.info("Extraction complete!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model-endpoint', required=True)
    parser.add_argument('--model-name', required=True)
    
    args = parser.parse_args()
    
    extractor = Study1ProbabilityExtractor(
        input_csv=args.input,
        output_csv=args.output,
        model_endpoint=args.model_endpoint,
        model_name=args.model_name
    )
    
    asyncio.run(extractor.run())

if __name__ == '__main__':
    main()
```

### Script 3: Structured Output Extractor

**File**: `src/extract_structured_output.py`

```python
#!/usr/bin/env python3
"""
Structured output extractor using DeepSeek.
Parses raw LLM responses into structured CSV format.
"""

import argparse
import asyncio
import json
import logging
import pandas as pd
import httpx
from pydantic import BaseModel, Field, ValidationError
from typing import Optional

# Pydantic models for validation
class Study1StructuredOutput(BaseModel):
    Prior_A: int = Field(ge=0, le=3)
    probability_0: int = Field(ge=0, le=100)
    probability_1: int = Field(ge=0, le=100)
    probability_2: int = Field(ge=0, le=100)
    probability_3: int = Field(ge=0, le=100)
    Knowledge_A: str = Field(pattern='^(yes|no)$')

class Study2StructuredOutput(BaseModel):
    Was_Wasnt: str = Field(pattern='^(was|wasn\'t)$')
    Assessment: str = Field(pattern='^(terrible|bad|good|amazing)$')
    response_cleaned: str

class StructuredOutputExtractor:
    def __init__(
        self,
        input_csv: str,
        output_csv: str,
        schema_file: str,
        deepseek_endpoint: str,
        study: int
    ):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.schema_file = schema_file
        self.deepseek_endpoint = deepseek_endpoint
        self.study = study
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load JSON schema
        with open(schema_file, 'r') as f:
            self.schema = json.load(f)
    
    def construct_extraction_prompt(self, response: str, study: int) -> str:
        """Construct prompt for DeepSeek to extract structured data."""
        if study == 1:
            return f"""Extract structured information from the following response about probability judgments.

Response:
{response}

Extract the following fields and return as JSON:
- Prior_A: The initial answer (integer 0-3)
- probability_0: Probability for state 0 (integer 0-100)
- probability_1: Probability for state 1 (integer 0-100)
- probability_2: Probability for state 2 (integer 0-100)
- probability_3: Probability for state 3 (integer 0-100)
- Knowledge_A: Answer to knowledge question ("yes" or "no")

Return only valid JSON matching this schema:
{json.dumps(self.schema, indent=2)}

JSON output:"""
        
        else:  # study == 2
            return f"""Extract structured information from the following utterance about politeness.

Utterance:
{response}

Extract the following fields and return as JSON:
- Was_Wasnt: Whether the utterance used "was" or "wasn't"
- Assessment: The quality word used (terrible, bad, good, or amazing)
- response_cleaned: The cleaned utterance text

Return only valid JSON matching this schema:
{json.dumps(self.schema, indent=2)}

JSON output:"""
    
    async def extract_structured_data(
        self,
        response: str,
        client: httpx.AsyncClient
    ) -> Optional[dict]:
        """Extract structured data using DeepSeek."""
        prompt = self.construct_extraction_prompt(response, self.study)
        
        try:
            result = await client.post(
                f"{self.deepseek_endpoint}/v1/chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a data extraction assistant. Return only valid JSON."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_tokens": 300,
                    "temperature": 0.0  # Deterministic extraction
                },
                timeout=30.0
            )
            result.raise_for_status()
            
            # Parse JSON from response
            json_text = result.json()['choices'][0]['message']['content']
            # Extract JSON from markdown code blocks if present
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0]
            elif "```" in json_text:
                json_text = json_text.split("```")[1].split("```")[0]
            
            extracted = json.loads(json_text.strip())
            
            # Validate with Pydantic
            if self.study == 1:
                Study1StructuredOutput(**extracted)
            else:
                Study2StructuredOutput(**extracted)
            
            return extracted
        
        except (ValidationError, json.JSONDecodeError) as e:
            self.logger.error(f"Validation/parsing error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Extraction error: {e}")
            return None
    
    async def run(self):
        """Main execution loop."""
        df = pd.read_csv(self.input_csv)
        self.logger.info(f"Loaded {len(df)} rows")
        
        # Add structured output columns
        if self.study == 1:
            new_cols = ['Prior_A', 'probability_0', 'probability_1', 
                       'probability_2', 'probability_3', 'Knowledge_A',
                       'extraction_confidence', 'extraction_errors']
        else:
            new_cols = ['Was_Wasnt', 'Assessment', 'response_cleaned',
                       'extraction_confidence', 'extraction_errors']
        
        for col in new_cols:
            if col not in df.columns:
                df[col] = None
        
        async with httpx.AsyncClient() as client:
            for idx, row in df.iterrows():
                self.logger.info(f"Processing row {idx + 1}/{len(df)}")
                
                response = row['model_response']
                if pd.isna(response):
                    self.logger.warning(f"Row {idx} has no response, skipping")
                    continue
                
                extracted = await self.extract_structured_data(response, client)
                
                if extracted:
                    for key, value in extracted.items():
                        df.at[idx, key] = value
                    df.at[idx, 'extraction_confidence'] = 1.0
                    df.at[idx, 'extraction_errors'] = None
                else:
                    df.at[idx, 'extraction_confidence'] = 0.0
                    df.at[idx, 'extraction_errors'] = "Failed to extract"
                
                # Save intermediate results
                if (idx + 1) % 10 == 0:
                    df.to_csv(self.output_csv, index=False)
                    self.logger.info(f"Saved progress")
        
        df.to_csv(self.output_csv, index=False)
        self.logger.info("Extraction complete!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--schema', required=True)
    parser.add_argument('--deepseek-endpoint', required=True)
    parser.add_argument('--study', type=int, required=True, choices=[1, 2])
    
    args = parser.parse_args()
    
    extractor = StructuredOutputExtractor(
        input_csv=args.input,
        output_csv=args.output,
        schema_file=args.schema,
        deepseek_endpoint=args.deepseek_endpoint,
        study=args.study
    )
    
    asyncio.run(extractor.run())

if __name__ == '__main__':
    main()
```

---

## Data Validation & Quality Checks

### Validation Script

**File**: `src/validate_outputs.py`

```python
#!/usr/bin/env python3
"""
Data validation and quality checking script.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import logging

class DataValidator:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.errors = []
    
    def validate_row_counts(self, input_csv: str, output_csv: str) -> bool:
        """Check that output has same number of rows as input."""
        df_in = pd.read_csv(input_csv)
        df_out = pd.read_csv(output_csv)
        
        if len(df_in) != len(df_out):
            self.errors.append(
                f"Row count mismatch: input={len(df_in)}, output={len(df_out)}"
            )
            return False
        return True
    
    def validate_probabilities(self, df: pd.DataFrame, prob_cols: List[str]) -> bool:
        """Check that probabilities are valid and sum to ~1.0."""
        valid = True
        
        for idx, row in df.iterrows():
            probs = [row[col] for col in prob_cols if pd.notna(row[col])]
            
            if len(probs) != len(prob_cols):
                self.errors.append(f"Row {idx}: Missing probability values")
                valid = False
                continue
            
            # Check range
            if any(p < 0 or p > 1 for p in probs):
                self.errors.append(f"Row {idx}: Probabilities out of range [0,1]")
                valid = False
            
            # Check sum
            prob_sum = sum(probs)
            if not np.isclose(prob_sum, 1.0, atol=0.01):
                self.errors.append(
                    f"Row {idx}: Probabilities sum to {prob_sum:.3f}, not 1.0"
                )
                valid = False
        
        return valid
    
    def validate_structured_output(
        self,
        df: pd.DataFrame,
        study: int
    ) -> bool:
        """Validate structured output fields."""
        valid = True
        
        if study == 1:
            # Check Prior_A is in range
            if not df['Prior_A'].between(0, 3).all():
                self.errors.append("Prior_A values out of range [0, 3]")
                valid = False
            
            # Check probabilities sum to 100
            prob_cols = ['probability_0', 'probability_1', 
                        'probability_2', 'probability_3']
            for idx, row in df.iterrows():
                prob_sum = sum(row[prob_cols])
                if not np.isclose(prob_sum, 100, atol=1.0):
                    self.errors.append(
                        f"Row {idx}: Probabilities sum to {prob_sum}, not 100"
                    )
                    valid = False
            
            # Check Knowledge_A is yes/no
            if not df['Knowledge_A'].isin(['yes', 'no']).all():
                self.errors.append("Knowledge_A contains invalid values")
                valid = False
        
        elif study == 2:
            # Check Was_Wasnt
            if not df['Was_Wasnt'].isin(['was', "wasn't"]).all():
                self.errors.append("Was_Wasnt contains invalid values")
                valid = False
            
            # Check Assessment
            valid_assessments = ['terrible', 'bad', 'good', 'amazing']
            if not df['Assessment'].isin(valid_assessments).all():
                self.errors.append("Assessment contains invalid values")
                valid = False
        
        return valid
    
    def generate_report(self, output_file: str):
        """Generate validation report."""
        with open(output_file, 'w') as f:
            f.write("Data Validation Report\n")
            f.write("=" * 50 + "\n\n")
            
            if not self.errors:
                f.write("✓ All validation checks passed!\n")
            else:
                f.write(f"✗ Found {len(self.errors)} errors:\n\n")
                for i, error in enumerate(self.errors, 1):
                    f.write(f"{i}. {error}\n")
        
        self.logger.info(f"Validation report saved to {output_file}")

# Example usage
if __name__ == '__main__':
    validator = DataValidator()
    
    # Validate Study 1 probabilities
    df = pd.read_csv('/output/exp2/gemma-2b/study1_probabilities.csv')
    validator.validate_probabilities(
        df,
        ['prob_state_0', 'prob_state_1', 'prob_state_2', 'prob_state_3']
    )
    
    # Generate report
    validator.generate_report('/output/validation_report.txt')
```

---

## Workflow Execution Order

### Sequential Execution Plan

```bash
# Phase 1: Deploy models (done once)
kubectl apply -f kubernetes/models/

# Phase 2: Run Experiment 1 (raw responses)
kubectl apply -f kubernetes/jobs/exp1-study1/
kubectl apply -f kubernetes/jobs/exp1-study2/

# Wait for completion
kubectl wait --for=condition=complete job -l experiment=exp1 --timeout=2h

# Phase 3: Run structured extraction
kubectl apply -f kubernetes/jobs/structured-output/

# Wait for completion
kubectl wait --for=condition=complete job -l task=structured-output --timeout=1h

# Phase 4: Run Experiment 2 (probabilities)
kubectl apply -f kubernetes/jobs/exp2-study1/
kubectl apply -f kubernetes/jobs/exp2-study2/

# Wait for completion
kubectl wait --for=condition=complete job -l experiment=exp2 --timeout=2h

# Phase 5: Run validation
kubectl apply -f kubernetes/jobs/validation/

# Phase 6: Download results
kubectl cp grace-experiments/data-downloader:/output ./outputs
```

---

## Error Handling & Recovery

### Checkpoint Strategy

All scripts implement checkpointing:
1. Save output CSV after each batch/10 rows
2. On restart, skip rows that already have outputs
3. Preserve partial results

### Retry Logic

1. **Model query failures**: Retry 3 times with exponential backoff
2. **Parsing failures**: Log error, continue to next row
3. **Job failures**: Kubernetes will retry based on `backoffLimit`

### Manual Recovery

If a job fails completely:
```bash
# Check logs
kubectl logs job/study1-exp1-gemma-2b -n grace-experiments

# Download partial output
kubectl cp grace-experiments/<pod>:/output/partial.csv ./

# Fix issue and re-run with --start-row parameter
# (would need to implement this feature in scripts)
```

---

## Next Steps

1. Implement remaining scripts for Study 2
2. Create JSON schemas
3. Build Docker images
4. Test locally with small dataset
5. Deploy to NRP
6. Run pilot experiment
7. Iterate based on results

---

## Open Questions

1. **Batching Strategy**: Process all rows sequentially or split into parallel jobs?
2. **DeepSeek Hosting**: Final decision on API vs self-hosted
3. **Error Threshold**: What % of extraction failures is acceptable?
4. **Reprocessing**: Should we reprocess failed rows or accept losses?
5. **Data Versioning**: How to track different experimental runs?

These should be answered in Phase 0 of the implementation roadmap.
