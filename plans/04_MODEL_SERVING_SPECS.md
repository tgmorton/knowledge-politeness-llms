# Model Serving Technical Specifications

## Overview

This document provides detailed technical specifications for serving transformer models using vLLM on Kubernetes, including configuration options, optimization strategies, and troubleshooting guides.

---

## Model Selection & Specifications

### Recommended Models for Grace Project

| Model | Size | Context Length | License | GPU Requirement | Rationale |
|-------|------|----------------|---------|-----------------|-----------|
| **Gemma 2 - 2B** | 2B params | 8K tokens | Apache 2.0 | 1x A100 40GB | Fast baseline, good instruction following |
| **Gemma 2 - 9B** | 9B params | 8K tokens | Apache 2.0 | 1x A100 40GB | Better reasoning, still efficient |
| **Llama 3.1 - 8B** | 8B params | 128K tokens | Llama 3.1 License | 1x A100 40GB | Strong general capability, huge context |
| **Llama 3.1 - 70B** | 70B params | 128K tokens | Llama 3.1 License | 4x A100 40GB | High quality, research-grade |
| **Phi-3 Mini** | 3.8B params | 128K tokens | MIT | 1x A100 40GB | Microsoft's small but powerful model |
| **Mistral 7B v0.3** | 7B params | 32K tokens | Apache 2.0 | 1x A100 40GB | Excellent performance/size ratio |
| **DeepSeek-V3** | 671B params (37B active) | 128K tokens | DeepSeek License | 2x A100 80GB | For structured output extraction |

**Recommended Subset for Budget Constraints**:
- Gemma 2 - 2B (baseline)
- Gemma 2 - 9B (mid-range)
- Llama 3.1 - 8B (mid-range alt)
- Llama 3.1 - 70B (high quality)
- DeepSeek-V3 (extraction)

**Total GPU Requirements**: 8x A100 40GB + 2x A100 80GB

---

## vLLM Configuration Deep Dive

### Essential Parameters

#### 1. Model Loading
```bash
--model <path_or_hf_id>
```
- Local path: `/model-weights/gemma-2-9b`
- HuggingFace ID: `google/gemma-2-9b-it`

#### 2. Data Type & Quantization
```bash
--dtype <auto|float16|bfloat16|float32>
```
- **float16**: Default, good for most models, memory efficient
- **bfloat16**: Better numerical stability, recommended for larger models
- **auto**: Let vLLM decide based on model config

**Quantization** (requires `--quantization`):
```bash
--quantization <awq|gptq|squeezellm|fp8>
```
- **awq**: 4-bit, best quality/speed tradeoff
- **gptq**: 4-bit, wider model support
- **fp8**: 8-bit, best for A100 GPUs
- **squeezellm**: 4-bit, older method

**Recommendation**: Use `--dtype float16` for 8B models, `--dtype bfloat16` for 70B models

#### 3. Tensor Parallelism
```bash
--tensor-parallel-size <N>
```
- Split model across N GPUs
- Required for models > 40GB
- Example: Llama 70B needs `--tensor-parallel-size 4`

#### 4. GPU Memory Management
```bash
--gpu-memory-utilization <0.0-1.0>
```
- Default: 0.90 (90% of GPU memory)
- Recommendation: 0.95 for stable workloads
- Lower to 0.85 if seeing OOM errors

```bash
--max-model-len <tokens>
```
- Maximum sequence length (prompt + generation)
- Lower this if running out of memory
- Example: `--max-model-len 2048` for short prompts

#### 5. Performance Optimization
```bash
--enable-chunked-prefill
```
- Better handling of long prompts
- Reduces latency spikes

```bash
--max-num-batched-tokens <N>
```
- Maximum tokens in a batch
- Default: determined by model
- Increase for better throughput on short prompts

```bash
--max-num-seqs <N>
```
- Maximum concurrent sequences
- Default: 256
- Increase if you have high request rate

#### 6. API Configuration
```bash
--host 0.0.0.0
--port 8000
```
- Bind to all interfaces in Kubernetes

```bash
--api-key <key>
```
- Optional authentication

---

## Complete vLLM Launch Commands

### Gemma 2 - 2B
```bash
python -m vllm.entrypoints.openai.api_server \
  --model /model-weights/gemma-2-2b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --enable-chunked-prefill \
  --max-num-seqs 256 \
  --trust-remote-code
```

### Gemma 2 - 9B
```bash
python -m vllm.entrypoints.openai.api_server \
  --model /model-weights/gemma-2-9b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --enable-chunked-prefill \
  --max-num-seqs 128 \
  --trust-remote-code
```

### Llama 3.1 - 8B
```bash
python -m vllm.entrypoints.openai.api_server \
  --model /model-weights/llama-3.1-8b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --enable-chunked-prefill \
  --max-num-seqs 128
```

### Llama 3.1 - 70B (Multi-GPU)
```bash
python -m vllm.entrypoints.openai.api_server \
  --model /model-weights/llama-3.1-70b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.92 \
  --enable-chunked-prefill \
  --max-num-seqs 64
```

### DeepSeek-V3 (MoE Model)
```bash
python -m vllm.entrypoints.openai.api_server \
  --model /model-weights/deepseek-v3 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill \
  --max-num-seqs 32 \
  --trust-remote-code
```

---

## API Endpoints & Usage

### OpenAI-Compatible Endpoints

#### 1. Health Check
```bash
GET /health
```
Response:
```json
{"status": "ok"}
```

#### 2. List Models
```bash
GET /v1/models
```
Response:
```json
{
  "object": "list",
  "data": [
    {
      "id": "gemma-2-9b",
      "object": "model",
      "created": 1234567890,
      "owned_by": "vllm"
    }
  ]
}
```

#### 3. Completions (for raw text generation)
```bash
POST /v1/completions
Content-Type: application/json

{
  "model": "gemma-2-9b",
  "prompt": "The capital of France is",
  "max_tokens": 50,
  "temperature": 0.7,
  "top_p": 0.9,
  "n": 1,
  "stream": false,
  "logprobs": null,
  "stop": ["\n"]
}
```

Response:
```json
{
  "id": "cmpl-xxxxx",
  "object": "text_completion",
  "created": 1234567890,
  "model": "gemma-2-9b",
  "choices": [
    {
      "text": " Paris.",
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 6,
    "completion_tokens": 2,
    "total_tokens": 8
  }
}
```

#### 4. Chat Completions (for instruction-tuned models)
```bash
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "gemma-2-9b",
  "messages": [
    {
      "role": "system",
      "content": "You are analyzing pragmatic language use."
    },
    {
      "role": "user",
      "content": "Context: Students in the introductory bio class almost always have passing grades..."
    }
  ],
  "max_tokens": 150,
  "temperature": 0.7,
  "top_p": 0.9
}
```

#### 5. Completions with Logprobs (for Experiment 2)
```bash
POST /v1/completions
Content-Type: application/json

{
  "model": "gemma-2-9b",
  "prompt": "How many of the 3 exams do you think have passing grades?",
  "max_tokens": 1,
  "temperature": 1.0,
  "logprobs": 10,
  "echo": false
}
```

Response includes:
```json
{
  "choices": [
    {
      "text": "2",
      "logprobs": {
        "tokens": ["2"],
        "token_logprobs": [-0.5],
        "top_logprobs": [
          {
            "0": -2.1,
            "1": -1.3,
            "2": -0.5,
            "3": -1.8
          }
        ]
      }
    }
  ]
}
```

**Key for Experiment 2**: Use `top_logprobs` to get probabilities for specific tokens

---

## Prompt Engineering for Each Study

### Study 1 - Experiment 1 (Raw Response)

**System Prompt**:
```
You are analyzing scenarios about knowledge and information access. 
Please respond thoughtfully to questions about what people likely know 
based on the information they have accessed.
```

**User Prompt Template**:
```python
def construct_study1_prompt(row):
    return f"""Context: {row['story_setup']}

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
```

### Study 1 - Experiment 2 (Probability Extraction)

**Approach**: For each trial, query the model 4 times (once per state) to get probability distributions over percentage values. This captures the model's uncertainty about each state.

**Prompt for Each State (example for state=2)**:
```python
f"""Context: {row['story_setup']}

{row['speach']}

Question: What percentage probability do you assign that exactly 2 of the 3 {items} {property}?

Express your answer as a percentage from 0% to 100%, in increments of 10% (e.g., 0%, 10%, 20%, etc.).

Answer with just the percentage:"""
```

**Then extract logprobs for tokens**: `["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"]`

**Repeat for states 0, 1, 2, and 3**

This creates a **4×11 probability matrix** per trial, showing:
- How confident the model is that each state has 0%, 10%, 20%, ... 100% probability
- Mean and standard deviation for each state's distribution
- Entropy of each distribution (uncertainty measure)

**Prompt for Knowledge Question**:
```python
f"""Context: {row['story_setup']}

Speaker's statement: {row['speach']}

Question: {row['knowledgeQ']}

Answer with only yes or no:"""
```

**Then extract logprobs for tokens**: `["yes", "no", "Yes", "No"]`

### Study 2 - Experiment 1 (Raw Response)

**System Prompt**:
```
You are analyzing polite communication in social contexts. 
Consider both the speaker's goals (informational vs social) and 
the true state of affairs when generating responses.
```

**User Prompt Template**:
```python
def construct_study2_prompt(row):
    return f"""{row['Precontext']}

{row['Scenario']}

{row['Goal']}

The true state is: {row['State']} (where 0 hearts = terrible, 1 heart = bad, 2 hearts = good, 3 hearts = amazing)

What would {row['SP_Name']} say? Generate a response in the format:
"It [was/wasn't] [terrible/bad/good/amazing]."

Response:"""
```

### Study 2 - Experiment 2 (Probability Extraction)

**Prompt for Was/Wasn't**:
```python
f"""{row['Scenario']}

{row['Goal']}

True state: {row['State']}

Would the speaker say "It was..." or "It wasn't..."?

Answer with only one word:"""
```

**Extract logprobs for**: `["was", "wasn't", "Was", "Wasn"]`

**Prompt for Assessment**:
```python
f"""{row['Scenario']}

{row['Goal']}

True state: {row['State']}

What quality word would the speaker use? (terrible, bad, good, or amazing)

Answer with only one word:"""
```

**Extract logprobs for**: `["terrible", "bad", "good", "amazing"]`

---

## Extracting and Normalizing Probabilities

### Python Implementation

```python
import numpy as np
from typing import Dict, List

def extract_token_probabilities(
    logprobs_response: dict,
    target_tokens: List[str]
) -> Dict[str, float]:
    """
    Extract and normalize probabilities for target tokens from logprobs.
    
    Args:
        logprobs_response: Response from vLLM with logprobs
        target_tokens: List of tokens to extract probabilities for
    
    Returns:
        Dictionary mapping tokens to probabilities
    """
    # Get top logprobs from response
    top_logprobs = logprobs_response['choices'][0]['logprobs']['top_logprobs'][0]
    
    # Extract logprobs for target tokens (case-insensitive)
    token_logprobs = {}
    for token in target_tokens:
        # Check exact match
        if token in top_logprobs:
            token_logprobs[token] = top_logprobs[token]
        # Check capitalized version
        elif token.capitalize() in top_logprobs:
            token_logprobs[token] = top_logprobs[token.capitalize()]
        # Check lowercase version
        elif token.lower() in top_logprobs:
            token_logprobs[token] = top_logprobs[token.lower()]
        else:
            # Token not in top logprobs, assign very low probability
            token_logprobs[token] = -10.0  # log(~0.00005)
    
    # Convert log probabilities to probabilities
    logprobs_array = np.array(list(token_logprobs.values()))
    
    # Normalize using softmax
    exp_logprobs = np.exp(logprobs_array)
    probabilities = exp_logprobs / exp_logprobs.sum()
    
    # Create output dictionary
    result = {
        token: float(prob)
        for token, prob in zip(target_tokens, probabilities)
    }
    
    return result

# Example usage for Study 1
response = client.completions.create(
    model="gemma-2-9b",
    prompt="How many of the 3 exams do you think have passing grades?",
    max_tokens=1,
    temperature=1.0,
    logprobs=10
)

probs = extract_token_probabilities(
    response,
    target_tokens=["0", "1", "2", "3"]
)

print(probs)
# Output: {"0": 0.05, "1": 0.15, "2": 0.65, "3": 0.15}
```

### Handling Edge Cases

1. **Token not in top logprobs**: Assign minimum probability
2. **Multiple token variants**: Aggregate probabilities for "yes", "Yes", "YES"
3. **Subword tokens**: Handle tokenization issues (e.g., "2" vs " 2")
4. **Numerical stability**: Use log-sum-exp trick if needed

```python
def safe_softmax(logprobs: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    max_logprob = np.max(logprobs)
    exp_logprobs = np.exp(logprobs - max_logprob)
    return exp_logprobs / exp_logprobs.sum()
```

---

## Performance Benchmarking

### Metrics to Track

1. **Throughput**: Requests per second
2. **Latency**: Time to first token (TTFT), Time per output token (TPOT)
3. **GPU Utilization**: % GPU memory used, % GPU compute used
4. **Queue Depth**: Number of waiting requests

### Benchmarking Script

```python
import time
import asyncio
import httpx
from statistics import mean, stdev

async def benchmark_model(endpoint: str, num_requests: int = 100):
    """Benchmark model throughput and latency."""
    
    prompt = "The capital of France is"
    latencies = []
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        start_time = time.time()
        
        tasks = []
        for _ in range(num_requests):
            task = client.post(
                f"{endpoint}/v1/completions",
                json={
                    "model": "model",
                    "prompt": prompt,
                    "max_tokens": 10,
                    "temperature": 0.7
                }
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
    
    throughput = num_requests / total_time
    
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {throughput:.2f} requests/second")
    print(f"Average latency: {total_time/num_requests:.2f}s per request")

# Run benchmark
asyncio.run(benchmark_model("http://vllm-gemma-2b:8000"))
```

### Expected Performance

| Model | GPU | Throughput (req/s) | TTFT (ms) | TPOT (ms/token) |
|-------|-----|-------------------|-----------|-----------------|
| Gemma 2-2B | 1x A100 | 50-80 | 50-100 | 10-20 |
| Gemma 2-9B | 1x A100 | 20-40 | 100-200 | 20-30 |
| Llama 8B | 1x A100 | 20-40 | 100-200 | 20-30 |
| Llama 70B | 4x A100 | 5-10 | 200-400 | 40-60 |

*Based on batch size=1, max_tokens=50*

---

## Troubleshooting Guide

### Issue 1: OOM (Out of Memory)

**Symptoms**: Pod crashes, logs show "CUDA out of memory"

**Solutions**:
1. Reduce `--gpu-memory-utilization` to 0.85
2. Reduce `--max-model-len` to 2048 or 1024
3. Reduce `--max-num-seqs` to 64 or 32
4. Use quantization: `--quantization awq`
5. Use smaller model variant

### Issue 2: Slow Inference

**Symptoms**: High latency, low throughput

**Solutions**:
1. Enable `--enable-chunked-prefill`
2. Increase `--max-num-batched-tokens`
3. Adjust `--max-num-seqs` (higher for short prompts, lower for long)
4. Check GPU utilization (should be >70%)
5. Verify tensor parallelism is working

### Issue 3: Model Not Loading

**Symptoms**: Init container fails, model download errors

**Solutions**:
1. Check HuggingFace token is valid
2. Verify model ID is correct
3. Check PVC has enough space
4. Increase init container timeout
5. Pre-download models to PVC manually

### Issue 4: API Errors

**Symptoms**: 500 errors, timeouts, connection refused

**Solutions**:
1. Check pod is in Running state
2. Verify readiness probe is passing
3. Test health endpoint: `curl http://service:8000/health`
4. Check logs for errors
5. Verify service is properly configured

### Issue 5: Incorrect Probabilities

**Symptoms**: Probabilities don't sum to 1, unexpected distributions

**Solutions**:
1. Verify using `temperature=1.0` for unbiased sampling
2. Check token is in model's vocabulary
3. Handle subword tokenization properly
4. Increase `logprobs` parameter to 10 or 20
5. Normalize using softmax correctly

---

## Resource Optimization Strategies

### 1. Model Quantization

**Benefits**:
- 2-4x memory reduction
- 1.5-2x inference speedup
- Can fit larger models on same GPU

**Trade-offs**:
- Slight quality degradation (usually <5%)
- Not all quantization methods support all models

**Recommendation**: Use AWQ quantization for production

```bash
# Install AutoAWQ
pip install autoawq

# Quantize model
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-3.1-8B"
quant_path = "llama-3.1-8b-awq"

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Quantize
model.quantize(tokenizer, quant_config={"bits": 4})

# Save
model.save_quantized(quant_path)
```

### 2. KV Cache Optimization

vLLM uses PagedAttention for efficient KV cache management automatically.

**Monitor KV cache usage**:
- Check vLLM metrics endpoint
- Adjust `--max-num-seqs` based on cache utilization

### 3. Batch Size Tuning

**For long prompts**: Lower batch size (1-8)
**For short prompts**: Higher batch size (16-64)

Test different batch sizes to find optimal throughput.

### 4. Multi-Instance vs Single-Instance

**Option A**: One large model instance
- Pros: Better quality, shared memory
- Cons: Single point of failure

**Option B**: Multiple smaller model replicas
- Pros: Higher availability, load balancing
- Cons: More memory usage

**Recommendation**: For this project, use single instances (simpler)

---

## Model-Specific Notes

### Gemma Models
- Instruction-tuned variants: `gemma-2-2b-it`, `gemma-2-9b-it`
- Use chat completion API for better results
- Requires `--trust-remote-code`

### Llama Models
- Very good at following instructions
- Use official HuggingFace IDs: `meta-llama/Llama-3.1-8B-Instruct`
- May require HuggingFace token for gated access

### DeepSeek Models
- MoE architecture requires careful memory management
- Best for structured output tasks
- Use `--trust-remote-code`
- May need larger `--max-model-len` for complex JSON

---

## Next Steps

1. Choose final model lineup based on GPU availability
2. Test vLLM commands locally with 1-2 models
3. Benchmark performance locally
4. Deploy to NRP cluster
5. Run pilot experiment with 10 rows
6. Iterate on prompts and configurations

---

## References

- vLLM Documentation: https://docs.vllm.ai/
- HuggingFace Model Hub: https://huggingface.co/models
- OpenAI API Reference: https://platform.openai.com/docs/api-reference
- Kubernetes GPU Scheduling: https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/
