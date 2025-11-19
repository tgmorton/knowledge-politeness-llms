# Local Testing Guide for M1 Mac

## Overview

Before deploying to Kubernetes (which can be intimidating if you're new to it!), you can test everything locally on your M1 Mac. This ensures your scripts work correctly before touching the cluster.

## Quick Start (5 minutes)

### Step 1: Install Dependencies

```bash
# Install local testing requirements
pip install -r requirements-local.txt
```

This installs:
- `transformers` - For loading Gemma-2B
- `flask` - For mock API server
- `torch` - PyTorch with M1 Metal support
- All the query script dependencies

### Step 2: Start Local Mock Server

Open a terminal and run:

```bash
python3 tests/local_vllm_mock.py --model google/gemma-2-2b-it
```

**What happens:**
- Downloads Gemma-2B from Hugging Face (~5GB, one-time)
- Starts a server on `http://localhost:8000`
- Mimics vLLM's OpenAI-compatible API
- Uses M1's Metal Performance Shaders (GPU acceleration!)

**Expected output:**
```
INFO:__main__:Loading model: google/gemma-2-2b-it
INFO:__main__:Using device: mps
INFO:__main__:Model loaded successfully!
INFO:__main__:Starting server on 127.0.0.1:8000
```

**Note**: First run takes ~5-10 minutes to download the model.

### Step 3: Run Quick Test (In Another Terminal)

```bash
./tests/quick_local_test.sh
```

**What this does:**
- Runs all 4 experiments with just 2 trials each
- Total: ~24 queries (very quick!)
- Validates your scripts work correctly
- Creates outputs in `outputs/local_test/`

**Expected output:**
```
[1/4] Testing Study 1 Experiment 1 (2 trials)
✅ Study 1 Exp 1 passed

[2/4] Testing Study 1 Experiment 2 (2 trials, 10 queries)
Note: This makes 5 queries per trial (10 total)
✅ Study 1 Exp 2 passed

[3/4] Testing Study 2 Experiment 1 (2 trials)
✅ Study 2 Exp 1 passed

[4/4] Testing Study 2 Experiment 2 (2 trials, 4 queries)
✅ Study 2 Exp 2 passed

ALL TESTS PASSED! 🎉
```

### Step 4: Review Outputs

```bash
# Look at the outputs
ls -lh outputs/local_test/

# Check Study 1 Exp 2 (the complex one!)
head outputs/local_test/study1_exp2_test.csv
```

Verify:
- ✅ Files exist
- ✅ CSVs have correct number of columns (68 for Study 1 Exp 2!)
- ✅ No error messages in responses
- ✅ Probability columns look reasonable

## What's Different from Real vLLM?

The local mock server:
- ✅ **Same API** - Identical endpoints to vLLM
- ✅ **Works on M1** - Uses Metal GPU acceleration
- ⚠️ **Slower** - 5-10x slower than real vLLM (CPU tokenization)
- ⚠️ **Less accurate logprobs** - Approximates vLLM's probability extraction
- ✅ **Good enough for testing** - Validates your scripts work!

## Performance on M1 Mac

With Gemma-2B on M1 Max (32GB):
- **Model loading**: 2-5 minutes (first time)
- **Per query**: 2-5 seconds
- **2 trials (all 4 experiments)**: ~3-5 minutes
- **Memory usage**: ~10-15GB RAM

**Tip**: Don't try to run full datasets locally - it will take hours! Just validate with 2-10 trials, then move to K8s cluster.

## Troubleshooting

### "Model not found" or download fails

```bash
# Login to Hugging Face first
huggingface-cli login

# Enter your token (get from: https://huggingface.co/settings/tokens)
```

### "MPS device not available"

If you see "Using device: cpu", it means Metal isn't available. This is fine, just slower.

To enable MPS:
```bash
# Check PyTorch MPS support
python3 -c "import torch; print(torch.backends.mps.is_available())"

# Should print: True
```

If False, update PyTorch:
```bash
pip install --upgrade torch
```

### Out of memory

Reduce model size or switch to CPU mode:

```bash
# Use smaller model for testing
python3 tests/local_vllm_mock.py --model google/gemma-2b

# Or force CPU mode (slower)
# Edit local_vllm_mock.py line 35:
# device = "cpu"  # Force CPU
```

### Server crashes or hangs

- Check Activity Monitor - kill python if needed
- Restart with more verbose logging:
  ```bash
  python3 tests/local_vllm_mock.py --model google/gemma-2-2b-it
  ```

## Advanced: Test Individual Experiments

Instead of running the full test suite, test one experiment at a time:

```bash
# Study 1 Experiment 1 (simplest)
python3 src/query_study1_exp1.py \
    --input data/test_samples/study1_sample.csv \
    --output outputs/local_test/s1e1.csv \
    --endpoint http://localhost:8000 \
    --model-name gemma-2-2b-it \
    --limit 1

# Study 1 Experiment 2 (5 queries per trial - the complex one!)
python3 src/query_study1_exp2.py \
    --input data/test_samples/study1_sample.csv \
    --output outputs/local_test/s1e2.csv \
    --endpoint http://localhost:8000 \
    --model-name gemma-2-2b-it \
    --limit 1
```

## Next Steps After Local Testing

Once local tests pass:

1. ✅ **Scripts validated** - Your code works!
2. 🚀 **Deploy to Kubernetes** - Much faster, real vLLM
3. 📊 **Run full datasets** - All 300 (Study 1) or 2,424 (Study 2) trials

See [KUBERNETES_DEPLOYMENT_GUIDE.md](./KUBERNETES_DEPLOYMENT_GUIDE.md) for next steps.

## FAQ

**Q: Do I need a GPU on my Mac?**
A: No, but M1/M2 Macs can use Metal (GPU) acceleration. Intel Macs will use CPU (slower but works).

**Q: Can I use a different model?**
A: Yes! Change `--model` argument. Try `google/gemma-2b` (smaller, faster) or any HF model.

**Q: How long does full local testing take?**
A: ~5 minutes for 2 trials of each experiment. Don't run full datasets locally!

**Q: Can I skip local testing?**
A: Yes, but it's riskier. Local testing catches bugs before using cluster resources.

**Q: Does this work on Intel Macs?**
A: Yes, but uses CPU instead of GPU (slower). Still good for validation.

## Summary

**Local testing workflow:**
1. Install dependencies (one-time)
2. Start mock server (downloads model first time)
3. Run quick test script (2 trials each)
4. Review outputs
5. Fix any issues
6. Deploy to Kubernetes with confidence!

**Benefits:**
- ✅ Validates scripts before K8s
- ✅ Catches bugs early
- ✅ No cluster access needed
- ✅ Fast iteration during development
- ✅ Builds confidence before production

Happy testing! 🚀
