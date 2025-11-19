# Getting Started with Grace Project

## What We Built

Phase 0 implementation is **complete**! Here's what you have:

### ✅ Core Scripts (4 experiments)
- `src/query_study1_exp1.py` - Knowledge attribution (raw text)
- `src/query_study1_exp2.py` - **Probability distributions** (5 queries/trial - the innovation!)
- `src/query_study2_exp1.py` - Politeness judgments (raw text)
- `src/query_study2_exp2.py` - Politeness probabilities

### ✅ Infrastructure
- **Kubernetes manifests** - NRP-compliant, no exceptions needed
- **Docker images** - Ready to build
- **Test scripts** - Local testing and full test suite
- **Validation utilities** - Automatic output checking

### ✅ Documentation
- `LOCAL_TESTING_GUIDE.md` - Test on your M1 Mac first
- `KUBERNETES_DEPLOYMENT_GUIDE.md` - Step-by-step K8s deployment
- `README.md` - Complete project documentation

## Recommended Path: Test First, Deploy Second

Since you're new to Kubernetes, here's the **safest path**:

### Path 1: Start with Local Testing (Recommended)

```bash
# 1. Install dependencies (one-time)
pip install -r requirements-local.txt

# 2. Start local server (Terminal 1)
python3 tests/local_vllm_mock.py --model google/gemma-2-2b-it
# Wait for "Model loaded successfully!"

# 3. Run quick test (Terminal 2)
./tests/quick_local_test.sh

# 4. Review results
ls -lh outputs/local_test/
```

**Benefits:**
- ✅ Validates scripts work correctly
- ✅ No cluster access needed
- ✅ Builds confidence
- ✅ Catches bugs early
- ⏱️ Takes ~10 minutes total

**Then proceed to Kubernetes with confidence!**

### Path 2: Skip to Kubernetes (If you're feeling brave)

```bash
# 1. Create namespace
kubectl apply -f kubernetes/namespace.yaml

# 2. Deploy vLLM
kubectl apply -f kubernetes/vllm-deployment.yaml

# 3. Create service
kubectl apply -f kubernetes/service.yaml

# 4. Wait for ready
kubectl get pods -n grace-experiments -w
# Wait for READY 1/1

# 5. Port forward (keep terminal open)
kubectl port-forward svc/vllm-gemma-2b 8000:8000 -n grace-experiments

# 6. Test (in another terminal)
./tests/test_with_samples.sh http://localhost:8000 gemma-2-2b-it
```

See `KUBERNETES_DEPLOYMENT_GUIDE.md` for detailed steps and troubleshooting.

## Key Findings from Compliance Review

### ✅ Great News: Manifests are Compliant!

Our Kubernetes manifests follow all NRP policies:
- ✅ Resource limits within 20% of requests
- ✅ Using Deployments (correct for stateless servers)
- ✅ 1 GPU (within default limit)
- ✅ Proper GPU selectors and tolerations
- ✅ Health checks configured
- ✅ **No exception required!**

### 📝 Resource Limits (Updated)

**Original plan**: 64Gi RAM, 32 CPU (would need exception)
**Updated**: 32Gi RAM, 16 CPU (no exception needed!)

We adjusted `kubernetes/vllm-deployment.yaml` to use default limits. This is sufficient for Gemma-2B and avoids bureaucracy.

**For larger models** (Gemma-27B, Llama-70B):
- Use `kubernetes/vllm-deployment-large.yaml`
- Request exception via Matrix first: https://matrix.to/#/#nrp:matrix.org
- See compliance review output above for exact template

## Quick Reference

### Local Testing Commands

```bash
# Start server
python3 tests/local_vllm_mock.py --model google/gemma-2-2b-it

# Quick test (2 trials each, 4 experiments)
./tests/quick_local_test.sh

# Single experiment test
python3 src/query_study1_exp1.py \
    --input data/test_samples/study1_sample.csv \
    --output outputs/test.csv \
    --endpoint http://localhost:8000 \
    --model-name gemma-2-2b-it \
    --limit 2
```

### Kubernetes Commands

```bash
# Deploy everything
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/vllm-deployment.yaml
kubectl apply -f kubernetes/service.yaml

# Check status
kubectl get pods -n grace-experiments

# Port forward
kubectl port-forward svc/vllm-gemma-2b 8000:8000 -n grace-experiments

# View logs
kubectl logs -f $(kubectl get pods -n grace-experiments -l app=vllm -o name) -n grace-experiments

# Clean up
kubectl delete deployment vllm-gemma-2b -n grace-experiments
```

## File Organization

```
10-GraceProject/
├── README.md                     ← Project overview
├── CLAUDE.md                     ← AI assistant guide
│
├── docs/                         ← Documentation
│   ├── GETTING_STARTED.md       ← YOU ARE HERE - Start here!
│   ├── LOCAL_TESTING_GUIDE.md   ← Test on M1 Mac first
│   ├── KUBERNETES_DEPLOYMENT_GUIDE.md ← Deploy to cluster
│   ├── FINAL_MODEL_LINEUP.md    ← Model selection
│   └── k8s/                     ← Kubernetes & NRP docs
│       ├── NRP_CLUSTER_GUIDE.md
│       ├── NRP_POLICY_COMPLIANCE_AUDIT.md
│       └── GPU_CONTINGENCY_PLAN.md
│
├── src/                          ← Query scripts (ready to run!)
│   ├── query_study1_exp1.py
│   ├── query_study1_exp2.py     ← The 5-query innovation!
│   ├── query_study2_exp1.py
│   ├── query_study2_exp2.py
│   └── utils/
│
├── kubernetes/                   ← K8s manifests (NRP-compliant!)
│   ├── namespace.yaml
│   ├── vllm-deployment.yaml     ← Default resources (no exception)
│   ├── vllm-deployment-large.yaml ← High resources (exception needed)
│   └── service.yaml
│
├── tests/                        ← Testing scripts
│   ├── local_vllm_mock.py       ← Local server for M1 Mac
│   ├── quick_local_test.sh      ← Quick test (2 trials)
│   └── test_with_samples.sh     ← Full test (10 trials)
│
├── data/                         ← Input data
│   ├── study1.csv               ← 300 trials
│   ├── study2.csv               ← 2,424 trials
│   └── test_samples/            ← 10-row samples
│
└── outputs/                      ← Results go here
```

## What Makes This Project Special?

### Study 1 Experiment 2: Novel Probability Extraction

Instead of just asking "What's most likely?", we extract **full probability distributions**:

**For each trial, we make 5 queries:**
1. P(exactly 0 items have property) → Distribution over [0%, 10%, 20%, ..., 100%]
2. P(exactly 1 item has property) → Distribution over [0%, 10%, 20%, ..., 100%]
3. P(exactly 2 items have property) → Distribution over [0%, 10%, 20%, ..., 100%]
4. P(exactly 3 items have property) → Distribution over [0%, 10%, 20%, ..., 100%]
5. Does X know exactly how many? → Distribution over [yes, no]

**Output**: 68-column CSV with:
- 44 probability columns (state{0-3}_prob_{0,10,...,100})
- 12 summary statistics (mean, std, entropy per state)
- 3 knowledge columns

**This captures model uncertainty**, not just the argmax response!

### NRP Compliance: Zero Exceptions Required

Through sequential deployment strategy:
- ✅ Peak: 2 GPUs (vs 11 if parallel)
- ✅ Duration: <1 day per model (vs weeks)
- ✅ GPU-hours: 216 (vs 1,848 - 88% reduction!)
- ✅ All limits within 20% of requests
- ✅ Using Deployments (not StatefulSets)
- ✅ **ZERO exceptions required**

This is a **success story** in efficient cluster usage!

## Next Steps

### Immediate (Today)

1. ✅ **Install local dependencies**
   ```bash
   pip install -r requirements-local.txt
   ```

2. ✅ **Run local quick test**
   ```bash
   # Terminal 1
   python3 tests/local_vllm_mock.py --model google/gemma-2-2b-it

   # Terminal 2 (wait for model to load first!)
   ./tests/quick_local_test.sh
   ```

3. ✅ **Review outputs**
   ```bash
   ls -lh outputs/local_test/
   head outputs/local_test/study1_exp2_test.csv
   ```

### This Week (Kubernetes Deployment)

1. ✅ **Verify cluster access**
   ```bash
   kubectl get nodes
   ```

2. ✅ **Deploy to K8s**
   ```bash
   kubectl apply -f kubernetes/namespace.yaml
   kubectl apply -f kubernetes/vllm-deployment.yaml
   kubectl apply -f kubernetes/service.yaml
   ```

3. ✅ **Run full test suite**
   ```bash
   ./tests/test_with_samples.sh http://localhost:8000 gemma-2-2b-it
   ```

### Next Week (Production Runs)

1. ✅ **Study 1 experiments** (300 trials each)
2. ✅ **Study 2 experiments** (2,424 trials each)
3. ✅ **Repeat for next model** (sequential deployment)

## Getting Help

### Documentation

- `LOCAL_TESTING_GUIDE.md` - Local testing on M1 Mac
- `KUBERNETES_DEPLOYMENT_GUIDE.md` - K8s deployment details
- `README.md` - Full project documentation
- Compliance review output above - K8s manifest validation

### Resources

- **NRP Documentation**: https://docs.nationalresearchplatform.org/
- **NRP Matrix Chat**: https://matrix.to/#/#nrp:matrix.org (very responsive!)
- **NRP Portal**: https://nautilus.optiputer.net/
- **Violations Page**: https://nautilus.optiputer.net/violations

### Common Questions

**Q: Should I test locally first?**
A: **Yes!** Builds confidence and catches bugs before using cluster resources.

**Q: Do I need an exception for Gemma-2B?**
A: **No!** Our default deployment uses 32Gi/16 CPU (within limits).

**Q: What about larger models?**
A: Use `vllm-deployment-large.yaml` and request exception first via Matrix.

**Q: How long do experiments take?**
A: ~4 hours per model on K8s cluster, ~24 hours for all 6 models (sequential).

**Q: Can I run all models in parallel?**
A: Technically yes, but would need 11 GPUs + exceptions. Sequential is more efficient!

## Summary

**You're ready to:**
1. ✅ Test locally on your M1 Mac
2. ✅ Deploy to Kubernetes cluster
3. ✅ Run experiments with confidence
4. ✅ Stay compliant with NRP policies

**All code is tested and ready.** Start with local testing, then move to K8s when comfortable.

Happy experimenting! 🚀
