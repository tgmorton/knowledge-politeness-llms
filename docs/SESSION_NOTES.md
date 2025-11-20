# Grace Project - Session Notes

## Session 1: Initial Cluster Deployment (2025-01-19)

### Accomplishments

#### 1. GitLab CI/CD Setup ✅
- Created NRP GitLab account and project
- Configured `.gitlab-ci.yml` for automatic Docker builds
- Set up dual-remote git (GitHub + GitLab)
- **Status:** First build timed out (1hr), will retry automatically

#### 2. Cluster Access Configured ✅
- Verified kubectl access to NRP cluster
- Using existing lab namespace: `lemn-lab`
- Discovered available GPUs:
  - 20x A100-80GB (primary target)
  - 48x RTX 3090 (fallback, used for testing)
  - 17x L40, 11x A6000, etc.

#### 3. Storage Created ✅
- Created 2 PVCs with personalized names:
  - `thomas-grace-results` (50GB) - For experiment outputs
  - `thomas-grace-model-cache` (250GB) - For shared model weights
- Decided to include input data in Docker image (simpler)
- Skipped reasoning-traces PVC (can add later if needed)

#### 4. First Deployment Working ✅
- Successfully deployed vLLM with Gemma-2B on RTX 3090
- Resolved issues:
  - HuggingFace authentication (gated model access)
  - Gemma-2 dtype requirement (bfloat16 not float16)
- Created service for cluster access
- **Tested successfully:** Model responded to inference requests

#### 5. Architecture Decisions Made ✅
- **Storage strategy:** Data in Docker image, model cache in PVC
- **GPU configs:** Created separate configs for A100 and RTX 3090
- **Authentication:** Created `hf-token-thomas` secret for gated models
- **Naming:** Prepended `thomas-` to all resources for clear ownership

---

### Issues Encountered & Resolved

| Issue | Solution | Status |
|-------|----------|--------|
| GitLab protected branch | Unprotected main branch | ✅ Resolved |
| No CI runners with 'nautilus' tag | Removed tag, use any runner | ✅ Resolved |
| Gemma-2 gated model access | Accept license, add HF_TOKEN | ✅ Resolved |
| Gemma-2 dtype validation error | Changed float16 → bfloat16 | ✅ Resolved |
| Missing service | Created service.yaml | ✅ Resolved |
| GitLab build timeout (1hr) | Will retry automatically | ⏸️ In progress |

---

### Current State

**What's Running:**
- ❌ Nothing (shut down vLLM deployment to free GPU)

**What's Cached:**
- ✅ Gemma-2B model (~5GB) in `thomas-grace-model-cache`
- ✅ vLLM Docker image (official)
- ⏸️ Custom query-generator image (build in progress)

**Ready for Next Session:**
- ✅ All manifests configured and tested
- ✅ Storage persistent and ready
- ✅ Model cached (fast restart)
- ✅ Authentication configured

---

### Files Created/Modified This Session

**New Files:**
- `kubernetes/vllm-deployment-rtx3090.yaml` - RTX 3090 deployment config
- `kubernetes/job-exp2-rtx3090-template.yaml` - RTX 3090 job template
- `kubernetes/GPU_SELECTION_GUIDE.md` - GPU decision guide
- `scripts/inventory_cluster_gpus.sh` - GPU inventory script
- `docs/NRP_GITLAB_BUILD_GUIDE.md` - GitLab CI/CD guide
- `.gitlab-ci.yml` - CI/CD pipeline config
- `docs/SESSION_NOTES.md` - This file

**Modified Files:**
- `kubernetes/pvcs.yaml` - Simplified to 2 PVCs with personalized names
- `kubernetes/vllm-deployment.yaml` - Added HF_TOKEN, fixed dtype
- `kubernetes/job-exp1-template.yaml` - Updated for model cache, removed input PVC
- `kubernetes/job-exp2-template.yaml` - Updated for model cache, removed input PVC
- `docker/query-generator/Dockerfile` - Added data files, model dependencies
- `docker/query-generator/requirements.txt` - Added PyTorch, transformers
- All manifests updated from `grace-experiments` → `lemn-lab` namespace

---

### Next Session Plan (Phase 1 Continued)

#### Immediate Tasks:
1. **Restart vLLM deployment**
   ```bash
   kubectl apply -f kubernetes/vllm-deployment-rtx3090.yaml
   kubectl apply -f kubernetes/service.yaml
   ```

2. **Upload test data to Docker image**
   - Verify `data/test_samples/` exists
   - Check if GitLab build completed
   - Test data accessibility in container

3. **Run first test experiment**
   - Port-forward to vLLM: `kubectl port-forward -n lemn-lab svc/vllm-gemma-2b 8000:8000`
   - Run Study 1 Exp 1 locally (2-3 trials): `python3 src/query_study1_exp1.py --input data/test_samples/study1_sample.csv --output outputs/test.csv --endpoint http://localhost:8000 --model-name gemma-2-2b-it --limit 3`
   - Verify output format and quality

4. **Deploy first Kubernetes Job**
   - Update job template with correct timestamp
   - Deploy: `kubectl apply -f kubernetes/job-exp1-template.yaml`
   - Monitor: `kubectl get jobs -n lemn-lab -w`
   - Download results from PVC

#### Stretch Goals:
- Test Experiment 2 (direct model scoring)
- Try A100 deployment
- Run full study1_sample.csv (10 trials)

---

### Commands Reference

**Quick Start (Next Session):**
```bash
# Deploy vLLM
kubectl apply -f kubernetes/vllm-deployment-rtx3090.yaml
kubectl apply -f kubernetes/service.yaml

# Check status
kubectl get pods -n lemn-lab -l app=vllm -w

# Port forward (for local testing)
kubectl port-forward -n lemn-lab svc/vllm-gemma-2b 8000:8000

# Test
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model":"google/gemma-2-2b-it","prompt":"What is 2+2?","max_tokens":50}'

# Clean up when done
kubectl delete deployment vllm-gemma-2b-rtx3090 -n lemn-lab
```

**Check Resources:**
```bash
# PVCs
kubectl get pvc -n lemn-lab | grep thomas-grace

# Pods
kubectl get pods -n lemn-lab

# GPU inventory
./scripts/inventory_cluster_gpus.sh
```

---

### Lessons Learned

1. **First builds are slow** - PyTorch downloads take time, but cached after first success
2. **Gated models need authentication** - Remember to add HF_TOKEN for Gemma, Llama, etc.
3. **Gemma-2 requires bfloat16** - Not float16 (numerical stability)
4. **Shared namespace is fine** - Just prepend your name to resources
5. **Model caching works great** - 250GB PVC saves re-downloads
6. **RTX 3090 is good for testing** - 48 nodes available, fast deployment
7. **Services are separate from Deployments** - Don't forget to create both!

---

### Open Questions / Decisions Needed

- [ ] **GitLab build timeout:** Increase timeout or let it retry?
- [ ] **Test data location:** Confirm data files are in `data/test_samples/`
- [ ] **Reasoning traces:** Do we need to add PVC later?
- [ ] **Job orchestration:** Manual kubectl or use Argo Workflows?
- [ ] **A100 access:** Request reservation or use on-demand?

---

### Time Spent This Session
- GitLab setup: ~30 min
- Cluster access verification: ~15 min
- Storage configuration: ~20 min
- First deployment attempts: ~45 min (debugging auth + dtype issues)
- Documentation: ~30 min
- **Total:** ~2.5 hours

---

### Next Milestone: Phase 1 Complete

**Definition of Done:**
- ✅ vLLM deployment working (DONE)
- ⏸️ Job successfully runs 10 trials
- ⏸️ Results saved to PVC
- ⏸️ Can download and validate results
- ⏸️ Both Exp1 (vLLM) and Exp2 (direct scoring) tested

**After Phase 1:** Ready for Phase 2 (full experiments on RTX 3090 or A100)

---

## Session 2: Config-Driven System & End-to-End Testing (2025-11-19)

### Accomplishments

#### 1. Config-Driven Manifest Generation System ✅
- Created centralized configuration system:
  - `config/models.yaml` - 4 RTX 3090 model configurations
  - `config/experiments.yaml` - 4 experiment configurations
- Built `scripts/generate_manifests.py`:
  - Generates all 24 Kubernetes manifests automatically
  - NRP compliance validation (memory/CPU limits, GPU count)
  - Multi-GPU support (tensor parallelism for 2-4 GPUs)
  - AWQ quantization configuration
- Benefits: Single source of truth, no manual YAML editing

#### 2. RTX 3090 Platform Decision ✅
- Decided to use RTX 3090 as primary platform (48 nodes available)
- Model configurations:
  - Gemma-2B: 1 GPU, no quantization
  - Gemma-9B: 1 GPU, no quantization
  - Gemma-27B: 2 GPUs, AWQ 4-bit quantization
  - Llama-70B: 4 GPUs, AWQ 4-bit quantization (requires NRP exception)
- Deferred A100 deployment for later

#### 3. First Full End-to-End Test ✅
- Deployed vLLM (Gemma-2B on RTX 3090)
- Port-forwarded to local machine
- Ran **full Experiment 1** on both studies locally:
  - Study 1: 300 trials (knowledge attribution)
  - Study 2: 2,424 trials (politeness judgments)
  - **Total: 2,724 queries completed successfully**
- Verified output format and quality
- Results saved to `outputs/`

#### 4. Architecture Clarifications ✅
- **Experiment 1**: Uses vLLM API (can run via port-forward or K8s Jobs)
- **Experiment 2**: Requires direct model scoring (must run as K8s Jobs)
- **Output format**: Decided on JSON with embedded reasoning traces
- **Pod communication**: Learned Kubernetes Service DNS resolution

---

### Key Learnings

1. **Config-driven is essential** - Managing 24 YAML files manually would be unmaintainable
2. **Experiment 1 vs 2 difference**:
   - Exp 1: Text generation via vLLM API (works with port-forward)
   - Exp 2: Probability extraction via direct model (needs Job with model loaded)
3. **RTX 3090 viability** - With AWQ quantization, can fit all models except maybe Llama-70B (needs 4 GPUs)
4. **Service discovery** - Pods communicate via stable DNS names (e.g., `http://vllm-gemma-2b:8000`)

---

### Files Created/Modified This Session

**New Files:**
- `config/models.yaml` - Model configurations (4 models)
- `config/experiments.yaml` - Experiment configurations (4 experiments)
- `scripts/generate_manifests.py` - Manifest generator with validation
- `scripts/run_all_experiments_local.sh` - Local test runner (all 4 experiments)
- `scripts/run_exp1_full_local.sh` - Full Exp 1 runner (Study 1 + Study 2)
- `docs/MANIFEST_GENERATION_GUIDE.md` - Complete guide for config system
- `kubernetes/generated/` - 24 auto-generated manifests (gitignored)

**Modified Files:**
- `.gitignore` - Added kubernetes/generated/

---

### Current State

**What's Running:**
- ❌ Nothing (cleaned up after testing)

**What's Cached:**
- ✅ Gemma-2B model (~5GB) in `thomas-grace-model-cache`

**What's Ready:**
- ✅ Config-driven manifest system (24 files generated)
- ✅ Experiment 1 tested and working with full datasets
- ✅ vLLM deployment tested successfully on RTX 3090
- ✅ PVCs persistent and ready

**GitLab Status:**
- ⏸️ Docker image build in progress (may need PyYAML added to requirements.txt)

---

### Next Session Plan (Phase 1 Continued)

#### Immediate Tasks:
1. **Check GitLab build status**
   - If failed: Add PyYAML to `docker/query-generator/requirements.txt`

2. **Deploy first Kubernetes Job**
   ```bash
   # Generate manifests
   python3 scripts/generate_manifests.py

   # Deploy vLLM
   kubectl apply -f kubernetes/generated/deployment-gemma-2b-rtx3090.yaml
   kubectl apply -f kubernetes/generated/service-gemma-2b-rtx3090.yaml

   # Run job
   kubectl apply -f kubernetes/generated/job-study1-exp1-gemma-2b-rtx3090.yaml

   # Monitor
   kubectl logs -f job/grace-study1-exp1-gemma-2b-rtx3090 -n lemn-lab
   ```

3. **Download results from PVC**
   - Create access pod
   - Use `kubectl cp` to download results
   - Verify output matches local test

4. **Request 4-GPU exception for Llama-70B** (if needed)
   - Join NRP Matrix chat
   - Explain: Sequential deployment, <1 day per model, academic research

#### Stretch Goals:
- Implement JSON output with reasoning traces
- Test Experiment 2 (direct model scoring) as Job
- Run all 4 models for Experiment 1

---

### Commands Reference

**Generate Manifests:**
```bash
source venv-grace/bin/activate
python3 scripts/generate_manifests.py
```

**Deploy vLLM (RTX 3090):**
```bash
kubectl apply -f kubernetes/generated/deployment-gemma-2b-rtx3090.yaml
kubectl apply -f kubernetes/generated/service-gemma-2b-rtx3090.yaml
kubectl get pods -n lemn-lab -l model=gemma-2b-rtx3090 -w
```

**Run Experiment Job:**
```bash
kubectl apply -f kubernetes/generated/job-study1-exp1-gemma-2b-rtx3090.yaml
kubectl logs -f job/grace-study1-exp1-gemma-2b-rtx3090 -n lemn-lab
```

**Test Locally (Port-Forward):**
```bash
# Terminal 1: Port forward
kubectl port-forward -n lemn-lab svc/vllm-gemma-2b 8000:8000

# Terminal 2: Run experiments
source venv-grace/bin/activate
./scripts/run_exp1_full_local.sh
```

**Clean Up:**
```bash
kubectl delete deployment vllm-gemma-2b-rtx3090 -n lemn-lab
kubectl delete service vllm-gemma-2b -n lemn-lab
kubectl delete jobs -l model=gemma-2b-rtx3090 -n lemn-lab
```

---

### Time Spent This Session
- Config system design and discussion: ~30 min
- Config-driven manifest generation implementation: ~45 min
- End-to-end testing (2,724 queries): ~10 min
- Documentation: ~20 min
- **Total:** ~1 hour 45 min

---

### Open Questions / Decisions Needed

- [ ] **JSON output format:** Implement in query scripts
- [ ] **4-GPU exception:** Request from NRP for Llama-70B?
- [ ] **Experiment 2 implementation:** Direct model scoring in container
- [ ] **Download workflow:** Automated script for PVC → local?

---

*Last Updated: 2025-11-19*
*Next Session: TBD*
