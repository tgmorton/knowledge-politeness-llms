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

*Last Updated: 2025-01-19*
*Next Session: TBD*
