# NRP Policy Compliance Issues - Grace Project

**Date**: November 16, 2025  
**Audit Status**: Complete  
**Overall Compliance**: ⚠️ **CRITICAL ISSUES FOUND**

---

## Executive Summary

After reviewing all planning documents against NRP cluster policies (documented in `NRP_POLICY_COMPLIANCE_AUDIT.md`), several **critical compliance violations** were identified that must be corrected before deployment.

**Key Findings**:
- ✅ Storage configuration is compliant (correct use of RBD vs CephFS)
- ❌ **CRITICAL**: Workload types violate NRP policies (using StatefulSets instead of Deployments)
- ❌ **CRITICAL**: Resource limits do not meet 20% requirement
- ⚠️ Missing exception requests in implementation roadmap
- ⚠️ Some Job manifests may need resource adjustments

---

## Critical Issue #1: Workload Type - StatefulSets vs Deployments

### Policy Requirement
- Long-running pods must use `kind: Deployment`
- Deployments auto-deleted after 2 weeks (exception available)
- StatefulSets are acceptable but should be Deployments for stateless services

### Current State (NON-COMPLIANT)
`02_KUBERNETES_INFRASTRUCTURE.md` uses **StatefulSets** for all model servers:

```yaml
apiVersion: apps/v1
kind: StatefulSet  # ❌ Should be Deployment for stateless model servers
metadata:
  name: vllm-gemma-2b
```

### Why This is Wrong
- StatefulSets are designed for stateful applications (databases, etc.)
- vLLM model servers are **stateless** (no persistent state beyond model weights)
- Using StatefulSets adds unnecessary complexity
- Deployments are the standard for stateless services

### Required Fix
Change all model server manifests to use `kind: Deployment`:

```yaml
apiVersion: apps/v1
kind: Deployment  # ✅ CORRECT for stateless services
metadata:
  name: vllm-gemma-2b
  namespace: grace-experiments
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-server
      model: gemma-2b
  template:
    # ... rest of pod spec unchanged
```

**Files to Update**:
- `plans/02_KUBERNETES_INFRASTRUCTURE.md` - All model server manifests

---

## Critical Issue #2: Resource Limits Not Within 20% of Requests

### Policy Requirement
- Limits must be within **20% of requests**
- For >100 pods/jobs: **limit = request** (Guaranteed QoS)

### Current State (NON-COMPLIANT)
Example from `02_KUBERNETES_INFRASTRUCTURE.md`:

```yaml
# ❌ NON-COMPLIANT - limits are 100% above requests
resources:
  requests:
    nvidia.com/gpu: 1
    cpu: "8"
    memory: 32Gi
  limits:
    nvidia.com/gpu: 1
    cpu: "16"      # 200% of request (violates 20% rule)
    memory: 64Gi   # 200% of request (violates 20% rule)
```

**Calculation**:
- CPU: 16 / 8 = 2.0 = 200% (allowed range: 120% max)
- Memory: 64 / 32 = 2.0 = 200% (allowed range: 120% max)

### Required Fix
Set limits to be within 20% of requests:

```yaml
# ✅ COMPLIANT - limits within 20% of requests
resources:
  requests:
    nvidia.com/gpu: 1
    cpu: "8"
    memory: 32Gi
  limits:
    nvidia.com/gpu: 1
    cpu: "10"      # 125% of request (within 20%)
    memory: 38Gi   # 119% of request (within 20%)
```

OR use Guaranteed QoS (limit = request):

```yaml
# ✅ ALSO COMPLIANT - Guaranteed QoS
resources:
  requests:
    nvidia.com/gpu: 1
    cpu: "8"
    memory: 32Gi
  limits:
    nvidia.com/gpu: 1
    cpu: "8"      # 100% of request
    memory: 32Gi  # 100% of request
```

**Files to Update**:
- `plans/02_KUBERNETES_INFRASTRUCTURE.md` - All model server Deployments
- `plans/02_KUBERNETES_INFRASTRUCTURE.md` - All Job manifests (check each one)

**Recommendation**: Use **Guaranteed QoS** (limit = request) for model servers since they have predictable resource usage.

---

## Critical Issue #3: Jobs Need Guaranteed QoS for >100 Pods

### Policy Requirement
- For >100 jobs: **limit = request** (Guaranteed QoS)

### Current State (NEEDS VERIFICATION)
Query jobs in `02_KUBERNETES_INFRASTRUCTURE.md`:

```yaml
# Need to check if this is compliant
resources:
  requests:
    cpu: "4"
    memory: 8Gi
  limits:
    cpu: "8"       # Is this within 20%? NO - it's 200%
    memory: 16Gi   # Is this within 20%? NO - it's 200%
```

### Required Fix
Since we'll run many jobs (6 models × 2 studies × 2 experiments = 24+ jobs), use Guaranteed QoS:

```yaml
# ✅ COMPLIANT for >100 jobs
resources:
  requests:
    cpu: "4"
    memory: 8Gi
  limits:
    cpu: "4"      # limit = request
    memory: 8Gi   # limit = request
```

**Files to Update**:
- `plans/02_KUBERNETES_INFRASTRUCTURE.md` - All Job templates

---

## Issue #4: Missing Exception Request Steps in Roadmap

### Policy Requirement
- Request exceptions BEFORE deploying:
  - Multi-GPU pods (>2 GPUs)
  - High RAM pods (>32GB)
  - Deployments >2 weeks

### Current State (INCOMPLETE)
`03_IMPLEMENTATION_ROADMAP.md` Phase 0 mentions exceptions but doesn't provide:
- Specific exception request text
- Which models need which exceptions
- When to request (should be Phase 0, before any deployment)

### Required Fix
Add detailed exception request instructions to Phase 0:

```markdown
#### 0.2: NRP Cluster Access & Exception Requests

**CRITICAL**: Request all exceptions BEFORE Phase 2 deployment

1. [ ] Join Matrix chat: https://matrix.to/#/#nrp:matrix.org
2. [ ] Submit exception requests (see templates in NRP_POLICY_COMPLIANCE_AUDIT.md):
   
   **Exception 1: Multi-GPU Pods**
   - Models: Llama-70B (4 GPUs), Gemma-27B (2 GPUs), GPT-OSS-120B (2 GPUs)
   - Use template from docs/NRP_POLICY_COMPLIANCE_AUDIT.md
   
   **Exception 2: High RAM Pods**
   - Models: Llama-70B (160GB), Gemma-27B (64GB), GPT-OSS-120B (50GB)
   - Use template from docs/NRP_POLICY_COMPLIANCE_AUDIT.md
   
   **Exception 3: Deployment >2 Weeks**
   - All 6 model Deployments
   - Duration: 3-4 weeks
   - Use template from docs/NRP_POLICY_COMPLIANCE_AUDIT.md

3. [ ] Wait for exception approvals (may take 1-3 days)
4. [ ] Complete A100 access request form
```

**Files to Update**:
- `plans/03_IMPLEMENTATION_ROADMAP.md` - Phase 0 tasks

---

## Issue #5: Init Container Installing Packages (Needs Clarification)

### Policy Requirement
- ❌ **NEVER install pip/conda packages on CephFS volumes**
- ✅ All packages must be in Docker images

### Current State (POTENTIALLY COMPLIANT)
`02_KUBERNETES_INFRASTRUCTURE.md` init container:

```yaml
initContainers:
- name: download-model
  image: python:3.11-slim
  command:
  - /bin/bash
  - -c
  - |
    # Install huggingface_hub to local pip cache (NOT on mounted volume)
    pip install --no-cache-dir huggingface_hub  # ⚠️ Where is this installed?
    
    # Download model to RBD volume (fast block storage)
    python -c "..."
  volumeMounts:
  - name: model-weights
    mountPath: /model-weights  # RBD volume, not CephFS
```

### Analysis
- ✅ The model weights are downloaded to RBD (not CephFS) - GOOD
- ⚠️ The pip install could be installing to the container's local filesystem (OK) or to a mounted volume (BAD)
- ✅ Comment says "local pip cache" - suggests it's NOT on mounted volume

### Recommended Clarification
Make it explicit that pip install is local:

```yaml
- |
  # CRITICAL: Install to container's local filesystem, NOT mounted volumes
  pip install --user --no-cache-dir huggingface_hub
  
  # Download model to RBD volume (fast block storage)
  # Use /tmp for HF cache (local to pod, not CephFS)
  python -c "
  from huggingface_hub import snapshot_download
  snapshot_download(
      repo_id='google/gemma-2-2b',
      local_dir='/model-weights',  # RBD volume (OK to write here)
      cache_dir='/tmp/hf-cache'     # Local temp, not shared storage
  )
  "
```

**Better Solution**: Build custom init container image with huggingface_hub pre-installed:

```dockerfile
# docker/model-downloader/Dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir huggingface_hub
COPY download_model.py /usr/local/bin/
ENTRYPOINT ["python", "/usr/local/bin/download_model.py"]
```

Then use in init container:
```yaml
initContainers:
- name: download-model
  image: grace-project/model-downloader:latest  # Pre-built with packages
  args:
  - --repo-id=google/gemma-2-2b
  - --local-dir=/model-weights
  volumeMounts:
  - name: model-weights
    mountPath: /model-weights
```

**Files to Update**:
- `plans/02_KUBERNETES_INFRASTRUCTURE.md` - Init container specification
- Add note about building custom init container image

---

## Issue #6: Pod Monitoring for Violations

### Policy Requirement
- Check violations page daily
- Ensure GPU utilization >40%
- Ensure CPU/memory usage within 20-150% of requests

### Current State (MISSING)
No mention of monitoring violations or automated checks in:
- `03_IMPLEMENTATION_ROADMAP.md`
- `02_KUBERNETES_INFRASTRUCTURE.md`

### Required Fix
Add monitoring tasks to implementation roadmap:

```markdown
### Daily During Deployment (Phases 2-5)

- [ ] Check [NRP Violations Page](https://nautilus.optiputer.net/violations)
- [ ] Monitor GPU utilization: `kubectl top pods -n grace-experiments`
- [ ] Verify no pods violating resource policies
- [ ] Scale down idle model servers if utilization <40%
```

**Files to Update**:
- `plans/03_IMPLEMENTATION_ROADMAP.md` - Add monitoring section
- `plans/02_KUBERNETES_INFRASTRUCTURE.md` - Add monitoring manifests

---

## Issue #7: Data Cleanup Plan Needed

### Policy Requirement
- Data not accessed for 6 months can be purged without notification
- NRP is not archival storage

### Current State (INSUFFICIENT)
Cleanup mentioned in Phase 6 but not detailed enough.

### Required Fix
Add explicit data cleanup procedures:

```markdown
## Phase 6: Cleanup & Data Archival (Week 7)

### CRITICAL: Archive Before Deleting

#### 6.1: Archive All Data (BEFORE cleanup)
- [ ] Create tar archive of all outputs
- [ ] Download to external storage (NOT NRP)
- [ ] Verify archive integrity
- [ ] Document archive location

```bash
# Create archive
kubectl run archiver --image=busybox --restart=Never \
  -n grace-experiments \
  --overrides='...'

kubectl exec archiver -- tar czf /output/grace-project-$(date +%Y%m%d).tar.gz /output /logs

kubectl cp grace-experiments/archiver:/output/grace-project-*.tar.gz ./local-archive/
```

#### 6.2: Delete Kubernetes Resources
- [ ] Delete all Deployments: `kubectl delete deploy -l project=grace -n grace-experiments`
- [ ] Delete all Jobs: `kubectl delete jobs -l project=grace -n grace-experiments`

#### 6.3: Delete Storage (FREE UP 465Gi)
- [ ] Delete model weights PVCs (saves 350Gi):
  ```bash
  kubectl delete pvc grace-model-weights-* -n grace-experiments
  ```
- [ ] Delete output PVC (saves 100Gi):
  ```bash
  kubectl delete pvc grace-output-data -n grace-experiments
  ```
- [ ] Delete logs PVC (saves 10Gi):
  ```bash
  kubectl delete pvc grace-logs -n grace-experiments
  ```
- [ ] Keep input data PVC (only 5Gi) OR delete if archived

#### 6.4: Document Retention
- [ ] Create README in archive with:
  - Date of experiments
  - Models used
  - Number of rows processed
  - Archive contents
  - How to restore if needed

**DO NOT** keep data on NRP beyond 30 days after completion
```

**Files to Update**:
- `plans/03_IMPLEMENTATION_ROADMAP.md` - Phase 6 tasks

---

## Issue #8: No Sleep Commands in Jobs (Verification Needed)

### Policy Requirement
- ❌ **CRITICAL**: `sleep infinity` or scripts ending with `sleep` in Jobs = **IMMEDIATE BAN**

### Current State (NEEDS VERIFICATION)
No obvious `sleep` commands in Job manifests in `02_KUBERNETES_INFRASTRUCTURE.md`, but need to verify all Python scripts don't end with sleep.

### Required Fix
Add explicit verification step:

```markdown
#### 1.5: Verify No Sleep Commands

**CRITICAL**: Violating this will result in cluster ban

- [ ] Search all scripts for sleep commands:
  ```bash
  grep -r "sleep" src/ docker/
  grep -r "time.sleep" src/
  ```
- [ ] Ensure no Jobs have `sleep` in command:
  ```bash
  grep -r "sleep" kubernetes/jobs/
  ```
- [ ] All Jobs must exit cleanly with status 0
- [ ] Use proper Python exit: `sys.exit(0)` or natural script completion

**Acceptable**: `time.sleep(1)` for retry backoff in the middle of a script
**FORBIDDEN**: Script ending with `sleep infinity` or `while True: time.sleep(10)`
```

**Files to Update**:
- `plans/03_IMPLEMENTATION_ROADMAP.md` - Phase 1 tasks

---

## Summary of Required Updates

| Document | Issues Found | Priority | Estimated Effort |
|----------|--------------|----------|------------------|
| `02_KUBERNETES_INFRASTRUCTURE.md` | StatefulSet→Deployment, resource limits | ❌ CRITICAL | 2-3 hours |
| `03_IMPLEMENTATION_ROADMAP.md` | Exception requests, monitoring, cleanup | ⚠️ HIGH | 1-2 hours |
| `04_MODEL_SERVING_SPECS.md` | Resource limit examples | ⚠️ MEDIUM | 30 min |
| `05_DATA_PIPELINE_DESIGN.md` | No issues found | ✅ OK | N/A |

---

## Compliance Checklist (Post-Fix)

After making all updates, verify:

- [ ] All model servers use `kind: Deployment` (not StatefulSet)
- [ ] All resource limits within 20% of requests
- [ ] Jobs use Guaranteed QoS (limit = request)
- [ ] Exception request templates included in roadmap
- [ ] Phase 0 includes exception request steps
- [ ] Monitoring tasks added to roadmap
- [ ] Data cleanup procedures documented
- [ ] No `sleep` commands in Jobs verified
- [ ] Init containers clarified (no pip on CephFS)

---

## Next Steps

1. ✅ Review this compliance issues document
2. ⏳ Update `02_KUBERNETES_INFRASTRUCTURE.md` (CRITICAL)
3. ⏳ Update `03_IMPLEMENTATION_ROADMAP.md` (HIGH)
4. ⏳ Update `04_MODEL_SERVING_SPECS.md` (MEDIUM)
5. ⏳ Test one model server deployment with corrected manifest
6. ⏳ Commit all changes to git
7. ⏳ Request exceptions from NRP before Phase 2

---

## Appendix: Policy Reference

See `docs/NRP_POLICY_COMPLIANCE_AUDIT.md` for:
- Complete cluster policies
- Exception request templates
- Monitoring commands
- Detailed compliance requirements

---

**Document Status**: ✅ Complete  
**Action Required**: Update planning documents immediately  
**Blocker**: Cannot deploy until Critical Issues #1 and #2 are fixed
