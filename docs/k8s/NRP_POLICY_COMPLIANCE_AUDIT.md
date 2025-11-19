# NRP Cluster Policy Compliance Audit

**Document Version**: 1.0  
**Last Updated**: November 16, 2025  
**Purpose**: Ensure Grace Project deployment adheres to all NRP Nautilus cluster policies

---

## Executive Summary

This document synthesizes all NRP cluster policies and provides a compliance checklist for the Grace Project. All project plans must be reviewed against these policies before deployment.

**Critical Policies**:
- ✅ Use **Jobs** for batch processing (no time limits)
- ✅ Use **Deployments** for model serving (requires exception for >2 weeks)
- ❌ **Never use standalone Pods** (auto-deleted after 6 hours)
- ✅ Set resource limits within 20% of requests
- ✅ Maintain >40% GPU utilization
- ❌ **Never use `sleep infinity` in Jobs** (immediate ban)
- ✅ Clean up data every 6 months

---

## 1. Workload Type Policies

### 1.1 Interactive Pods (Standalone Pods)

**Policy**:
- Maximum runtime: **6 hours** (auto-deleted)
- Resource limits: **2 GPUs, 32GB RAM, 16 CPU cores**
- Assumed to be temporary/development use
- Okay to use `sleep` in interactive pods
- Exception available for namespaces running JupyterHub

**Grace Project Compliance**:
- ❌ **DO NOT USE** standalone Pods for any Grace Project workloads
- All model servers must use Deployments
- All experiments must use Jobs

### 1.2 Batch Jobs

**Policy**:
- Use `Job` workload controller for batch processing
- Jobs ensure completion (exit status 0) and cleanup
- Must set resources carefully
- For >100 jobs: **limit = request** (Guaranteed QoS)
- ❌ **CRITICAL**: `sleep infinity` or scripts ending with `sleep` are **FORBIDDEN**
  - Violators will be **banned from cluster**

**Grace Project Compliance**:
- ✅ Use Jobs for:
  - Query generation (Experiment 1 & 2)
  - Probability extraction
  - Structured output extraction
- ✅ All Jobs must exit cleanly (no sleep commands)
- ✅ Set `limit = request` if running >100 parallel jobs

**Example Violation** (DO NOT DO THIS):
```yaml
# ❌ FORBIDDEN - Will result in BAN
command: ["python", "query.py", "&&", "sleep", "infinity"]
```

**Correct Approach**:
```yaml
# ✅ CORRECT - Job runs and exits
command: ["python", "query.py"]
restartPolicy: Never
```

### 1.3 Long-Running Deployments

**Policy**:
- Use `Deployment` for long-running pods
- Auto-deleted after **2 weeks** unless on exception list
- Must use **Burstable QoS** (minimal requests, proper limits)
- **Cannot request GPUs** (unless on exception list)
- Receive 3 notifications before deletion
- Data in PVCs remains after deletion

**Exception Process**:
- Contact admins in Matrix
- Provide estimated duration
- Provide service description
- Long idle pods cannot get exceptions

**Grace Project Compliance**:
- ✅ Use Deployments for model servers (vLLM pods)
- ✅ **Request exception BEFORE deployment**:
  - Duration: 3-4 weeks
  - Description: "Academic research - self-hosted LLM model servers for comparative study"
- ✅ All 6 model Deployments need exception (GPU + >2 weeks)
- ✅ Set minimal requests with appropriate limits (see Resource Policies)

---

## 2. Resource Allocation Policies

### 2.1 Resource Limits

**Policy**:
- Limits must be within **20% of requests**
- For >100 pods/jobs: **limit = request**
- Aim for requests close to average usage
- Set limits slightly above peak usage
- Use monitoring to fine-tune

**Consequences**:
- Memory exceeded → Pod killed (OOM)
- CPU exceeded → Node pressure, affects other users

**Grace Project Compliance**:
```yaml
# Model Server Example (Gemma-2B)
resources:
  requests:
    cpu: "8"
    memory: "32Gi"
    nvidia.com/gpu: "1"
  limits:
    cpu: "10"        # 125% of request (within 20%)
    memory: "38Gi"   # 119% of request (within 20%)
    nvidia.com/gpu: "1"
```

### 2.2 Resource Waste Prevention

**Policy**:
- Do not waste resources
- If requested, must use it
- Free resources once computation done
- Admins monitor utilization
- Namespaces with underutilized requests will be **banned**

**Grace Project Compliance**:
- ✅ Scale down or delete model Deployments when not actively querying
- ✅ Monitor GPU dashboard to ensure >40% utilization
- ✅ Use monitoring to right-size requests
- ✅ Terminate Jobs promptly after completion

---

## 3. Resource Usage Violation Policies

### 3.1 Per-User Pod Limits

**Policy**:
A single user cannot submit more than **4 pods** violating:
- GPU utilization: **< 40%** of requested GPUs
- CPU usage: **outside 20-200%** of requested CPUs
- Memory usage: **outside 20-150%** of requested memory

**Exception**: Pods requesting ≤1 CPU core and ≤2GB memory are exempt

**Grace Project Compliance**:
- ✅ Check [Violations page](https://nautilus.optiputer.net/) regularly
- ✅ Each model server should maintain:
  - GPU util: >40% (ideally >80% during queries)
  - CPU usage: 20-200% of request
  - Memory usage: 20-150% of request
- ✅ Scale down idle model servers to avoid violations

**Monitoring**:
```bash
# Check GPU utilization for namespace
kubectl top pods -n grace-experiments --sort-by=gpu
```

### 3.2 GPU Utilization Requirements

**Policy**:
- Minimum GPU utilization: **40%**
- Ideally close to **100%**
- Only request multiple GPUs if:
  - Current utilization is ~100%
  - You can leverage additional GPUs
- Large jobs (>50 GPUs): present plan in Matrix first

**Grace Project Compliance**:
- ✅ Total GPU request: 11 GPUs (within large job threshold)
- ✅ Present plan in Matrix: "Grace Project - 11 A100 GPUs for LLM research"
- ✅ Monitor GPU dashboard by namespace
- ✅ Justify multi-GPU requests:
  - Llama-70B: 4 GPUs (tensor parallelism required)
  - Gemma-27B: 2 GPUs (tensor parallelism required)
  - GPT-OSS-120B: 2 GPUs (MoE routing + tensor parallelism)

**A100 Access**:
- ✅ Complete [A100 access request form](https://nautilus.optiputer.net/a100-request)
- ✅ Default quota is **zero** - must request access
- ✅ Provide workflow details in form

---

## 4. Data Management Policies

### 4.1 Data Purging

**Policy**:
- Clean up storage at regular intervals
- NRP is **not archival storage**
- Only store data actively used for computations
- Any volume not accessed for **6 months** can be **purged without notification**

**Grace Project Compliance**:
- ✅ Total storage: ~465Gi (0.47TB)
- ✅ Storage breakdown:
  - Model weights: 350Gi (accessed during deployment)
  - Input data: 5Gi (accessed during experiments)
  - Output data: 100Gi (accessed during analysis)
  - Logs: 10Gi (accessed for debugging)
- ✅ Cleanup plan:
  - Delete model weights after experiments complete (if not reusing)
  - Archive output data to external storage (not NRP)
  - Purge logs after analysis complete
  - Download all results before 6-month mark

**Post-Experiment Cleanup** (Critical):
```bash
# After experiments complete, archive and delete
kubectl cp -n grace-experiments <pod>:/outputs ./local-archive/
kubectl delete pvc model-weights-* -n grace-experiments  # Save 350Gi
```

---

## 5. Acceptable Use Policy (AUP)

**Policy**:
- Read and accept [NRP AUP](https://nationalresearchplatform.org/aup/)
- Must accept before using cluster
- Academic research use
- No commercial use without approval
- No illegal activities
- No cryptocurrency mining

**Grace Project Compliance**:
- ✅ Academic research project (comparative LLM study)
- ✅ All users must accept AUP before deployment
- ✅ No commercial use
- ✅ Cite NRP in publications

---

## 6. Grace Project-Specific Compliance Checklist

### Pre-Deployment (Phase 0)

- [ ] All team members accept NRP AUP
- [ ] Complete A100 GPU access request form
- [ ] Join NRP Matrix chat
- [ ] Request namespace exceptions in Matrix:
  - [ ] Multi-GPU pods (4 GPUs for Llama-70B)
  - [ ] High RAM pods (>32GB for large models)
  - [ ] Deployment >2 weeks runtime (3-4 weeks)
- [ ] Present GPU usage plan in Matrix (11 A100 GPUs)

### Model Deployment (Phase 2)

**Model Server Deployments**:

| Model | GPUs | RAM | Exception Needed | Compliance Notes |
|-------|------|-----|------------------|------------------|
| Gemma-2B | 1 | 32GB | ❌ No (within limits) | ✅ Standard deployment |
| Gemma-9B | 1 | 32GB | ❌ No | ✅ Standard deployment |
| Gemma-27B | 2 | 64GB | ✅ **YES** (>32GB RAM, 2 GPUs) | ⚠️ Request exception |
| Llama-70B | 4 | 160GB | ✅ **YES** (>2 GPUs, >32GB RAM) | ⚠️ Request exception |
| GPT-OSS-20B | 1 | 32GB | ❌ No | ✅ Standard deployment |
| GPT-OSS-120B | 2 | 50GB | ✅ **YES** (>32GB RAM, 2 GPUs) | ⚠️ Request exception |

**Deployment YAML Requirements**:
- [ ] Use `kind: Deployment` (not Pod)
- [ ] Set `limits` within 20% of `requests`
- [ ] Use Burstable QoS (requests < limits)
- [ ] Include resource monitoring sidecar
- [ ] Set `restartPolicy: Always`

### Batch Jobs (Phases 3-5)

**Query Generation Jobs**:
- [ ] Use `kind: Job` (not Pod)
- [ ] Set `restartPolicy: Never`
- [ ] ❌ **No `sleep` commands** in scripts
- [ ] Set `limit = request` (running >100 jobs)
- [ ] Jobs exit cleanly with status 0
- [ ] Include retry logic in Python (not Kubernetes retries)

**Resource Requests for Jobs**:
```yaml
# Query Job Example
resources:
  requests:
    cpu: "4"
    memory: "8Gi"
  limits:
    cpu: "4"      # limit = request for >100 jobs
    memory: "8Gi"
```

### Monitoring (Continuous)

- [ ] Check [Violations page](https://nautilus.optiputer.net/) daily
- [ ] Monitor GPU dashboard for namespace
- [ ] Ensure GPU utilization >40% (ideally >80%)
- [ ] Verify CPU/memory usage within 20-150% of requests
- [ ] Scale down idle model servers

### Cleanup (Phase 6)

- [ ] Archive all outputs to external storage (not NRP)
- [ ] Delete Deployments
- [ ] Delete Jobs
- [ ] Delete PVCs with model weights
- [ ] Keep only essential data (<6 months retention)

---

## 7. Common Violations to Avoid

### ❌ Immediate Ban Offenses

1. **Using `sleep infinity` in Jobs**:
   ```yaml
   # ❌ FORBIDDEN
   command: ["bash", "-c", "python query.py && sleep infinity"]
   ```

2. **Running standalone Pods for long-running services**:
   ```yaml
   # ❌ FORBIDDEN (will be deleted in 6 hours)
   kind: Pod
   metadata:
     name: vllm-server
   ```

### ⚠️ Namespace Ban Risks

1. **Underutilized resource requests**:
   - Requesting 8 CPUs but using <1.6 CPUs
   - Requesting 4 GPUs but utilization <40%

2. **Resource waste**:
   - Leaving idle model servers running
   - Not cleaning up completed Jobs

3. **>4 pods violating usage policies**:
   - 5+ pods with <40% GPU utilization
   - 5+ pods with memory usage <20% of requests

### ⚠️ Data Loss Risks

1. **Not accessing storage for 6 months**:
   - PVCs can be purged without notification
   - Always archive important data externally

---

## 8. Exception Request Templates

### Template 1: Multi-GPU Pod Exception

**Subject**: Exception Request - Multi-GPU Pods for Academic LLM Research

**Message**:
```
Namespace: grace-experiments
Requested Exception: Allow >2 GPUs per pod

Pods requiring exceptions:
- vllm-llama-70b: 4x A100 GPUs (tensor parallelism)
- vllm-gemma-27b: 2x A100 GPUs (tensor parallelism)
- vllm-gpt-oss-120b: 2x A100 GPUs (MoE routing)

Reason: Academic research comparing large language models. 
Tensor parallelism required to fit models in GPU memory.

Project: Grace Project - Knowledge & Politeness in LLMs
Duration: 3-4 weeks
Total GPUs: 11 A100 GPUs
```

### Template 2: Deployment >2 Weeks Exception

**Subject**: Exception Request - Long-Running Model Servers (>2 weeks)

**Message**:
```
Namespace: grace-experiments
Requested Exception: Allow Deployments to run >2 weeks

Deployments:
- vllm-gemma-2b, vllm-gemma-9b, vllm-gemma-27b
- vllm-llama-70b
- vllm-gpt-oss-20b, vllm-gpt-oss-120b

Duration: 3-4 weeks

Description: Self-hosted LLM model servers for academic research study. 
Models serve inference requests for experimental data collection. 
Not idle - actively processing queries during experiments.

Will scale down when not in use and delete after experiments complete.
```

### Template 3: High RAM Pod Exception

**Subject**: Exception Request - High RAM Pods for Large Models

**Message**:
```
Namespace: grace-experiments
Requested Exception: Allow >32GB RAM per pod

Pods requiring exceptions:
- vllm-llama-70b: 160GB RAM (70B parameters)
- vllm-gemma-27b: 64GB RAM (27B parameters)
- vllm-gpt-oss-120b: 50GB RAM (117B parameters, MoE)

Reason: Large model parameters + KV cache for inference. 
Academic LLM research project.
```

---

## 9. Monitoring Commands

### Check Violations
```bash
# Check violations page (web UI)
open https://nautilus.optiputer.net/violations

# Check pod resource usage
kubectl top pods -n grace-experiments

# Check GPU usage
kubectl top pods -n grace-experiments --sort-by=gpu
```

### Monitor GPU Utilization
```bash
# SSH into pod and check nvidia-smi
kubectl exec -it vllm-gemma-2b-0 -n grace-experiments -- nvidia-smi

# Check vLLM metrics endpoint
kubectl port-forward -n grace-experiments vllm-gemma-2b-0 8000:8000
curl http://localhost:8000/metrics
```

### Check Resource Requests vs Limits
```bash
# Describe pod to see requests/limits
kubectl describe pod vllm-gemma-2b-0 -n grace-experiments

# Get all pods with resources
kubectl get pods -n grace-experiments -o custom-columns=\
NAME:.metadata.name,\
CPU-REQ:.spec.containers[0].resources.requests.cpu,\
CPU-LIM:.spec.containers[0].resources.limits.cpu,\
MEM-REQ:.spec.containers[0].resources.requests.memory,\
MEM-LIM:.spec.containers[0].resources.limits.memory
```

---

## 10. Policy Compliance Matrix

| Policy Category | Grace Project Status | Actions Required |
|----------------|---------------------|------------------|
| **Workload Types** | ⚠️ Partial | Update all manifests to use Jobs/Deployments |
| **Resource Limits** | ⚠️ Needs Review | Ensure all limits within 20% of requests |
| **GPU Utilization** | ✅ Compliant | Monitor during experiments (>40%) |
| **Batch Jobs** | ⚠️ Needs Audit | Verify no `sleep` commands in scripts |
| **Deployments** | ❌ Not Compliant | Request exceptions for >2 weeks + GPUs |
| **Multi-GPU Pods** | ❌ Not Compliant | Request exceptions for 2-4 GPU pods |
| **Data Management** | ✅ Compliant | Cleanup plan documented |
| **AUP Acceptance** | ⚠️ Pending | All users must accept |

---

## 11. Next Steps

### Immediate Actions
1. ✅ Create this audit document
2. ⏳ Review all planning documents for compliance
3. ⏳ Update Kubernetes manifests with policy-compliant configurations
4. ⏳ Create exception request messages for Matrix
5. ⏳ Document cleanup procedures

### Before Phase 0
1. All users accept NRP AUP
2. Submit A100 access request form
3. Join Matrix and submit exception requests
4. Wait for exception approvals

### Before Phase 2
1. Verify all Deployment manifests are compliant
2. Confirm exceptions are approved
3. Add monitoring to all pods

### During Experiments
1. Check violations page daily
2. Monitor GPU utilization (>40%)
3. Scale down idle pods

### After Experiments
1. Archive all data externally
2. Delete all Deployments
3. Delete PVCs
4. Clean up namespace

---

## 12. Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-11-16 | Initial compliance audit document | Grace Project Team |

---

## 13. References

- [NRP Cluster Policies](https://ucsd-prp.gitlab.io/userdocs/running/policies/)
- [NRP Acceptable Use Policy](https://nationalresearchplatform.org/aup/)
- [NRP Portal](https://nautilus.optiputer.net/)
- [A100 Access Request Form](https://nautilus.optiputer.net/a100-request)
- Grace Project Planning Documents: `../plans/`

---

**Document Status**: ✅ Complete  
**Next Review**: Before Phase 0 deployment  
**Compliance Status**: ⚠️ Requires updates to planning documents
