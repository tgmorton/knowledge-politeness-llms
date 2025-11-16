# Claude Assistant Guide - Grace Project

This document helps Claude (and other AI assistants) understand the Grace Project structure, key decisions, and how to navigate the documentation when assisting with implementation.

---

## Project Overview

**Grace Project**: Running LLM experiments on the National Research Platform (NRP) Kubernetes cluster to study knowledge attribution and politeness in language models.

**Key Innovation**: Sequential model deployment strategy that eliminates ALL NRP exception requirements while achieving 88% reduction in GPU-hours.

---

## Critical Context

### What Makes This Project Unique

1. **NRP Compliance Achievement**: Through sequential deployment, we achieved **zero exceptions required**
   - Previous plan: 11 GPUs for 2+ weeks, multiple exceptions needed
   - Current plan: 2 GPUs for <1 day per model, fully compliant

2. **Resource Efficiency**: 
   - GPU-hours: 216 (vs 1,848 for parallel) - 88% reduction
   - Peak GPUs: 2 A100-80GB (vs 11 simultaneously)
   - Total runtime: ~1 week for 6 models

3. **NRP Policy Compliance**:
   - All resource limits within 20% of requests
   - Using Deployments (not StatefulSets) for stateless servers
   - Each deployment <1 day (under 2-week auto-delete limit)

---

## Document Hierarchy & Navigation

### Start Here Documents

**When you first join a conversation**, read these in order:

1. **`docs/FINAL_MODEL_LINEUP.md`** - Model selection, resource requirements, compliance status
2. **`plans/00_EXECUTIVE_SUMMARY.md`** - High-level overview, timeline, resource summary
3. **`plans/03_IMPLEMENTATION_ROADMAP.md`** - Phased approach (Phase 0 → Phase 5)

### Policy & Compliance Documents

**When questions involve NRP cluster policies or compliance:**

1. **`docs/NRP_POLICY_COMPLIANCE_AUDIT.md`** - Comprehensive NRP policy synthesis
2. **`docs/COMPLIANCE_ISSUES_FOUND.md`** - Historical issues found (now fixed)
3. **`docs/NRP_CLUSTER_GUIDE.md`** - Complete NRP cluster documentation
4. **`docs/NRP_STORAGE_GUIDE.md`** - Storage classes and Ceph details
5. **`docs/NRP_A100_NODES.md`** - Available A100 nodes on cluster

### Architecture & Infrastructure

**When working on Kubernetes manifests or infrastructure:**

1. **`plans/01_ARCHITECTURE_OVERVIEW.md`** - System architecture, components, workflow
2. **`plans/02_KUBERNETES_INFRASTRUCTURE.md`** - PVCs, Deployments, Services (NRP compliant)
3. **`plans/04_MODEL_SERVING_SPECS.md`** - vLLM configurations per model (compliant)

### Contingency & Alternatives

**When discussing GPU availability or fallback options:**

1. **`docs/GPU_CONTINGENCY_PLAN.md`** - 5-tier fallback strategy (RTX 3090, L40, etc.)
2. **`docs/NRP_A100_NODES.md`** - A100 node availability

### Research Methodology

**When working on experimental design or data processing:**

1. **`plans/STUDY1_EXP2_CLARIFICATION.md`** - Probability extraction methodology (critical!)
2. **`docs/experiment1-grace.md`** - Original Experiment 1 design
3. **`docs/experiment2-grace.md`** - Original Experiment 2 design
4. **`plans/DECISION_CHECKLIST.md`** - Operational decisions tracker

---

## Key Decisions Already Made

### ✅ Architecture Decisions

- **Deployment Strategy**: Sequential (one model at a time)
- **Workload Type**: Kubernetes Deployments (not StatefulSets)
- **GPU Type**: A100-80GB (not A100-40GB)
- **Peak GPU Count**: 2 GPUs
- **NRP Exceptions**: NONE REQUIRED

### ✅ Model Selection (6 models)

1. **Gemma-2 2B** - 1x A100-80GB
2. **Gemma-2 9B** - 1x A100-80GB
3. **Gemma-2 27B** - 2x A100-80GB
4. **Llama-3 70B** - 2x A100-80GB (using tensor parallelism)
5. **GPT-OSS 20B** - 1x A100-80GB (pending OpenAI release)
6. **GPT-OSS 120B** - 2x A100-80GB (pending OpenAI release)

### ✅ Infrastructure Decisions

- **Model weights storage**: `rook-ceph-block` (RBD, fastest)
- **Input data**: `rook-cephfs` (CephFS, shared read)
- **Output data**: `rook-cephfs` (CephFS, shared write)
- **Logs**: `rook-cephfs` (CephFS, shared write)
- **Extraction model**: DeepSeek API (external, no GPU)

### ⏳ Pending Decisions

See `plans/DECISION_CHECKLIST.md` for operational decisions still needed:
- Quantization strategy (fp16 vs AWQ 4-bit)
- Context length (2048 vs 4096 tokens)
- Batch sizes
- Orchestration tool (kubectl vs Argo Workflows)
- Monitoring approach
- Temperature settings
- Error handling parameters

---

## NRP Policy Requirements (Critical!)

### Resource Specification Rules

**ALWAYS ensure compliance when creating/modifying Kubernetes manifests:**

1. **Memory/CPU limits must be within 20% of requests**
   ```yaml
   resources:
     requests:
       cpu: "32"
       memory: 192Gi
     limits:
       cpu: "38"      # 119% - COMPLIANT ✅
       memory: 230Gi  # 120% - COMPLIANT ✅
   ```

2. **Use Deployments for stateless services** (model servers)
   ```yaml
   kind: Deployment  # NOT StatefulSet
   ```

3. **Use Jobs for batch processing** (query generation, extraction)
   ```yaml
   kind: Job
   ```

4. **Never use `sleep infinity` in Jobs** (immediate ban!)

5. **Deployments auto-delete after 2 weeks** (our deployments last <1 day, so compliant)

### Storage Rules

- ✅ **DO**: Use `rook-ceph-block` (RBD) for model weights
- ✅ **DO**: Use `rook-cephfs` for shared data
- ❌ **DON'T**: Install pip/conda on CephFS volumes
- ❌ **DON'T**: Use `rook-cephfs-ucsd` (may be purged)
- ✅ **DO**: Each job writes to unique files (avoid conflicts)

### GPU Rules

- Default limit: 2 GPUs per pod (we stay within this!)
- Our max: 2 GPUs (Gemma-27B, Llama-70B, GPT-OSS-120B)
- No exception needed with A100-80GB + sequential deployment

---

## Implementation Phases

**Current Status**: Phase 0 (Local Development)

### Phase 0: Local Development (Weeks 1-2)
- No NRP cluster access needed
- Test vLLM locally with Gemma-2B
- Develop query generation scripts
- Validate output formats

### Phase 1: RTX 3090 Testing (Week 3)
- Deploy to NRP RTX 3090 cluster (no special access needed)
- Test with Gemma-2B/9B
- Validate end-to-end pipeline
- No exceptions required (standard resources)

### Phase 2: Full RTX 3090 Experiments (Week 4)
- Run all models on RTX 3090 with AWQ quantization
- Complete Study 1 and Study 2
- Generate publishable results
- Fallback if A100 access denied

### Phase 3: Request A100 Access (Week 5)
- ONLY if Phase 2 validates everything works
- Request 2x A100-80GB reservation (optional convenience)
- No exceptions needed - within standard limits

### Phase 4: Production A100 Runs (Week 6)
- Sequential deployment: one model at a time
- Each model: <1 day
- Total: ~1 week for all 6 models
- Clean up between models: `kubectl delete deployment <name>`

### Phase 5: Analysis & Cleanup (Week 7)
- Export all data
- Analyze results
- Clean up cluster resources
- Document lessons learned

---

## Common Tasks & Where to Find Info

### Task: Create Kubernetes Deployment for Model

**Documents to reference:**
1. `plans/02_KUBERNETES_INFRASTRUCTURE.md` - Template Deployment YAML (lines 45-120)
2. `plans/04_MODEL_SERVING_SPECS.md` - Model-specific vLLM args and resources
3. `docs/FINAL_MODEL_LINEUP.md` - GPU/memory requirements per model

**Key requirements:**
- `kind: Deployment` (not StatefulSet)
- Resource limits within 20% of requests
- Use A100-80GB node selector
- vLLM container with correct model args

### Task: Create Query Generation Job

**Documents to reference:**
1. `plans/02_KUBERNETES_INFRASTRUCTURE.md` - Template Job YAML (lines 250-310)
2. `plans/STUDY1_EXP2_CLARIFICATION.md` - Probability extraction approach (4 queries per trial)
3. `docs/experiment1-grace.md` or `docs/experiment2-grace.md` - Study details

**Key requirements:**
- `kind: Job`
- No GPU needed (CPU-only)
- Mount input/output PVCs
- Write to unique output files

### Task: Estimate Resource Requirements

**Documents to reference:**
1. `docs/FINAL_MODEL_LINEUP.md` - Per-model GPU/memory (lines 28-65)
2. `plans/01_ARCHITECTURE_OVERVIEW.md` - Resource requirements table (lines 232-268)

**Quick reference:**
- Single-GPU models: 1x A100-80GB, 64GB RAM
- Dual-GPU models: 2x A100-80GB, 128-192GB RAM
- All limits within 20% of requests

### Task: Troubleshoot NRP Cluster Issues

**Documents to reference:**
1. `docs/NRP_CLUSTER_GUIDE.md` - Complete cluster documentation
2. `docs/NRP_POLICY_COMPLIANCE_AUDIT.md` - Policy synthesis
3. Join NRP Matrix chat: https://matrix.to/#/#nrp:matrix.org

**Common issues:**
- Pod stuck in Pending → Check GPU availability, node selectors
- OOM errors → Check memory limits vs actual usage
- Storage issues → Verify PVC exists, check storage class

### Task: Plan for GPU Unavailability

**Documents to reference:**
1. `docs/GPU_CONTINGENCY_PLAN.md` - 5-tier fallback strategy
2. `docs/NRP_A100_NODES.md` - Current A100 availability

**Fallback order:**
1. RTX 3090 cluster (50+ nodes, no quota)
2. L40 cluster (48GB VRAM)
3. RTX A6000 + A40 mix
4. V100-32GB cluster
5. Hybrid multi-site

All viable with AWQ 4-bit quantization.

---

## Terminology & Conventions

### Kubernetes Workload Types

- **Deployment**: Stateless applications (model servers)
  - Auto-deleted after 2 weeks on NRP
  - Use for vLLM model serving pods
  
- **Job**: Batch processing (query generation, extraction)
  - Runs to completion
  - No time limit on NRP
  - Use for experimental data collection

- **Pod**: Interactive only on NRP
  - 6-hour max lifetime
  - DON'T use for model serving or batch jobs

### Storage Types

- **RBD** (`rook-ceph-block`): Block storage, fastest, single-access
  - Use for: Model weights
  
- **CephFS** (`rook-cephfs`): Shared filesystem, multi-access
  - Use for: Input data, output data, logs
  - DON'T: Install packages on it

### Model Terminology

- **Tensor Parallelism**: Splitting model across multiple GPUs
  - Gemma-27B: `--tensor-parallel-size 2`
  - Llama-70B: `--tensor-parallel-size 2`
  - GPT-OSS-120B: `--tensor-parallel-size 2`

- **MoE (Mixture of Experts)**: GPT-OSS models
  - 20B total params, 3.6B active
  - 120B total params, 5.1B active
  - Much more efficient than dense models

### Quantization

- **fp16/bf16**: Full precision (16-bit)
- **AWQ 4-bit**: 4x memory reduction, <3% quality loss
- **GPTQ 4-bit**: Alternative quantization method

---

## File Naming Conventions

### Output Files

```
{model}_study{N}_{type}_{timestamp}.csv

Examples:
- gemma_2b_study1_raw_responses_20250116.csv
- llama_70b_study2_probabilities_20250120.csv
- gemma_9b_study1_structured_20250118.csv
```

### Kubernetes Resources

```
{type}-{model}-{purpose}

Examples:
- deployment-gemma-2b
- service-llama-70b
- job-query-generation-study1-gemma-9b
- pvc-model-weights-gemma-27b
```

---

## Experiment Details Quick Reference

### Study 1 (300 trials)

**Input**: `study1.csv`
- Columns: participant_id, story_shortname, access, observe, story_setup, priorQ, speech, speechQ

**Experiment 1 Output**: Raw text responses
- File: `{model}_study1_raw_responses.csv`
- One query per trial

**Experiment 2 Output**: Probability distributions
- File: `{model}_study1_probabilities.csv`
- **5 queries per trial**: 4 states (0,1,2,3) + 1 knowledge question
- **Total**: 300 × 5 = 1,500 queries per model
- **68 columns**: 9 original + 44 probabilities + 12 summary stats + 3 knowledge

### Study 2 (2,424 trials)

**Input**: `study2.csv`
- Columns: participant_id, Scenario, Goal, State, Response

**Experiment 1 Output**: Raw politeness responses
- File: `{model}_study2_raw_responses.csv`
- "was/wasn't good/bad/terrible/amazing" style responses

**Experiment 2 Output**: Probability distributions
- File: `{model}_study2_probabilities.csv`
- Logprobs for: "was"/"wasn't" and "good"/"bad"/"terrible"/"amazing"

---

## Best Practices When Assisting

### 1. Always Check Compliance

Before suggesting any Kubernetes manifest:
- ✅ Resource limits within 20% of requests?
- ✅ Using Deployment (not StatefulSet) for model servers?
- ✅ Using Job (not Deployment) for batch processing?
- ✅ Correct storage class (`rook-ceph-block` or `rook-cephfs`)?

### 2. Reference Sequential Deployment

When discussing resource requirements or timelines:
- Mention "sequential deployment" (one model at a time)
- Peak: 2 GPUs, not 11
- Duration: <1 day per model, not weeks
- No exceptions required

### 3. Cite Specific Documents

When providing information, cite the source document:
- Example: "According to FINAL_MODEL_LINEUP.md:28, Llama-70B requires 2x A100-80GB"
- This helps the user verify and learn the documentation structure

### 4. Distinguish Between Decided and Pending

- ✅ Decided: Models, deployment strategy, storage, GPU type
- ⏳ Pending: Quantization, batch sizes, orchestration tool, monitoring

Don't make operational decisions without consulting DECISION_CHECKLIST.md.

### 5. Acknowledge Project Evolution

The project has evolved significantly:
- Started with parallel deployment (11 GPUs, exceptions needed)
- Now using sequential deployment (2 GPUs, no exceptions)
- This is a SUCCESS STORY - emphasize the efficiency gains

### 6. Provide Fallback Options

Always mention RTX 3090 contingency when discussing A100 requirements:
- "If A100 access is unavailable, RTX 3090 cluster is viable with AWQ quantization"
- See GPU_CONTINGENCY_PLAN.md

### 7. Context for Code Generation

When generating code (Python scripts, YAML manifests):
- Use NRP-compliant resource specifications
- Include comments explaining compliance requirements
- Provide validation steps (e.g., "check limits are within 20%")

---

## Common Questions & Answers

### Q: Do we need NRP exceptions?

**A**: NO! Sequential deployment eliminates all exception requirements. See FINAL_MODEL_LINEUP.md:310-340.

### Q: How many GPUs do we need?

**A**: Peak of 2x A100-80GB (sequential deployment). See FINAL_MODEL_LINEUP.md:28 or 00_EXECUTIVE_SUMMARY.md:45.

### Q: Why not use StatefulSets?

**A**: Model servers are stateless (no persistent identity needed). Deployments are simpler and NRP-compliant. See 02_KUBERNETES_INFRASTRUCTURE.md:65-71.

### Q: What if we can't get A100 access?

**A**: RTX 3090 cluster is Plan B (50+ nodes, no quota needed). See GPU_CONTINGENCY_PLAN.md.

### Q: How long will experiments take?

**A**: ~1 week total. Each model <1 day, 6 models sequential. See 03_IMPLEMENTATION_ROADMAP.md:95-140.

### Q: How many API calls for Study 1 Experiment 2?

**A**: 1,500 queries per model (300 trials × 5 queries). See STUDY1_EXP2_CLARIFICATION.md:140-150.

### Q: Can we run models in parallel to go faster?

**A**: Technically yes, but would need 11 GPUs simultaneously + exceptions. Sequential is more efficient (88% fewer GPU-hours) and avoids bureaucracy. See DECISION_CHECKLIST.md:130-137.

---

## Red Flags to Watch For

### In Kubernetes Manifests

❌ `kind: StatefulSet` for model servers → Should be `Deployment`
❌ Resource limits >20% of requests → Non-compliant
❌ `command: ["sleep", "infinity"]` in Jobs → Immediate ban
❌ `storageClassName: rook-cephfs-ucsd` → May be purged
❌ GPU request >2 per pod → Would need exception (we avoid this)

### In Planning Discussions

❌ "We need 11 GPUs" → No, we need 2 (sequential)
❌ "We need exceptions for Llama-70B" → No, fits on 2x A100-80GB
❌ "Deployments need to run for 3 weeks" → No, <1 day each
❌ "Use StatefulSets for state management" → Models are stateless
❌ "Install conda on shared filesystem" → Violates NRP policy

### In Timeline Estimates

❌ "This will take 6-9 weeks" → 4-6 weeks total (updated)
❌ "Need 1,848 GPU-hours" → 216 GPU-hours (sequential)
❌ "All models deployed simultaneously" → Sequential deployment

---

## Quick Command Reference

### Deployment Management (Sequential)

```bash
# Deploy first model
kubectl apply -f deployments/gemma-2b-deployment.yaml

# Check status
kubectl get deployments -n grace-experiments
kubectl get pods -n grace-experiments

# Run experiments for this model
kubectl apply -f jobs/study1-exp1-gemma-2b.yaml
kubectl apply -f jobs/study1-exp2-gemma-2b.yaml

# Wait for completion
kubectl wait --for=condition=complete job/study1-exp1-gemma-2b -n grace-experiments

# Clean up before next model
kubectl delete deployment gemma-2b -n grace-experiments

# Repeat for next model...
```

### Storage Management

```bash
# List PVCs
kubectl get pvc -n grace-experiments

# Check PVC status
kubectl describe pvc model-weights -n grace-experiments

# Create namespace and PVCs (one-time setup)
kubectl apply -f infrastructure/namespace.yaml
kubectl apply -f infrastructure/pvcs.yaml
```

### Debugging

```bash
# Check pod logs
kubectl logs <pod-name> -n grace-experiments

# Describe pod (see events)
kubectl describe pod <pod-name> -n grace-experiments

# Get into pod for debugging
kubectl exec -it <pod-name> -n grace-experiments -- /bin/bash

# Check GPU allocation
kubectl describe node <node-name> | grep -A 10 "Allocated resources"
```

---

## Version History & Evolution

### Initial Plan (Nov 2024)
- Parallel deployment: 11 GPUs for 2+ weeks
- Needed multiple NRP exceptions
- 1,848 GPU-hours estimated
- StatefulSets planned for model servers

### Current Plan (Jan 2025) ✅
- Sequential deployment: 2 GPUs for <1 day each
- ZERO exceptions required
- 216 GPU-hours (88% reduction)
- Deployments for model servers (compliant)
- All resource limits within 20% of requests

**Key Innovation**: Realizing that sequential deployment eliminates exception requirements while still completing experiments in reasonable time (~1 week).

---

## Contact & Resources

### NRP Resources
- **Matrix Chat**: https://matrix.to/#/#nrp:matrix.org
- **Documentation**: https://docs.nationalresearchplatform.org/
- **Portal**: https://nautilus.optiputer.net/

### Project Repository
- **GitHub**: https://github.com/tgmorton/knowledge-politeness-llms.git
- **Main branch**: Contains all compliant documentation

### When to Ask the User

You should ask the user for decisions on:
- Operational parameters (batch sizes, timeouts, retries)
- Quantization strategy preference
- Orchestration tool choice
- Specific timeline/milestone commitments
- Budget/cost constraints
- Risk tolerance and priorities

You should NOT ask the user about:
- Already-decided architecture (sequential deployment)
- Model selection (6 models finalized)
- Storage classes (already configured)
- NRP exception requirements (none needed)

---

## Success Metrics

When assisting with implementation, success means:

✅ All Kubernetes manifests are NRP-compliant
✅ Resource limits within 20% of requests
✅ Sequential deployment maintained (2 GPU peak)
✅ No exceptions requested/required
✅ All 6 models successfully deployed and tested
✅ Study 1 and Study 2 data collected with high quality (>95% success rate)
✅ Total GPU-hours ≤ 250 (target: 216)
✅ Total timeline ≤ 7 weeks (target: 4-6 weeks)
✅ Documentation maintained and updated with lessons learned

---

## Final Notes

This project demonstrates that **thoughtful architecture can eliminate bureaucratic friction**. By choosing sequential deployment over parallel, we:
- Eliminated ALL exception requirements
- Reduced GPU-hours by 88%
- Stayed fully compliant with NRP policies
- Maintained reasonable timeline (~1 week for experiments)

When assisting with this project, **emphasize efficiency and compliance**. Every suggestion should align with these principles.

**Remember**: The goal isn't just to run experiments, but to run them **efficiently, compliantly, and reproducibly** on a shared research infrastructure.

---

*Last Updated*: 2025-01-16  
*Document Version*: 1.0  
*Status*: Active - Use this as primary assistant guide
