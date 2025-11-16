# Pre-Implementation Decision Checklist

Before beginning implementation, make decisions on the following items. Check off each item as decisions are made.

---

## 1. Model Selection ✅ COMPLETED

### Primary Models (Choose 3-5)

- [x] **Gemma 2 - 2B** (✅ SELECTED - fast baseline)
- [x] **Gemma 2 - 9B** (✅ SELECTED - mid-range quality)
- [x] **Gemma 2 - 27B** (✅ SELECTED - high Gemma quality)
- [ ] **Llama 3.1 - 8B** (❌ Skipped - using Gemma-9B instead)
- [x] **Llama 3.1 - 70B** (✅ SELECTED - high quality)
- [x] **GPT-OSS 20B** (✅ SELECTED - pending OpenAI release)
- [x] **GPT-OSS 120B** (✅ SELECTED - pending OpenAI release)

**Decision**: 6 models selected (see FINAL_MODEL_LINEUP.md)

**GPU Requirements**: 2x A100-80GB (sequential deployment)

### Extraction Model ✅ COMPLETED

- [ ] **DeepSeek-V3** (self-hosted, 2x A100 80GB)
- [x] **DeepSeek API** (✅ SELECTED - external, no GPU required)
- [ ] **Llama 3.1 70B** (repurpose existing model for extraction)

**Decision**: DeepSeek API (simpler, no additional GPUs needed)

---

## 2. Model Optimization

### Quantization Strategy

- [ ] **Full Precision (fp16/bf16)** - Best quality, most memory
- [ ] **AWQ 4-bit** - 50% memory reduction, <5% quality loss
- [ ] **GPTQ 4-bit** - Alternative quantization method
- [ ] **fp8** - 8-bit, good for A100 GPUs

**Decision**: ___________________

**Rationale**: _________________________________________________

### Context Length

- [ ] **2048 tokens** (conservative, more memory for batching)
- [ ] **4096 tokens** (standard, enough for our prompts)
- [ ] **8192 tokens** (high, reduces batch size)

**Decision**: ___________________

---

## 3. Infrastructure Configuration

### Storage Classes (NRP Ceph) ✅ COMPLETED

**Reference**: See `docs/NRP_STORAGE_GUIDE.md` for complete details

**Configured storage classes**:
- [x] Model weights (RBD, fastest): `rook-ceph-block`
- [x] Input data (CephFS, shared read): `rook-cephfs`
- [x] Output data (CephFS, shared write): `rook-cephfs`
- [x] Logs (CephFS, shared write): `rook-cephfs`

**Critical Rules Acknowledged**:
- [x] ❌ Never install pip/conda on CephFS volumes
- [x] ✅ Each job writes to unique files (no conflicts)
- [x] ✅ Use RBD for model weights (faster than CephFS)
- [x] ❌ Never use `rook-cephfs-ucsd` (may be purged)

### GPU Quota and NRP Exceptions ✅ COMPLETED

**Reference**: See `docs/NRP_CLUSTER_GUIDE.md` for NRP cluster policies

**Sequential Deployment Strategy**: Deploy one model at a time (<1 day each)

**Current GPU allocation**: Standard NRP allocation

**Peak GPU requirement**: 2x A100-80GB (sequential deployment)

- [ ] GPU quota request submitted (NOT NEEDED - within limits)
- [ ] GPU quota request approved (NOT NEEDED)

**NRP Exception Requests**: 

✅ **NO EXCEPTIONS REQUIRED!**

**Why No Exceptions Needed**:
- [x] ✅ **Deployment duration**: Each model runs <1 day (well under 2 week auto-delete limit)
- [x] ✅ **GPU count**: Max 2 GPUs per pod (using A100-80GB, Llama-70B fits on 2 GPUs)
- [x] ✅ **Memory limits**: All limits within 20% of requests (NRP compliant)
- [x] ✅ **Resource quotas**: Peak usage fits within standard allocations

**Previous Plan (Parallel Deployment)** - ❌ NOT USED:
- ~~Would have needed: 11 GPUs simultaneously for 2+ weeks~~
- ~~Would have needed: 4 GPU per pod exception for Llama-70B~~
- ~~Would have needed: Deployment duration exception~~

**Current Plan (Sequential Deployment)** - ✅ COMPLIANT:
- Deploy one model at a time
- Run all experiments for that model
- Clean up (`kubectl delete deployment`)
- Repeat for next model
- Total time: ~1 week for all 6 models

### Network Configuration

- [ ] NRP network policies understood
- [ ] Ingress requirements identified (if any)
- [ ] Container registry decided: ___________________

---

## 4. Execution Strategy

### Batch Sizes

- [ ] **Query Generation**: Batch size = _____ (recommended: 10-20)
- [ ] **Probability Extraction**: Batch size = _____ (recommended: 5-10)
- [ ] **Structured Extraction**: Batch size = _____ (recommended: 10)

### Parallelism ✅ COMPLETED

- [ ] Run all models in parallel (faster, more GPUs)
- [x] Run models sequentially (slower, fewer GPUs) ✅ SELECTED
- [ ] Hybrid: Small models parallel, large models sequential

**Decision**: Sequential deployment (one model at a time)

**Rationale**: 
- Reduces peak GPU requirement from 11 to 2 GPUs
- Eliminates need for NRP exceptions
- Each model deployment <1 day (acceptable runtime)
- 88% reduction in GPU-hours (216 vs 1,848)

### Error Handling

- [ ] **Max retries**: _____ (recommended: 3)
- [ ] **Timeout per request**: _____ seconds (recommended: 60)
- [ ] **Acceptable failure rate**: _____ % (recommended: <5%)

---

## 5. Data Management

### Output Organization

- [ ] **Directory structure**: `/output/{experiment}/{model}/{timestamp}/`
- [ ] **Filename convention**: `{model}_study{N}_{type}_{timestamp}.csv`
- [ ] **Version control**: Git tag or manual versioning?

**Decision**: ___________________

### Backup Strategy

- [ ] Export to local storage after each phase
- [ ] Keep PVCs mounted (costs storage)
- [ ] Use Kubernetes volume snapshots
- [ ] No backup (re-run if needed)

**Decision**: ___________________

### Data Retention

- [ ] Keep all raw responses: _____ (YES/NO)
- [ ] Keep intermediate structured outputs: _____ (YES/NO)
- [ ] Retention period: ___________________

---

## 6. Validation & Quality

### Validation Thresholds

- [ ] **Structured output success rate**: > _____% (recommended: 95%)
- [ ] **Manual review threshold**: < _____% (recommended: 90%)
- [ ] **Probability sum tolerance**: ± _____ (recommended: 0.01)

### Quality Checks

- [ ] Spot-check N random responses per model: N = _____ (recommended: 10-20)
- [ ] Human review of failed extractions
- [ ] Statistical comparison with original human data

---

## 7. Workflow Orchestration

### Orchestration Tool

- [ ] **kubectl only** (simple, manual)
- [ ] **Argo Workflows** (DAG-based, visual monitoring)
- [ ] **Airflow** (complex, overkill)
- [ ] **Custom shell scripts** (flexible, harder to monitor)

**Decision**: ___________________

### Job Dependencies

- [ ] Sequential execution with manual gates
- [ ] Automated pipeline with dependencies
- [ ] Hybrid: automatic within phases, manual between phases

**Decision**: ___________________

---

## 8. Monitoring & Observability

### Metrics Collection

- [ ] Prometheus + Grafana (if available on NRP)
- [ ] vLLM built-in metrics
- [ ] Custom logging only
- [ ] No metrics (just logs)

**Decision**: ___________________

### Logging Level

- [ ] **Development**: DEBUG (verbose)
- [ ] **Production**: INFO (standard)
- [ ] **Minimal**: WARNING (errors only)

**Decision for Phase 0-2**: ___________________

**Decision for Phase 3+**: ___________________

---

## 9. Prompt Engineering

### Temperature Settings

- [ ] **Experiment 1 (text generation)**: T = _____ (recommended: 0.7)
- [ ] **Experiment 2 (probabilities)**: T = _____ (recommended: 1.0)
- [ ] **Structured extraction**: T = _____ (recommended: 0.0)

### Prompt Format

- [ ] Use chat completion API (for instruction-tuned models)
- [ ] Use completion API (for base models)
- [ ] Test both, choose best

**Decision**: ___________________

### System Prompts

- [ ] Use same system prompt across all models
- [ ] Customize per model based on training
- [ ] No system prompt

**Decision**: ___________________

---

## 10. Timeline & Resources

### Start Date

**Target start date**: ___________________

### Resource Allocation

**Primary engineer**: ___________________

**Hours per week**: ___________________

**Support from team**: ___________________

### Milestones

- [ ] Phase 0 complete by: ___________________
- [ ] Phase 1 complete by: ___________________
- [ ] Phase 2 complete by: ___________________
- [ ] Full experiments complete by: ___________________
- [ ] Analysis complete by: ___________________

### Budget (if applicable)

- [ ] NRP allocation: Free / Paid
- [ ] External API costs (DeepSeek): $ _____ estimated
- [ ] Storage costs: $ _____ estimated
- [ ] Total budget: $ _____

---

## 11. Risk Mitigation

### Top 3 Risks Identified

1. **Risk**: ___________________
   **Mitigation**: ___________________

2. **Risk**: ___________________
   **Mitigation**: ___________________

3. **Risk**: ___________________
   **Mitigation**: ___________________

### Contingency Plans

- [x] If GPU quota insufficient: Use RTX 3090 cluster (50+ nodes, no quota needed) - see GPU_CONTINGENCY_PLAN.md
- [ ] If models too slow: ___________________
- [ ] If extraction quality poor: ___________________
- [ ] If timeline slips: ___________________

---

## 12. Communication Plan

### Status Updates

- [ ] **Frequency**: Daily / Weekly / Bi-weekly
- [ ] **Format**: Email / Slack / Meeting
- [ ] **Stakeholders**: ___________________

### Issue Escalation

- [ ] **NRP issues contact**: ___________________
- [ ] **Research questions contact**: ___________________
- [ ] **Decision authority**: ___________________

---

## 13. Testing Strategy

### Local Testing

- [ ] Test with 1 model (Gemma-2B) locally
- [ ] Test with 10 rows of data
- [ ] Validate output format
- [ ] Test error handling

**Local test date**: ___________________

### Pilot on NRP

- [ ] Deploy 1 model to NRP
- [ ] Run with 10-50 rows
- [ ] Validate full pipeline
- [ ] Measure performance

**Pilot test date**: ___________________

### Full Experiment

- [ ] All models deployed
- [ ] Full dataset (300 + 2,424 rows)
- [ ] Both experiments complete

**Full experiment date**: ___________________

---

## 14. Success Criteria

### Technical Success

- [ ] All models deployed and serving: YES/NO
- [ ] Experiments 1 & 2 complete: YES/NO
- [ ] Data quality validated: YES/NO
- [ ] Outputs ready for analysis: YES/NO

### Research Success

- [ ] Model comparison insights gained: YES/NO
- [ ] Probability distributions collected: YES/NO
- [ ] Publication-quality results: YES/NO
- [ ] Reproducible pipeline: YES/NO

### Minimum Viable Success

What's the minimum outcome you'd accept as "successful"?

___________________________________________________________

___________________________________________________________

---

## 15. Post-Experiment Plan

### Data Archival

- [ ] Export all data to: ___________________
- [ ] Create Zenodo/OSF repository: YES/NO
- [ ] Share with research community: YES/NO

### Cluster Cleanup

- [ ] Delete Jobs: YES (after each model's experiments complete)
- [ ] Delete Deployments: YES (after each model's experiments complete)
- [ ] Delete Pods: YES (automatic when deleting Deployments)
- [ ] Delete PVCs: YES/NO (decide after experiments - model weights can be re-downloaded)
- [ ] Release GPU quota: N/A (using standard allocation, no special quota requested)

### Documentation

- [ ] Write lessons learned document
- [ ] Update plans with actual experience
- [ ] Create troubleshooting FAQ
- [ ] Document for publication methods section

### Future Work

- [ ] Plan for additional experiments: ___________________
- [ ] Plan for different models: ___________________
- [ ] Plan for larger datasets: ___________________

---

## Sign-off

**Decisions reviewed by**: ___________________

**Date**: ___________________

**Approved to proceed**: YES / NO

**Next action**: ___________________

---

## Notes & Comments

### Key Decisions Made (2025)

**Architecture Strategy**:
- Sequential deployment eliminates ALL NRP exception requirements
- Using A100-80GB instead of A100-40GB reduces Llama-70B from 4 GPUs to 2 GPUs
- All resource limits within 20% of requests (NRP policy compliant)
- Using Deployments (not StatefulSets) for stateless model servers

**Cost Efficiency**:
- GPU-hours: 216 (vs 1,848 for parallel deployment) - 88% reduction
- Peak GPUs: 2 (vs 11 for parallel deployment)
- Total runtime: ~1 week for all 6 models

**Compliance Status**:
✅ All planning documents updated for NRP compliance
✅ No exceptions required
✅ See FINAL_MODEL_LINEUP.md, NRP_POLICY_COMPLIANCE_AUDIT.md, and COMPLIANCE_ISSUES_FOUND.md

**Pending Decisions**:
- Quantization strategy (fp16 vs AWQ 4-bit)
- Context length (2048 vs 4096 tokens)
- Batch sizes for different experiment types
- Orchestration tool (kubectl vs Argo Workflows)
- Monitoring approach
- Temperature settings
- Error handling parameters

---

**Save this checklist and refer back to it throughout the project!**
