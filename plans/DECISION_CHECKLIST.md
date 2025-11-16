# Pre-Implementation Decision Checklist

Before beginning implementation, make decisions on the following items. Check off each item as decisions are made.

---

## 1. Model Selection

### Primary Models (Choose 3-5)

- [ ] **Gemma 2 - 2B** (Recommended: ✅ YES - fast baseline)
- [ ] **Gemma 2 - 9B** (Recommended: ✅ YES - mid-range quality)
- [ ] **Gemma 2 - 27B** (Optional: only if sufficient GPU)
- [ ] **Llama 3.1 - 8B** (Recommended: Maybe - alternative to Gemma-9B)
- [ ] **Llama 3.1 - 70B** (Recommended: ✅ YES - high quality)
- [ ] **Phi-3 Mini (3.8B)** (Optional: Microsoft's efficient model)
- [ ] **Mistral 7B v0.3** (Optional: strong performance/size ratio)

**Decision**: Which models? _________________________________

**GPU Requirements**: _____ A100 40GB + _____ A100 80GB

### Extraction Model

- [ ] **DeepSeek-V3** (self-hosted, 2x A100 80GB)
- [ ] **DeepSeek API** (external, no GPU required)
- [ ] **Llama 3.1 70B** (repurpose existing model for extraction)

**Decision**: ___________________

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

### Storage Classes (NRP Ceph)

**Reference**: See `docs/NRP_STORAGE_GUIDE.md` for complete details

**Configured storage classes**:
- [x] Model weights (RBD, fastest): `rook-ceph-block`
- [x] Input data (CephFS, shared read): `rook-cephfs`
- [x] Output data (CephFS, shared write): `rook-cephfs`
- [x] Logs (CephFS, shared write): `rook-cephfs`

**Critical Rules Acknowledged**:
- [ ] ❌ Never install pip/conda on CephFS volumes
- [ ] ✅ Each job writes to unique files (no conflicts)
- [ ] ✅ Use RBD for model weights (faster than CephFS)
- [ ] ❌ Never use `rook-cephfs-ucsd` (may be purged)

### GPU Quota and NRP Exceptions

**Reference**: See `docs/NRP_CLUSTER_GUIDE.md` for NRP cluster policies

**Current GPU allocation**: _____ GPUs

**Requested GPU allocation**: _____ GPUs (recommend 10-15 A100s)

- [ ] GPU quota request submitted (if needed)
- [ ] GPU quota request approved

**NRP Exception Requests** (CRITICAL):

- [ ] **Join NRP Matrix chat**: https://matrix.to/#/#nrp:matrix.org
- [ ] **Request exception**: Deployments to run >2 weeks
  - Default: Deployments auto-deleted after 2 weeks
  - Need: 2-4 weeks for Grace Project model servers
  - Status: ___________________
- [ ] **Request exception**: >2 GPUs per pod (for Llama-70B)
  - Default: Max 2 GPUs per pod
  - Need: 4 GPUs for Llama-70B StatefulSet
  - Status: ___________________
- [ ] **Request exception**: >32GB RAM per pod (if needed)
  - Default: Max 32GB RAM per pod
  - May need for some models
  - Status: ___________________

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

### Parallelism

- [ ] Run all models in parallel (faster, more GPUs)
- [ ] Run models sequentially (slower, fewer GPUs)
- [ ] Hybrid: Small models parallel, large models sequential

**Decision**: ___________________

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

- [ ] If GPU quota insufficient: ___________________
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

- [ ] Delete Jobs: YES/NO
- [ ] Delete Pods: YES/NO
- [ ] Delete PVCs: YES/NO (or keep for future runs)
- [ ] Release GPU quota: YES/NO

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

Use this space for additional notes, concerns, or open questions:

___________________________________________________________

___________________________________________________________

___________________________________________________________

___________________________________________________________

___________________________________________________________

---

**Save this checklist and refer back to it throughout the project!**
