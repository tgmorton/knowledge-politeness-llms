# Grace Project - Kubernetes Implementation Plan
## Executive Summary

**Date**: November 12, 2025  
**Project**: Grace Project - Distributed LLM Experiments on National Research Platform  
**Status**: Planning Phase Complete, Ready for Implementation

---

## Overview

This planning package provides a complete blueprint for running the Grace Project experiments on a Kubernetes cluster at the National Research Platform (NRP). The system will self-host multiple open-source transformer models, generate responses to experimental scenarios, extract structured outputs, and produce analysis-ready datasets.

### Current State

You have completed two studies using OpenAI's API:
- **Study 1**: 300 rows analyzing knowledge and information access
- **Study 2**: 2,424 rows analyzing politeness in speaker utterances

Both studies have existing data, R analysis scripts, and published/in-progress research outputs.

### Proposed Enhancement

Run two new experiments using **self-hosted open-source models** (Gemma, Llama, etc.) on NRP:

1. **Experiment 1**: Generate raw text responses (similar to current OpenAI approach)
2. **Experiment 2**: Extract probability distributions over response options using logprobs
   - **Study 1**: For each state (0,1,2,3), extract probability distribution over percentage values (0%, 10%, 20%, ..., 100%)
   - **Study 2**: Extract probabilities for categorical response options
   - Creates rich uncertainty measures: mean, std deviation, entropy, full density plots

This will enable:
- Comparison of model behaviors across architectures
- **Uncertainty quantification**: Full probability density over model beliefs (not just argmax)
- Analysis of model confidence and calibration
- Full control over inference parameters
- Cost-effective large-scale experimentation
- Reproducible research with versioned models

---

## Documents in This Planning Package

| Document | Purpose | Key Content |
|----------|---------|-------------|
| **00_EXECUTIVE_SUMMARY.md** | This document | High-level overview and quick start |
| **01_ARCHITECTURE_OVERVIEW.md** | System design | Architecture principles, components, workflows |
| **02_KUBERNETES_INFRASTRUCTURE.md** | K8s specifications | Complete YAML manifests for deployment |
| **03_IMPLEMENTATION_ROADMAP.md** | Execution plan | 7-week phased timeline with tasks |
| **04_MODEL_SERVING_SPECS.md** | Model configurations | vLLM setup, prompts, optimization |
| **05_DATA_PIPELINE_DESIGN.md** | Data processing | Python scripts, schemas, validation |

---

## Architecture at a Glance

```
┌────────────────────────────────────────────────────────────┐
│  INPUT: study1.csv (300 rows) + study2.csv (2,424 rows)   │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│  MODEL SERVING: vLLM StatefulSets                          │
│  • Gemma-2B, Gemma-9B, Llama-8B, Llama-70B, DeepSeek      │
│  • OpenAI-compatible API endpoints                         │
│  • GPU-accelerated inference with PagedAttention           │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│  EXPERIMENT 1: Query Generation Jobs (Parallel)            │
│  • Generate raw text responses from each model             │
│  • Output: {model}_study{1|2}_raw_responses.csv           │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│  STRUCTURED OUTPUT: DeepSeek Extraction Jobs               │
│  • Parse responses into CSV structure                      │
│  • Validate against JSON schemas                           │
│  • Output: {model}_study{1|2}_structured.csv              │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│  EXPERIMENT 2: Probability Extraction Jobs (Parallel)      │
│  • Extract logprobs for response tokens                    │
│  • Compute probability distributions                       │
│  • Output: {model}_study{1|2}_probabilities.csv           │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│  OUTPUT: Analysis-ready datasets for R scripts             │
└────────────────────────────────────────────────────────────┘
```

---

## Resource Requirements

### Compute (Sequential Deployment)
- **Peak GPUs**: 2-4 NVIDIA A100 GPUs (one model deployed at a time)
- **CPUs**: ~40 cores per model
- **Memory**: ~128GB RAM per model (high RAM exception may be needed for some models)
- **Storage**: 600GB persistent storage

### Time Estimates
- **Setup & Testing**: 1-2 weeks (local dev → RTX 3090 testing → A100 production)
- **Model Deployment**: 1 day per model × 6 models = 6 days total (sequential)
- **Experiment Execution**: 1-2 days per run
- **Analysis**: 1 week
- **Total Project**: 4-6 weeks

### NRP GPU Hours
- **Estimated**: ~168 GPU-hours per model × 6 models = ~1,000 GPU-hours total
- **Cost**: Free on NRP (subject to allocation)

---

## Key Technologies

| Component | Technology | Why |
|-----------|-----------|-----|
| **Model Serving** | vLLM | 24x faster than HuggingFace, supports logprobs |
| **Container Orchestration** | Kubernetes | Native to NRP, scalable, fault-tolerant |
| **Language** | Python 3.11+ | Async/await for concurrency, strong ecosystem |
| **Data Processing** | Pandas + asyncio | Familiar, efficient for CSV operations |
| **Validation** | Pydantic | Type-safe schema validation |
| **Workflow** | Argo Workflows (optional) | DAG-based orchestration with monitoring |

---

## Recommended Models

### Final Lineup (Sequential Deployment)

| Model | Size | GPU | Use Case | Priority |
|-------|------|-----|----------|----------|
| Gemma 2 - 2B | 2B | 1x A100 40GB | Fast baseline | High |
| Gemma 2 - 9B | 9B | 1x A100 40GB | Mid-range quality | High |
| Llama 3.1 - 8B | 8B | 1x A100 40GB | Alternative mid-range | Medium |
| Llama 3.1 - 70B | 70B | 2x A100 80GB | High quality | High |
| DeepSeek-V3 | 37B active | 2x A100 80GB | Structured extraction | High |

**Deployment Strategy**: Deploy one model at a time, run experiments, tear down, deploy next model

**Peak Resources**: 2x A100 80GB (for Llama-70B or DeepSeek-V3)

### NRP Exception Requirements
- **High RAM**: May be needed for Llama-70B and DeepSeek-V3 (128GB+ RAM)
- **2-GPU**: Needed for Llama-70B and DeepSeek-V3
- **No deployment duration exception needed**: Each model deployed <1 day sequentially

---

## Implementation Phases

### Phase 0: Local Development (Week 1)
- Set up local development environment
- Test model downloads and vLLM locally
- Create project structure
- Test Python scripts with small datasets

### Phase 1: RTX 3090 Testing (Week 2)
- Deploy to RTX 3090 for initial testing
- Validate Docker images and Kubernetes configs
- Run pilot experiments with 1-2 models
- Debug and optimize performance

### Phase 2: Sequential A100 Deployment (Days 1-6)
- **Day 1**: Deploy Gemma-2B, run experiments, tear down
- **Day 2**: Deploy Gemma-9B, run experiments, tear down
- **Day 3**: Deploy Llama-8B, run experiments, tear down
- **Day 4**: Deploy Llama-70B, run experiments, tear down
- **Day 5**: Deploy DeepSeek-V3, run experiments, tear down
- **Day 6**: Buffer for any reruns or issues

### Phase 3: Experiment 1 - Raw Text Generation
- Generate raw text responses (included in each model's deployment day)
- Run structured output extraction with DeepSeek
- Validate data quality

### Phase 4: Experiment 2 - Probability Extraction
- Extract probability distributions (included in each model's deployment day)
- Validate probability distributions
- Compare with Experiment 1 results

### Phase 5: Analysis (Week 4)
- Adapt R scripts for multiple models
- Generate comparative visualizations
- Run statistical analyses

### Phase 6: Documentation (Week 4-5)
- Archive data and code
- Write reproducibility guide
- Document lessons learned

---

## Critical Success Factors

### Technical
1. **vLLM Configuration**: Get GPU memory utilization right to avoid OOM errors
2. **Prompt Engineering**: Design prompts that elicit structured responses
3. **Logprobs Extraction**: Correctly normalize probabilities from token logprobs
4. **Error Handling**: Implement robust retry logic and checkpointing

### Operational
1. **NRP Rules**: Follow NRP cluster policies (sequential deployment, proper teardown)
2. **Sequential Workflow**: Deploy one model at a time, complete all experiments, tear down before next
3. **Data Management**: Preserve data integrity with PVCs and backups
4. **Monitoring**: Track job progress and model performance
5. **Phased Testing**: Validate on local → RTX 3090 → A100 before production runs

### Research
1. **Prompt Consistency**: Use same prompts across models for fair comparison
2. **Temperature Settings**: Use 0.7 for text generation, 1.0 for probabilities
3. **Validation**: Ensure output quality meets research standards
4. **Reproducibility**: Version everything (models, code, configs)

---

## Key Decisions Needed

Before starting implementation, decide:

1. **Final Model Selection**: Which specific models and sizes?
   - Recommendation: Start with Gemma-2B, Gemma-9B, Llama-70B, DeepSeek-V3

2. **DeepSeek Approach**: Self-host or use API?
   - Recommendation: Start with API for simplicity, self-host if high volume

3. **Quantization**: Use full precision or quantized models?
   - Recommendation: Start with fp16, use AWQ if GPU-constrained

4. **Orchestration**: Use Argo Workflows or simple kubectl?
   - Recommendation: Start simple with kubectl, add Argo if needed

5. **Batch Size**: How many rows to process in parallel?
   - Recommendation: Start with 10, tune based on throughput

6. **Validation Threshold**: What % extraction accuracy is acceptable?
   - Recommendation: Aim for >95%, manual review anything <90%

---

## Quick Start Guide

### 1. Get NRP Access
```bash
# Get kubeconfig from NRP portal
export KUBECONFIG=~/nrp-kubeconfig.yaml
kubectl cluster-info
kubectl get nodes
```

### 2. Test Model Locally
```bash
# Install vLLM
pip install vllm

# Download and run Gemma-2B
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-2-2b-it \
  --dtype float16

# Test endpoint
curl http://localhost:8000/v1/models
```

### 3. Clone and Setup Project
```bash
cd ~/10-GraceProject
mkdir -p kubernetes/{base,models,jobs} docker src tests

# Copy CSV files to upload
cp data/study1.csv data/study2.csv ./uploads/
```

### 4. Read the Plans
- Start with `01_ARCHITECTURE_OVERVIEW.md` for big picture
- Review `03_IMPLEMENTATION_ROADMAP.md` for detailed timeline
- Use `02_KUBERNETES_INFRASTRUCTURE.md` as reference during deployment

### 5. Begin Phase 0
Follow Phase 0 checklist in `03_IMPLEMENTATION_ROADMAP.md`

---

## Risk Management

| Risk | Impact | Mitigation |
|------|--------|------------|
| Insufficient GPU quota | Low | Sequential deployment requires only 2-4 GPUs peak |
| Model OOM errors | Medium | Test on RTX 3090 first, tune memory settings |
| Job failures | Medium | Implement checkpointing and retry logic |
| Data corruption | High | Use PVCs with backups, validate outputs |
| API rate limits (DeepSeek) | Medium | Self-host or batch requests |
| Slow inference | Medium | Optimize batch sizes, use vLLM features |
| Extraction failures | Medium | Improve prompts, manual review threshold |
| Sequential delays | Medium | 6-day window provides buffer for issues |

---

## Expected Outputs

### Data Files
- **Experiment 1**: 8-10 raw response CSVs + structured CSVs (per model × study)
- **Experiment 2**: 8-10 probability distribution CSVs (per model × study)
- **Total**: ~40-50 CSV files

### Analysis
- Updated R scripts for multi-model comparison
- Comparative visualizations (model × condition plots)
- Statistical reports (ANOVA with model as factor)
- Model performance comparison table

### Documentation
- Complete reproduction guide
- Kubernetes deployment guide
- Lessons learned document
- Troubleshooting FAQ

---

## Next Actions

### Immediate (This Week)
1. [ ] Set up local development environment
2. [ ] Test vLLM with Gemma-2B locally
3. [ ] Request NRP exceptions (high RAM for 2 models, 2-GPU for 2 models)
4. [ ] Make final decision on model lineup
5. [ ] Review all planning documents with team

### Short-term (Week 2)
1. [ ] Implement Python scripts for query generation
2. [ ] Build Docker images
3. [ ] Create Kubernetes manifests
4. [ ] Deploy to RTX 3090 for testing
5. [ ] Run pilot test with 1-2 models on 10 rows

### Medium-term (Week 3)
1. [ ] Execute sequential A100 deployment (6 days, one model per day)
2. [ ] Run both experiments for each model
3. [ ] Validate all outputs
4. [ ] Archive data after each model

### Long-term (Week 4-5)
1. [ ] Run analyses and generate visualizations
2. [ ] Write up results
3. [ ] Document lessons learned
4. [ ] Submit for publication

---

## Support & Questions

### Documentation
- All plans in `plans/` folder
- Kubernetes manifests will be in `kubernetes/`
- Python scripts will be in `src/`

### Key Contacts
- **NRP Support**: support@nrp-nautilus.io (example)
- **Your Team**: [Add team members]

### External Resources
- vLLM Docs: https://docs.vllm.ai/
- NRP Portal: https://nautilus.optiputer.net/
- HuggingFace Models: https://huggingface.co/models

---

## Opinion: Recommended Approach

As an opinionated engineer, here's what I recommend:

### Sequential Deployment is Key
1. **Deploy one model at a time**. This is NRP-compliant and reduces complexity. Complete all experiments for one model, archive the data, tear down, then move to the next.

2. **Follow the phased testing approach**: Local → RTX 3090 → A100. Don't skip straight to A100. Catch issues early when debugging is easier.

3. **Use DeepSeek API initially**. Self-hosting is complex; validate your extraction prompts with the API first. You can always switch to self-hosted later if cost becomes an issue.

4. **Skip Argo Workflows initially**. Use simple Kubernetes Jobs first. Only add Argo if you need complex dependencies and retries.

### Prioritize Robustness Over Speed
1. **Checkpoint aggressively**. Save after every batch, even if it's slower. With sequential deployment, you can't easily re-run a model.

2. **Log everything**. You'll thank yourself when debugging at 2 AM.

3. **Validate early and often**. Don't wait until all 2,424 rows are processed to discover your extraction prompts are failing.

### Design for Iteration
1. **Make prompts configurable**. You'll want to tweak them based on initial results.

2. **Version your outputs**. Use timestamps in filenames: `gemma-2b_study1_20250112_143022.csv`

3. **Keep raw responses**. Always save the raw model output before structuring. You might want to re-extract with different prompts.

### Sequential Deployment Benefits
- **NRP Compliant**: No need for >2 week deployment exception
- **Lower GPU quota**: Only 2-4 GPUs at peak vs 8-10 simultaneously
- **Easier debugging**: Focus on one model at a time
- **Lower risk**: Issues with one model don't block others
- **Total time**: Still only 6 days for all models (vs weeks of simultaneous deployment)

### Key Insight
The hardest part won't be the infrastructure—it will be **prompt engineering** to get consistent structured outputs. Budget extra time for this. Run lots of small tests before committing to 2,424 row runs. The sequential approach gives you time to refine prompts between models.

---

## Conclusion

This planning package provides everything needed to execute the Grace Project experiments on NRP. The architecture is proven (vLLM + Kubernetes is battle-tested), the resource requirements are realistic and NRP-compliant, and the sequential deployment approach reduces risk and complexity.

**Estimated effort**: 4-6 weeks with one engineer working full-time

**Complexity**: Medium (sequential deployment simplifies orchestration)

**Risk level**: Low-Medium (phased testing and sequential approach minimize issues)

**Expected outcome**: High-quality dataset enabling novel cross-model comparisons

**NRP Compliance**: Sequential deployment requires only 2-4 GPUs peak, ~1,000 GPU-hours total, and no deployment duration exceptions

---

**Ready to proceed?** Start with Phase 0 in `03_IMPLEMENTATION_ROADMAP.md`

**Questions?** All technical details are in the other planning documents

**Need help?** The scripts in `05_DATA_PIPELINE_DESIGN.md` are 80% complete—just need testing and iteration

**Good luck! 🚀**
