# Grace Project - Implementation Plans

This folder contains comprehensive planning documents for running the Grace Project experiments on the National Research Platform using Kubernetes and self-hosted LLM models.

## 📋 Document Index

### Start Here
- **[00_EXECUTIVE_SUMMARY.md](00_EXECUTIVE_SUMMARY.md)** - High-level overview, quick start guide, and recommendations

### Critical NRP Documentation ⚠️ **READ BEFORE IMPLEMENTING**
- **[../docs/NRP_CLUSTER_GUIDE.md](../docs/NRP_CLUSTER_GUIDE.md)** - NRP Nautilus cluster architecture, policies, and workload types
- **[../docs/NRP_STORAGE_GUIDE.md](../docs/NRP_STORAGE_GUIDE.md)** - NRP Ceph storage classes and restrictions

### Core Planning Documents
1. **[01_ARCHITECTURE_OVERVIEW.md](01_ARCHITECTURE_OVERVIEW.md)** - System architecture, components, and design principles
2. **[02_KUBERNETES_INFRASTRUCTURE.md](02_KUBERNETES_INFRASTRUCTURE.md)** - Complete Kubernetes manifests and configurations
3. **[03_IMPLEMENTATION_ROADMAP.md](03_IMPLEMENTATION_ROADMAP.md)** - 7-week phased implementation timeline
4. **[04_MODEL_SERVING_SPECS.md](04_MODEL_SERVING_SPECS.md)** - vLLM configuration, model setup, and optimization
5. **[05_DATA_PIPELINE_DESIGN.md](05_DATA_PIPELINE_DESIGN.md)** - Data processing scripts, schemas, and validation

### Specialized Documents
- **[STUDY1_EXP2_CLARIFICATION.md](STUDY1_EXP2_CLARIFICATION.md)** - Detailed explanation of Study 1 Experiment 2 probability extraction approach
- **[DECISION_CHECKLIST.md](DECISION_CHECKLIST.md)** - Pre-implementation decisions and sign-off checklist

## 🎯 Project Goals

Run two experiments using self-hosted open-source models (Gemma, Llama, etc.):

### Experiment 1: Raw Response Generation
- Generate text responses to Study 1 (knowledge/access) and Study 2 (politeness) scenarios
- Extract structured outputs using DeepSeek

### Experiment 2: Probability Distribution Extraction
- Extract token-level probabilities for response options
- Compute normalized probability distributions

## 🏗️ Architecture Summary

```
Input CSVs → vLLM Model Servers → Query Jobs → DeepSeek Extraction → Analysis-Ready Data
            (Kubernetes)         (Parallel)    (Structured)         (CSV/JSON)
```

## 📊 Resource Requirements

- **GPUs**: 8-10 NVIDIA A100 GPUs
- **Timeline**: 6-9 weeks
- **Storage**: 600GB persistent
- **GPU Hours**: ~1,860 hours

## 🚀 Quick Start

1. Read **00_EXECUTIVE_SUMMARY.md** for the big picture
2. Review **03_IMPLEMENTATION_ROADMAP.md** for the execution plan
3. Start with Phase 0 (Preparation) tasks
4. Use other documents as references during implementation

## 📝 Key Technologies

- **Model Serving**: vLLM (OpenAI-compatible API)
- **Orchestration**: Kubernetes on NRP
- **Language**: Python 3.11+ with asyncio
- **Data**: Pandas, Pydantic for validation
- **Optional**: Argo Workflows for complex orchestration

## 🎓 Recommended Reading Order

**For Engineers Implementing**:
1. Executive Summary → Implementation Roadmap → Architecture Overview
2. Then dive into specific docs as needed during each phase

**For Research Team**:
1. Executive Summary → Data Pipeline Design
2. Focus on schemas and validation strategies

**For System Administrators**:
1. Executive Summary → Kubernetes Infrastructure → Model Serving Specs
2. Focus on resource allocation and deployment

## ⚠️ Important Notes

- All plans assume access to National Research Platform (NRP) Kubernetes cluster
- NRP-specific configurations (storage classes, node selectors) need to be updated based on actual cluster setup
- Cost estimates are based on free NRP allocation; adjust if using commercial cloud
- Model selection can be adjusted based on available GPU quota

## 🔧 Decisions Needed Before Implementation

1. Final model lineup (see Executive Summary recommendations)
2. DeepSeek approach: API or self-hosted
3. Quantization strategy
4. Batch sizes and parallelism levels
5. Validation thresholds

## 📖 Additional Resources

- vLLM Documentation: https://docs.vllm.ai/
- Kubernetes Documentation: https://kubernetes.io/docs/
- HuggingFace Models: https://huggingface.co/models
- NRP Portal: https://nautilus.optiputer.net/

## 🤝 Contributing

As you implement, please:
- Update plans with lessons learned
- Document any deviations from the plan
- Add troubleshooting notes
- Version control all changes

## 📧 Questions?

Refer to the Executive Summary's "Support & Questions" section for resources.

---

**Last Updated**: November 12, 2025  
**Status**: Planning Phase Complete ✅  
**Next Phase**: Phase 0 - Preparation
