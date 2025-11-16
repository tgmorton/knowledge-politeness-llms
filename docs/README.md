# Grace Project Documentation

This folder contains essential reference documentation for implementing the Grace Project on the National Research Platform.

## 📚 Documents

### NRP_CLUSTER_GUIDE.md ⚠️ **READ FIRST**

**Essential guide to NRP Nautilus Kubernetes cluster**

This document covers the NRP cluster architecture, access methods, and critical policies that **must be understood** before deployment:

- Cluster architecture (75+ distributed sites)
- Available resources (CPUs, GPUs, FPGAs, memory)
- Access methods (JupyterHub, Coder, kubectl)
- **Kubernetes workload types** (Pods, Jobs, Deployments)
- **Critical policies**: 2-week Deployment deletion, 6-hour Pod limit
- Resource allocation (requests vs limits)
- System load monitoring
- Exception request process

**Key Takeaways**:
1. ✅ Use **Jobs** for batch processing (no time limit)
2. ✅ Use **Deployments** for model serving (need exception for >2 weeks)
3. ⚠️ **Request exceptions early**: >2 weeks runtime, >2 GPUs per pod
4. ❌ Never use standalone Pods (auto-deleted after 6 hours)
5. ✅ Join Matrix chat for support and exception requests

---

### NRP_STORAGE_GUIDE.md ⚠️ **MUST READ**

**Critical reference for NRP Ceph storage system**

This document is **essential reading** before implementing any part of the Grace Project. It covers:

- Available storage classes (CephFS vs RBD)
- **Critical restrictions** (no pip/conda on CephFS, write conflict avoidance)
- Best practices for file access and data management
- Performance optimization strategies
- Troubleshooting common storage issues

**Key Takeaways**:
1. ❌ **Never install pip/conda packages on CephFS volumes**
2. ✅ Use `rook-ceph-block` (RBD) for model weights (fastest)
3. ✅ Use `rook-cephfs` (CephFS) for shared data
4. ✅ Each job must write to unique files (no conflicts)
5. ❌ Never use `rook-cephfs-ucsd` for persistent data (may be purged)

---

## Why This Documentation Matters

### Common Pitfalls Avoided

**Without reading NRP_STORAGE_GUIDE.md, you might:**
- ❌ Install Python packages on CephFS → Performance degradation for all users
- ❌ Have multiple jobs write to same file → Data corruption
- ❌ Use wrong storage class → Poor performance or data loss
- ❌ Not understand NRP-specific policies → Violate cluster rules

**After reading NRP_STORAGE_GUIDE.md, you will:**
- ✅ Use correct storage classes for each component
- ✅ Follow NRP best practices
- ✅ Avoid common mistakes
- ✅ Achieve optimal performance
- ✅ Keep your data safe

---

## Integration with Planning Documents

The NRP storage information has been integrated throughout the planning documents in `../plans/`:

### Updated Sections

1. **02_KUBERNETES_INFRASTRUCTURE.md**
   - All PVC definitions use correct storage classes (`rook-ceph-block`, `rook-cephfs`)
   - Init containers avoid installing packages on CephFS
   - Comments explain storage choices

2. **05_DATA_PIPELINE_DESIGN.md**
   - Python scripts create unique output files per job
   - No pip installations on mounted volumes
   - Proper use of local temp storage

3. **DECISION_CHECKLIST.md**
   - Storage class section pre-filled with NRP classes
   - Critical rules checklist added
   - References NRP_STORAGE_GUIDE.md

---

## Quick Reference

### Storage Class Decision Matrix

| Data Type | Size | Access Pattern | Storage Class | Rationale |
|-----------|------|----------------|---------------|-----------|
| Input CSVs | <10MB | Multiple reads | `rook-cephfs` | Small, shared read access |
| Model weights | 10-150GB | Single pod, sequential reads | `rook-ceph-block` | Large, fastest I/O needed |
| Output CSVs | 10-100MB | Multiple writes | `rook-cephfs` | Shared write (unique files) |
| Logs | <1GB | Multiple writes | `rook-cephfs` | Shared write (unique files) |
| Temp scratch | Variable | Single pod, any | `emptyDir` | Local, auto-cleaned |

### Critical Commands

**Check available storage classes:**
```bash
kubectl get storageclass
```

**Check PVC status:**
```bash
kubectl get pvc -n grace-experiments
```

**Monitor storage usage:**
```bash
kubectl exec -it <pod-name> -n grace-experiments -- df -h /mount-path
```

**Upload data to CephFS:**
```bash
kubectl run uploader --image=busybox --overrides='...'
kubectl cp local-file.csv uploader:/mount-path/
```

---

## For Future Reference

As you implement the Grace Project:

1. **Before writing any Kubernetes manifests**: Review NRP_STORAGE_GUIDE.md
2. **Before writing Python scripts**: Check best practices for file access
3. **When debugging storage issues**: Consult troubleshooting section
4. **When optimizing performance**: Review performance characteristics table

---

## Additional Resources

- **NRP Portal**: https://nautilus.optiputer.net/
- **NRP Documentation**: Check with NRP admin for latest docs
- **Ceph Grafana Dashboard**: Monitor cluster-wide storage usage
- **Planning Documents**: `../plans/` for complete implementation guide

---

## Document Maintenance

This documentation reflects NRP configuration as of **November 2025**.

**If you discover updates or corrections:**
1. Update this documentation
2. Update affected sections in `../plans/`
3. Document changes in git commit message

**NRP-specific items that may change:**
- Storage class names (unlikely)
- Available storage pools
- Quota policies
- GPU node labels

---

## Questions?

If you have questions about NRP storage:
1. Review NRP_STORAGE_GUIDE.md thoroughly
2. Check NRP official documentation
3. Contact NRP support: support@nrp-nautilus.io
4. Check with other NRP users in your research group

**Don't guess!** Incorrect storage usage can affect other users and may result in data loss.
