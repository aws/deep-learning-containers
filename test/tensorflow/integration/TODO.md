# Integration Tests — TensorFlow AL2023 DLC

These tests require real multi-node infrastructure (placement groups, EFA devices, multiple instances) and cannot run in the current CI environment.

## Planned Tests

### EFA / NCCL Transport

- [ ] NCCL all_reduce over EFA via `MultiWorkerMirroredStrategy` (verify `NET/OFI` transport, not `NET/Socket`)
- [ ] NCCL bandwidth test across 2 nodes (measure GB/s, compare against EFA spec)
- [ ] Multi-node `MultiWorkerMirroredStrategy` training over EFA with gradient sync verification
- [ ] EFA multi-rail (multiple ENIs per instance, e.g. p5.48xlarge with 32 EFA devices)

### Multi-Node Training at Scale

- [ ] `MultiWorkerMirroredStrategy` across 2+ nodes with checkpoint save/load
- [ ] `ParameterServerStrategy` across 2+ nodes (worker / PS split)
- [ ] Large model training (model that doesn't fit on a single GPU)
- [ ] Elastic training — node join/leave during training (preemption tolerance)

### SageMaker Integration

- [x] Training job launch with custom image URI
- [x] Multi-node SageMaker training job (2+ instances) via `Mpi()` distribution
- [ ] S3 checkpoint upload/download during training
- [ ] SageMaker training with `SM_HOSTS`-derived `TF_CONFIG` (no MPI fallback path)

### EKS Integration

- [ ] TFJob (Kubeflow training operator) single-node
- [ ] TFJob multi-node with NCCL (`MultiWorkerMirroredStrategy`)
- [ ] Pod scheduling with GPU resource limits
- [ ] Shared filesystem (FSx/EFS) checkpoint persistence

### Performance Baselines

- [ ] Single-GPU throughput (samples/sec) for reference model
- [ ] Multi-GPU scaling efficiency (2, 4, 8 GPUs) via `MirroredStrategy`
- [ ] Multi-node scaling efficiency (2, 4 nodes) via `MultiWorkerMirroredStrategy`
- [ ] Mixed precision (fp16/bf16) vs fp32 throughput comparison

## Infrastructure Requirements

| Test Category       | Min Instances | Instance Type  | EFA Required         | Placement Group |
| ------------------- | ------------- | -------------- | -------------------- | --------------- |
| EFA / NCCL          | 2             | p4d.24xlarge+  | Yes                  | Yes (cluster)   |
| Multi-node training | 2             | g6.12xlarge+   | No (but recommended) | Recommended     |
| SageMaker           | N/A           | Managed        | N/A                  | N/A             |
| EKS                 | EKS cluster   | GPU node group | Optional             | Optional        |
| Performance         | 1-4           | p4d.24xlarge+  | Yes for multi-node   | Yes             |
