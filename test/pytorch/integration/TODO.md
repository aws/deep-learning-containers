# Integration Tests — PyTorch AL2023 DLC

These tests require real multi-node infrastructure (placement groups, EFA devices, multiple instances) and cannot run in the current DevBox or CI environment.

## Planned Tests

### EFA / NCCL Transport

- [ ] NCCL all_reduce over EFA (verify `NET/OFI` transport, not `NET/Socket`)
- [ ] NCCL bandwidth test across 2 nodes (measure GB/s, compare against EFA spec)
- [ ] Multi-node DDP training over EFA with gradient sync verification
- [ ] EFA multi-rail (multiple ENIs per instance, e.g. p5.48xlarge with 32 EFA devices)

### Multi-Node Training at Scale

- [ ] FSDP training across 2+ nodes with checkpoint save/load
- [ ] DeepSpeed ZeRO-3 across 2+ nodes (ZeRO-3 partitions optimizer+gradients+parameters across nodes)
- [ ] Large model training (model that doesn't fit on a single GPU)
- [ ] Elastic training — node join/leave during training (torch elastic)

### SageMaker Integration

- [ ] Training job launch with custom image URI
- [ ] Multi-node SageMaker training job (2+ instances)
- [ ] SageMaker distributed data parallel (SMDDP) if applicable
- [ ] S3 checkpoint upload/download during training

### EKS Integration

- [ ] PyTorchJob (Kubeflow training operator) single-node
- [ ] PyTorchJob multi-node with NCCL
- [ ] Pod scheduling with GPU resource limits
- [ ] Shared filesystem (FSx/EFS) checkpoint persistence

### Performance Baselines

- [ ] Single-GPU throughput (samples/sec) for reference model
- [ ] Multi-GPU scaling efficiency (2, 4, 8 GPUs)
- [ ] Multi-node scaling efficiency (2, 4 nodes)
- [ ] AMP (fp16/bf16) vs fp32 throughput comparison

## Infrastructure Requirements

| Test Category       | Min Instances | Instance Type  | EFA Required         | Placement Group |
| ------------------- | ------------- | -------------- | -------------------- | --------------- |
| EFA / NCCL          | 2             | p4d.24xlarge+  | Yes                  | Yes (cluster)   |
| Multi-node training | 2             | g6.12xlarge+   | No (but recommended) | Recommended     |
| SageMaker           | N/A           | Managed        | N/A                  | N/A             |
| EKS                 | EKS cluster   | GPU node group | Optional             | Optional        |
| Performance         | 1-4           | p4d.24xlarge+  | Yes for multi-node   | Yes             |
