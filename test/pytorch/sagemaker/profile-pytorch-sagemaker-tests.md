# PyTorch SageMaker Test Profiling

**Tenet: Use a minimal set of tests to cover all SageMaker-related PyTorch functionality. No redundant tests.**

Source: [V1 master branch](https://github.com/aws/deep-learning-containers/tree/master/test/sagemaker_tests/pytorch/training)

## Functionalities Being Tested

There are 7 functionalities at 3 priority levels. "SM training job launch" is not listed separately — it's implicitly validated by every test.

**P0 — Must test:**

| #   | Functionality                                              | Why P0                                                                             |
| --- | ---------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| F1  | Distributed training (nccl)                                | 90%+ of customers use nccl for GPU training. If this breaks, the image is useless. |
| F2  | torch.distributed primitives (all_reduce, broadcast, etc.) | Validates the communication layer that all distributed training depends on         |

**P1 — Nice to have:**

| #   | Functionality             | Why P1                                                              |
| --- | ------------------------- | ------------------------------------------------------------------- |
| F3  | EFA connectivity          | Important for large-scale training, but not all customers use EFA   |
| F4  | GDRCopy (GPU Direct RDMA) | Performance optimization, depends on EFA                            |
| F5  | fastai training           | High-level library on top of PyTorch — in V1, should validate in V2 |

**P2 — Don't need in SM tests (covered elsewhere or low value):**

| #   | Functionality                | Why P2                                                                              |
| --- | ---------------------------- | ----------------------------------------------------------------------------------- |
| F6  | torch.compile/inductor on SM | Already covered by EC2 single-GPU tests — SM entrypoint doesn't affect the compiler |
| F7  | Heterogeneous Clusters API   | Newer SM API, low usage                                                             |

**Remove:**

| #   | Functionality               | Why                                                      |
| --- | --------------------------- | -------------------------------------------------------- |
| D1  | SM Model Parallel (SMP)     | Deprecated                                               |
| D2  | Distributed training (gloo) | nccl is the primary backend for GPU — gloo is not needed |

## Test → Functionality Mapping

### test_mnist.py — 6 test cases

| Test                                  | F1  | F2  | F3  | F4  | F5  | F6  |
| ------------------------------------- | --- | --- | --- | --- | --- | --- |
| `test_mnist_distributed_cpu[gloo]`    | ✅  |     | ✅  |     |     |     |
| `test_mnist_distributed_gpu[gloo]`    | ✅  |     | ✅  |     |     |     |
| `test_mnist_distributed_gpu[nccl]`    | ✅  | ✅  |     |     |     |     |
| `test_hc_mnist_distributed_cpu[gloo]` | ✅  |     | ✅  |     |     | ✅  |
| `test_hc_mnist_distributed_gpu[gloo]` | ✅  |     | ✅  |     |     | ✅  |
| `test_hc_mnist_distributed_gpu[nccl]` | ✅  | ✅  |     |     |     | ✅  |

### test_mnist_inductor.py — 6 test cases

| Test                                  | F1  | F2  | F3  | F4  | F5  | F6  |
| ------------------------------------- | --- | --- | --- | --- | --- | --- |
| `test_mnist_distributed_cpu[gloo]`    | ✅  |     | ✅  | ✅  |     |     |
| `test_mnist_distributed_gpu[gloo]`    | ✅  |     | ✅  | ✅  |     |     |
| `test_mnist_distributed_gpu[nccl]`    | ✅  | ✅  |     | ✅  |     |     |
| `test_hc_mnist_distributed_cpu[gloo]` | ✅  |     | ✅  | ✅  |     | ✅  |
| `test_hc_mnist_distributed_gpu[gloo]` | ✅  |     | ✅  | ✅  |     | ✅  |
| `test_hc_mnist_distributed_gpu[nccl]` | ✅  | ✅  |     | ✅  |     | ✅  |

### test_distributed_operations.py — 11 test cases

| Test                                   | F1  | F2  | F3  | F4  | F5  | F6  | Deprecated |
| -------------------------------------- | --- | --- | --- | --- | --- | --- | ---------- |
| `test_dist_operations_cpu[gloo]`       | ✅  |     | ✅  |     | ✅  |     |            |
| `test_dist_operations_gpu[gloo]`       | ✅  |     | ✅  |     | ✅  |     |            |
| `test_dist_operations_gpu[nccl]`       | ✅  | ✅  |     |     | ✅  |     |            |
| `test_dist_operations_multi_gpu[gloo]` | ✅  |     | ✅  |     | ✅  |     |            |
| `test_dist_operations_multi_gpu[nccl]` | ✅  | ✅  |     |     | ✅  |     |            |
| `test_dist_operations_fastai_gpu`      |     |     |     |     |     |     | D2         |
| `test_smmodelparallel_*` (6 tests)     |     |     |     |     |     |     | D1         |
| `test_sanity_efa`                      | ✅  |     |     |     |     |     | F7         |

### test_torch_distributed.py — 1 test

| Test                                    | F1  | F2  | F3  | F4  | F5  | F6  |
| --------------------------------------- | --- | --- | --- | --- | --- | --- |
| `test_torch_distributed_throughput_gpu` | ✅  | ✅  |     |     |     |     |

### test_torch_distributed_inductor.py — 1 test

| Test                                    | F1  | F2  | F3  | F4  | F5  | F6  |
| --------------------------------------- | --- | --- | --- | --- | --- | --- |
| `test_torch_distributed_throughput_gpu` | ✅  | ✅  |     | ✅  |     |     |

### test_gdrcopy.py — 1 test

| Test                  | F1  | F7  | F8  |
| --------------------- | --- | --- | --- |
| `test_sanity_gdrcopy` | ✅  |     | ✅  |

## Summary

- **36 total tests** covering **8 functionalities** at 3 priority levels
- **6 tests** are for deprecated features (D1: SMP) → remove
- **30 remaining tests** have heavy overlap — most functionalities are tested 3-6× across CPU/GPU, gloo/nccl, regular/HC, with/without inductor

## Minimum tests needed per functionality

F5 (inductor) is covered by EC2 single-GPU tests. F6/F7/F8 are low value for SM-specific testing.

| Priority | Functionality                    | Min tests | Covered by                         |
| -------- | -------------------------------- | --------- | ---------------------------------- |
| P0       | F1: Distributed nccl             | 1         | `test_mnist_distributed_gpu[nccl]` |
| P0       | F2: torch.distributed primitives | 1         | `test_dist_operations_gpu[nccl]`   |
| P1       | F3: EFA connectivity             | 1         | `test_sanity_efa`                  |
| P1       | F4: GDRCopy                      | 1         | `test_sanity_gdrcopy`              |
| P1       | F5: fastai training              | 1         | `test_dist_operations_fastai_gpu`  |
| P2       | F6-F7                            | 0         | Covered by EC2 tests or low value  |
|          | **Minimal: 2 (P0), 5 (P0+P1)**   |           |                                    |

## Proposed Plan

Run 2 tests:

| Test                               | F1  | F2  | F3  | F4  | F5  | F6  | F7  | F8  |
| ---------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
| `test_mnist_distributed_gpu[nccl]` | ✅  |     |     |     |     |     |     |     |
| `test_dist_operations_gpu[nccl]`   | ✅  | ✅  |     |     |     |     |     |     |

Covers: F1 (distributed nccl training) + F2 (torch.distributed primitives) = 2/8 functionalities directly tested in SM.

Remaining functionalities:

- F3/F4 (EFA, GDRCopy): add later when p4d quota is available — nice to have
- F5 (inductor): covered by EC2 single-GPU tests
- F6 (gloo), F7 (HC), F8 (fastai): low value for SM-specific testing

**36 tests → 2 tests. All P0 functionality covered.**

## Decision

1. Go with P0 (2 tests) or P0+P1 (4 tests)?
1. Refactor now or after merge?
1. Move to `test/pytorch/sagemaker/` (V2 pattern) or keep V1 path?
