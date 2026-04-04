# XGBoost Container Testing

Tests for the SageMaker XGBoost deep learning container, organized into three tiers.
All tests run automatically in CI on every PR and release.

## Test Tiers

```
test/xgboost/
├── container/       # Tier 1 — Local Docker container tests (minutes)
├── e2e/             # Tier 2 — SageMaker E2E tests (30-60 min)
└── benchmarks/      # Tier 3 — SageMaker performance benchmarks (hours)
```

### Tier 1: Container Tests (`container/`)

Runs the XGBoost container locally via docker-py. The container is mounted with
`/opt/ml/` directory structures and exercised directly — no SageMaker jobs are created.

| File                      | What it tests                                                                                                                                                |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `test_training.py`        | Algorithm-mode training: libsvm/csv, single/multi-file, weights, HPO metrics, objectives, verbosity, checkpoint/reload, distributed, invalid hyperparameters |
| `test_scoring.py`         | Inference: csv/libsvm/protobuf payloads, execution parameters, 20 MB payload, content type validation                                                        |
| `test_batch_transform.py` | Batch transform with `SAGEMAKER_BATCH=True`                                                                                                                  |

Supporting files:

- `container_helper.py` — `run_training()` and `ServingContainer` context manager
- `generate_models.py` — generates XGBoost 3.0.5-compatible inference models

### Tier 2: E2E Tests (`e2e/`)

Launches real SageMaker training jobs, endpoints, and batch transform jobs using the
SageMaker Python SDK. Validates the container works end-to-end on SageMaker infrastructure.

| File                        | What it tests                                                     |
| --------------------------- | ----------------------------------------------------------------- |
| `test_training_libsvm.py`   | Single/distributed/checkpoint/GPU training with libsvm data       |
| `test_training_csv.py`      | Single/distributed/pipe-mode/Dask GPU training with CSV data      |
| `test_training_pb.py`       | Single/distributed/pipe-mode/sparse training with protobuf data   |
| `test_training_pq.py`       | Single/distributed/pipe-mode/Dask GPU training with parquet data  |
| `test_e2e.py`               | Train → deploy → invoke (CPU + GPU), Dask GPU training            |
| `test_e2e_selectable.py`    | Multiclass train → inference with CSV/JSON/JSONLINES accept types |
| `test_inference.py`         | Train a model → deploy → invoke with libsvm/csv                   |
| `test_inference_mme.py`     | Multi-model endpoint inference                                    |
| `test_transform.py`         | Train a model → batch transform with libsvm input                 |
| `test_hpo.py`               | Hyperparameter tuning: rmse, aucpr, GPU                           |
| `test_script_mode_e2e.py`   | Script-mode train → deploy → invoke                               |
| `test_network_isolation.py` | Algo-mode training with network isolation                         |

### Tier 3: Benchmark Tests (`benchmarks/`)

SageMaker training jobs that measure performance across different configurations.

| File                             | What it tests                                    |
| -------------------------------- | ------------------------------------------------ |
| `test_training_objective.py`     | reg:squarederror, binary:logistic, multi:softmax |
| `test_training_tree_method.py`   | exact, approx, hist, gpu_hist                    |
| `test_training_max_depth.py`     | Depth 2/5/8/12                                   |
| `test_training_num_round.py`     | 10/50/100/200 rounds                             |
| `test_training_data_size.py`     | 10k/100k/500k rows                               |
| `test_training_instance_type.py` | m5.large/xlarge/2xlarge, g4dn.xlarge             |
| `test_training_content_type.py`  | libsvm, csv, protobuf                            |

## CI Workflows

| Workflow                            | Trigger                                   | What runs                                                           |
| ----------------------------------- | ----------------------------------------- | ------------------------------------------------------------------- |
| `pr-sagemaker-xgboost.yml`          | PR to `main` touching `docker/xgboost/**` | Build → unit tests → security → upstream integration                |
| `release-sagemaker-xgboost.yml`     | `workflow_dispatch` / push                | Build → unit tests → security → `sagemaker-xgboost-integ-tests.yml` |
| `sagemaker-xgboost-integ-tests.yml` | Called by release workflow                | Container tests → E2E tests → benchmarks                            |

### Release build flow

```
release-sagemaker-xgboost.yml
  ├── load-config
  ├── build-image
  ├── unit-test              (upstream sagemaker-xgboost-container tests + flake8)
  ├── security-test          (reusable-security-tests.yml)
  └── xgboost-tests          (sagemaker-xgboost-integ-tests.yml)
        ├── generate-models              (XGBoost 3.0.5 model generation)
        ├── container-test-training      (parallel with generate-models)
        ├── container-test-scoring       (after generate-models)
        ├── container-test-batch-transform (after generate-models)
        ├── e2e-test                     (after container tests, 12 modules in parallel)
        └── benchmark-test               (after e2e tests, 7 modules in parallel)
```
