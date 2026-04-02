# XGBoost Container Testing

This directory contains all tests for the SageMaker XGBoost deep learning container.
Tests are organized into three tiers that validate the container at different levels,
from fast local Docker tests to full SageMaker training jobs.

## Access Requirements

| Tier | Requires AWS? | Requires internal S3? | Who can run |
|------|--------------|----------------------|-------------|
| Container tests | No (Docker only) | Yes — test resources in S3 | CI only (internal) |
| Integration tests | Yes — SageMaker jobs | Yes — `amazonai-algorithms-integ-tests` bucket | CI only (internal) |
| Benchmark tests | Yes — SageMaker jobs | Yes — `amazonai-algorithms-benchmarking` bucket | CI only (internal) |

> **Note for external contributors:** All three test tiers depend on internal AWS
> resources (S3 buckets, IAM roles, ECR registries) that are not publicly accessible.
> Tests are run automatically by CI when a PR is submitted. You do not need to run
> them locally — the CI bot will report results on your PR.
>
> If you want to validate your changes locally before submitting a PR, you can build
> the Docker image and run it manually:
>
> ```bash
> # Build the container
> docker build -t xgboost-test -f docker/xgboost/Dockerfile .
>
> # Quick smoke test — run a training job locally
> docker run --rm \
>   -v /tmp/test_data:/opt/ml/input/data \
>   -v /tmp/test_model:/opt/ml/model \
>   -v /tmp/test_output:/opt/ml/output \
>   xgboost-test train
> ```

## Test Tiers

```
test/xgboost/
├── container/       # Tier 1 — Local Docker container tests (minutes)
├── integration/     # Tier 2 — SageMaker SDK integration tests (30-60 min)
├── benchmarks/      # Tier 3 — SageMaker performance benchmarks (hours)
├── conftest.py      # Shared fixtures (role, image_uri)
└── requirements.txt # sagemaker SDK dependency
```

### Tier 1: Container Tests (`container/`)

Run the XGBoost container locally via docker-py. The container is pulled, mounted
with `/opt/ml/` directory structures, and exercised directly. These are the fastest
tests and run on every push.

Test resources (training data, inference payloads) are downloaded from
`s3://dlc-cicd-models/xgboost/container_test_resources/` at the start of the session.

| File | What it tests |
|------|---------------|
| `test_training.py` | Algorithm-mode training: libsvm/csv, single/multi-file, weights, HPO metrics, objectives, verbosity, checkpoint/reload, distributed, invalid hyperparameters |
| `test_scoring.py` | Inference: csv/libsvm/protobuf payloads, execution parameters, 20 MB payload, content type validation |
| `test_batch_transform.py` | Batch transform with `SAGEMAKER_BATCH=True` |
| `container_helper.py` | `run_training()` and `ServingContainer` context manager using docker-py |
| `conftest.py` | `--image` flag, S3 resource download, docker client fixture |
| `generate_models.py` | Generates XGBoost 3.0.5-compatible inference models and uploads to S3 |

**How to run (CI / internal):**

```bash
cd test/
pip install docker pytest boto3 requests
python3 -m pytest -v --tb=short --log-cli-level=INFO \
  --image <IMAGE_URI> \
  xgboost/container/test_training.py \
  xgboost/container/test_scoring.py \
  xgboost/container/test_batch_transform.py
```

> Scoring and batch transform tests require inference models in S3.
> Run `generate_models.py` first, or let the CI workflow handle it.

### Tier 2: Integration Tests (`integration/`)

Launch real SageMaker training jobs, endpoints, and batch transform jobs using the
SageMaker Python SDK. These validate the container works end-to-end on SageMaker
infrastructure. Migrated from `SMFrameworksXGBoost3_0-5Tests/src/integration_tests/`.

| File | What it tests |
|------|---------------|
| `test_training_libsvm.py` | Single/distributed/checkpoint/GPU training with libsvm data |
| `test_training_csv.py` | Single/distributed/pipe-mode training with CSV data |
| `test_training_pb.py` | Single/distributed/pipe-mode/sparse training with protobuf data |
| `test_training_pq.py` | Single/distributed/pipe-mode training with parquet data |
| `test_e2e.py` | Train a model → deploy endpoint → invoke (CPU + GPU) |
| `test_inference.py` | Deploy pre-trained model → invoke with libsvm/csv/protobuf |
| `test_transform.py` | Batch transform with libsvm input |
| `test_hpo.py` | Hyperparameter tuning (rmse minimization) |
| `test_script_mode_e2e.py` | Script-mode train → deploy → invoke |
| `test_network_isolation.py` | Algo-mode + script-mode with network isolation enabled |

**Data location:** `s3://amazonai-algorithms-integ-tests/input/xgboost/` (internal)

```
input/xgboost/
├── train/                          # libsvm abalone training data
├── test/                           # libsvm abalone test data
├── csv/{train,test}/               # CSV format
├── parquet/{train,test}/           # Parquet format
├── recordio-protobuf/{train,test}/ # Protobuf format (+ sparse/)
├── iris/{train,test}/              # Iris CSV (multiclass)
├── model_1.0/models/model.tar.gz  # Pre-trained model for inference tests
├── script_mode/
│   ├── code/abalone.1.2-1.tar.gz  # Script-mode training script
│   └── data/{train,validation}/    # Script-mode data
└── testdata/abalone_test.libsvm   # Batch transform input
```

**How to run (CI / internal):**

```bash
cd test/
pip install -r requirements.txt -r xgboost/requirements.txt
python3 -m pytest -v --tb=short \
  --image-uri <IMAGE_URI> \
  xgboost/integration/
```

### Tier 3: Benchmark Tests (`benchmarks/`)

SageMaker training jobs that measure performance across objectives, tree methods,
data sizes, instance types, content types, max depth, and num_round.

| File | What it tests |
|------|---------------|
| `test_training_objective.py` | reg:squarederror, binary:logistic, multi:softmax |
| `test_training_tree_method.py` | exact, approx, hist, gpu_hist |
| `test_training_max_depth.py` | Depth 2/5/8/12 |
| `test_training_num_round.py` | 10/50/100/200 rounds |
| `test_training_data_size.py` | 10k/100k/500k rows |
| `test_training_instance_type.py` | m5.large/xlarge/2xlarge, g4dn.xlarge |
| `test_training_content_type.py` | libsvm, csv, protobuf |

**Data location:** `s3://amazonai-algorithms-benchmarking/xgboost/` (internal)

**How to run (CI / internal):**

```bash
cd test/
pip install -r requirements.txt -r xgboost/requirements.txt
python3 -m pytest -v --tb=short \
  --image-uri <IMAGE_URI> \
  xgboost/benchmarks/test_training_objective.py
```

## CI Workflows

Three GitHub Actions workflows orchestrate these tests automatically:

| Workflow | Trigger | What runs |
|----------|---------|-----------|
| `pr-sagemaker-xgboost.yml` | PR to `main` touching `docker/xgboost/**` | Build → unit tests → security → integration (upstream local-mode) |
| `release-sagemaker-xgboost.yml` | `workflow_dispatch` / push | Build → unit tests → security → `sagemaker-xgboost-integ-tests.yml` |
| `sagemaker-xgboost-integ-tests.yml` | Called by release workflow | Container tests (training, scoring, batch transform) with model generation |

### How a release build flows

```
release-sagemaker-xgboost.yml
  ├── load-config
  ├── build-image
  ├── unit-test          (upstream sagemaker-xgboost-container unit tests + flake8)
  ├── security-test      (reusable-security-tests.yml)
  └── xgboost-tests      (sagemaker-xgboost-integ-tests.yml)
        ├── generate-models          (XGBoost 3.0.5 model generation)
        ├── container-test-training  (parallel, no model dependency)
        ├── container-test-scoring   (after generate-models)
        └── container-test-batch-transform (after generate-models)
```

## For External Contributors

You don't need to run any tests locally. When you open a PR:

1. The CI bot builds the container image from your changes
2. All test tiers run automatically against the built image
3. Results are posted on the PR

If CI fails, check the workflow logs linked in the PR checks. Common issues:
- **Training test failures** — usually a hyperparameter or entrypoint change
- **Scoring test failures** — usually a content type or model format issue
- **Security test failures** — usually a new CVE in a dependency

## Common Flags

| Flag | Used by | Description |
|------|---------|-------------|
| `--image` | container tests | Docker image URI to test |
| `--image-uri` | integration + benchmark tests | SageMaker image URI to test |
| `--region` | all | AWS region (default: `us-west-2`) |
| `--benchmark-bucket` | benchmarks | Override S3 bucket for benchmark data |
