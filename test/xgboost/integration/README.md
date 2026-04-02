# XGBoost Integration Tests

## Overview

These tests are migrated from `SMFrameworksXGBoost3_0-5Tests/src/integration_tests/`.
They replace the `ai_algorithms_qa` (Hydra) framework with direct SageMaker Python SDK calls.

## Test Files

| File | Description | Source |
|------|-------------|--------|
| `test_training_libsvm.py` | Training with libsvm data | `test_training_libsvm.py` |
| `test_training_csv.py` | Training with CSV data | `test_training_csv.py` |
| `test_training_pb.py` | Training with protobuf data | `test_training_pb.py` |
| `test_training_pq.py` | Training with parquet data | `test_training_pq.py` |
| `test_e2e.py` | End-to-end train + deploy | `test_e2e.py` |
| `test_inference.py` | Inference with pre-trained models | `test_inference.py` |
| `test_transform.py` | Batch transform | `test_transform.py` |
| `test_hpo.py` | Hyperparameter optimization | `test_hpo.py` |
| `test_script_mode_e2e.py` | Script mode train + inference | `test_script_mode_e2e.py` |
| `test_network_isolation.py` | Network isolation training | `test_network_isolation.py` |

## Data Setup

Integration test data must be available at:
```
s3://dlc-cicd-models/xgboost/integ_test_resources/
```

### Required S3 structure

```
xgboost/integ_test_resources/
├── train/                          # libsvm abalone training data
├── test/                           # libsvm abalone test data
├── csv/train/                      # CSV training data
├── csv/test/                       # CSV test data
├── parquet/train/                  # Parquet training data
├── parquet/test/                   # Parquet test data
├── recordio-protobuf/train/        # Protobuf training data
├── recordio-protobuf/test/         # Protobuf test data
├── recordio-protobuf/sparse/train/
├── recordio-protobuf/sparse/test/
├── iris/train/                     # Iris CSV training data
├── iris/test/                      # Iris CSV test data
├── model_1.0/models/model.tar.gz  # Pre-trained model for inference
├── script_mode/
│   ├── code/abalone.1.2-1.tar.gz  # Script mode training script
│   └── data/
│       ├── train/                  # Script mode training data
│       └── validation/             # Script mode validation data
└── testdata/
    └── abalone_test.libsvm        # Batch transform input
```

### Uploading script mode resources

```bash
cd test/xgboost/integration
python upload_resources.py
```

### Copying data from existing bucket

The original data lives in the `amazonai-algorithms-integ-tests` bucket under `input/xgboost/`.
Copy it to the DLC CI bucket:

```bash
aws s3 sync s3://amazonai-algorithms-integ-tests-us-west-2/input/xgboost/ \
            s3://dlc-cicd-models/xgboost/integ_test_resources/
```

## Running Tests

```bash
cd test/
python3 -m pytest -v --tb=short \
  --image-uri <IMAGE_URI> \
  xgboost/integration/
```

## Resources

- `resources/script_mode/code/abalone.py` — Script mode training script
- `upload_resources.py` — Helper to package and upload resources to S3
