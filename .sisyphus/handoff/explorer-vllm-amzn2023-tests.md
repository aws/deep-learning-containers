# vLLM AMZN2023 Model Deployment & Serving Tests — Complete Findings

## 1. Workflow YAML Files

### Primary PR Workflows (amzn2023-specific)

| File                                               | Purpose                                                                                                                                                                                                                                                                            |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `.github/workflows/pr-vllm-ec2-amzn2023.yml`       | PR workflow for vLLM EC2 AL2023 image. Triggers on changes to `docker/vllm/Dockerfile.amzn2023`, `scripts/vllm/amzn2023/**`, `.github/config/vllm-ec2-amzn2023.yml`. Jobs: `build-image`, `sanity-test`, `security-test`, `telemetry-test`, `upstream-tests`, `model-smoke-tests`. |
| `.github/workflows/pr-vllm-sagemaker-amzn2023.yml` | PR workflow for vLLM SageMaker AL2023 image. Same triggers plus `test/vllm/sagemaker/**`. Jobs: same as EC2 plus `endpoint-test` (SageMaker endpoint deployment test).                                                                                                             |

### Reusable Workflows (shared across vllm variants)

| File                                                  | Purpose                                                                                                                                                                                                                                                                                                                                                                      |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `.github/workflows/reusable-vllm-upstream-tests.yml`  | 3 jobs: `regression-test` (runs `vllm_regression_test.sh`), `cuda-test` (runs `vllm_cuda_test.sh`), `example-test` (runs platform-specific example script). Accepts `setup-script` and `example-test-script` inputs — amzn2023 passes `scripts/vllm/amzn2023/vllm_test_setup.sh` and `scripts/vllm/amzn2023/vllm_ec2_examples_test.sh` or `vllm_sagemaker_examples_test.sh`. |
| `.github/workflows/reusable-vllm-model-tests.yml`     | Model smoke tests. Reads model matrix from `.github/config/vllm-model-tests.yml`. Downloads models from S3, runs `scripts/vllm/vllm_model_smoke_test.sh` inside container. Supports both CodeBuild fleet and runner-scale-sets.                                                                                                                                              |
| `.github/workflows/reusable-vllm-sagemaker-tests.yml` | SageMaker endpoint test. Installs `test/requirements.txt` + `test/vllm/sagemaker/requirements.txt`, runs `pytest -vs -rA --image-uri <URI> vllm/sagemaker` from `test/` directory.                                                                                                                                                                                           |
| `.github/workflows/reusable-sanity-tests.yml`         | Shared sanity tests (not vllm-specific).                                                                                                                                                                                                                                                                                                                                     |
| `.github/workflows/reusable-security-tests.yml`       | Shared security tests.                                                                                                                                                                                                                                                                                                                                                       |
| `.github/workflows/reusable-telemetry-tests.yml`      | Shared telemetry tests.                                                                                                                                                                                                                                                                                                                                                      |
| `.github/workflows/vllm-heavy-test.yml`               | Manual dispatch EFA test. Uses `gpu-efa-runners`. Not amzn2023-specific but can be used with amzn2023 images.                                                                                                                                                                                                                                                                |

### Key Workflow Pattern

The amzn2023 workflows differentiate from non-amzn2023 by:

- Using `Dockerfile.amzn2023` (vs `Dockerfile`)
- Using `scripts/vllm/amzn2023/` scripts (vs `scripts/vllm/` root scripts)
- Config files: `vllm-ec2-amzn2023.yml` / `vllm-sagemaker-amzn2023.yml`
- Base image: `nvidia/cuda:12.9.1-devel-amzn2023`

## 2. Configuration Files

| File                                         | Purpose                 | Key Values                                                                                                                                                                          |
| -------------------------------------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `.github/config/vllm-ec2-amzn2023.yml`       | EC2 build config        | `framework: vllm`, `framework_version: 0.17.1`, `python_version: py312`, `cuda_version: cu129`, `os_version: amzn2023`, `customer_type: ec2`, `prod_image: vllm:0.17-gpu-py312-ec2` |
| `.github/config/vllm-sagemaker-amzn2023.yml` | SageMaker build config  | Same as EC2 except `customer_type: sagemaker`, `prod_image: vllm:0.17-gpu-py312-amzn2023-sagemaker`                                                                                 |
| `.github/config/vllm-model-tests.yml`        | Model smoke test matrix | `codebuild-fleet`: gpt-oss-20b (1xGPU), llama-3.3-70b (4xGPU). `runner-scale-sets`: qwen3-32b (4xGPU). Models from `s3://dlc-cicd-models/vllm_models/`.                             |

## 3. Test Files

### Python Test Files

| File                                      | Purpose                             | Key Signatures                                                                                                                                                                                                                               |
| ----------------------------------------- | ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test/vllm/sagemaker/test_sm_endpoint.py` | SageMaker endpoint integration test | Fixtures: `model_id`, `instance_type`, `model_package`, `model_endpoint`. Test: `test_vllm_sagemaker_endpoint` — deploys `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` on `ml.g5.12xlarge`, sends chat completion, asserts non-empty response. |
| `test/vllm/sagemaker/requirements.txt`    | Test deps                           | `sagemaker>=2,<3`                                                                                                                                                                                                                            |

### Shell Test Scripts (amzn2023-specific)

| File                                                    | Purpose                                                                                                                                                                                                                                                                                                 |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scripts/vllm/amzn2023/vllm_test_setup.sh`              | Test environment setup for amzn2023. Compiles vllm test deps with `uv pip compile` using `--torch-backend cu129`, installs dev deps, pytest, hf_transfer. Moves vllm source into `src/vllm` structure. Key difference from non-amzn2023: no `--system` flag when VIRTUAL_ENV is set (AL2023 uses venv). |
| `scripts/vllm/amzn2023/vllm_ec2_examples_test.sh`       | EC2 example tests. Runs ~14 vllm offline inference examples: generate, chat, prefix_caching, audio_language, vision_language, tensorizer, whisper, classify, embed, score, spec_decode (eagle + eagle3).                                                                                                |
| `scripts/vllm/amzn2023/vllm_sagemaker_examples_test.sh` | SageMaker example tests. Same offline inference examples as EC2, PLUS upstream pytest tests: `test_sagemaker_lora_adapters.py`, `test_sagemaker_stateful_sessions.py`, `test_sagemaker_middleware_integration.py`, `test_sagemaker_handler_overrides.py`, `test_lora_adapters.py`.                      |

### Shell Test Scripts (shared, used by amzn2023 workflows)

| File                                    | Purpose                                                                                                                                                     |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scripts/vllm/vllm_model_smoke_test.sh` | Model smoke test. Starts `vllm serve`, waits for `/health` (600s timeout), sends `/v1/completions` request with "Hello" prompt, asserts non-empty response. |
| `scripts/vllm/vllm_regression_test.sh`  | Runs `pytest -v -s test_regression.py` from vllm upstream source.                                                                                           |
| `scripts/vllm/vllm_cuda_test.sh`        | Runs `pytest -v -s cuda/test_cuda_context.py` from vllm upstream source.                                                                                    |

## 4. Shared Test Utilities & Fixtures

### conftest.py (test/conftest.py)

```python
# Fixtures: image_uri (session), region (session), aws_session (session)
# CLI options: --image-uri, --region (default us-west-2)
```

### test_utils/ package

| File                               | Key Exports                                                                                                                                                                                                                                                          |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_utils/__init__.py`           | `random_suffix_name()`, `clean_string()`, `wait_for_status()`                                                                                                                                                                                                        |
| `test_utils/constants.py`          | `DEFAULT_REGION="us-west-2"`, `SAGEMAKER_ROLE="SageMakerRole"`, `INFERENCE_AMI_VERSION="al2-ami-sagemaker-inference-gpu-3-1"`                                                                                                                                        |
| `test_utils/aws.py`                | `AWSSessionManager` class — wraps boto3 session with clients for ec2, sagemaker, ecr, s3, secretsmanager, etc. Methods: `get_latest_ami()`, `launch_instance()`, `terminate_instance()`, `get_ssh_connection()`, `create_key_pair()`, `create_ssh_security_group()`. |
| `test_utils/huggingface_helper.py` | `get_hf_token(aws_session)` — retrieves HF token from Secrets Manager at `test/hf_token`.                                                                                                                                                                            |
| `test_utils/docker_helper.py`      | (exists, not read — likely Docker container management helpers)                                                                                                                                                                                                      |
| `test_utils/logger.py`             | (exists, not read — logging config)                                                                                                                                                                                                                                  |

### Test requirements (test/requirements.txt)

```
boto3, botocore, fabric, pytest
```

## 5. Architecture Summary

```
PR triggers workflow → load-config (reads .github/config/vllm-{ec2|sagemaker}-amzn2023.yml)
  → build-image (Dockerfile.amzn2023, base: nvidia/cuda:12.9.1-devel-amzn2023)
  → parallel test jobs:
      ├── sanity-test (reusable-sanity-tests.yml) — shared across all DLCs
      ├── security-test (reusable-security-tests.yml) — shared
      ├── telemetry-test (reusable-telemetry-tests.yml) — shared
      ├── upstream-tests (reusable-vllm-upstream-tests.yml)
      │     ├── regression-test → vllm_regression_test.sh
      │     ├── cuda-test → vllm_cuda_test.sh
      │     └── example-test → scripts/vllm/amzn2023/vllm_{ec2|sagemaker}_examples_test.sh
      ├── model-smoke-tests (reusable-vllm-model-tests.yml) — EC2 only
      │     └── per-model: S3 download → vllm_model_smoke_test.sh
      └── endpoint-test (reusable-vllm-sagemaker-tests.yml) — SageMaker only
            └── pytest test/vllm/sagemaker/test_sm_endpoint.py
```

## 6. Key Patterns for SGLang Replication

1. **Config-driven**: Each image variant has a `.github/config/<framework>-<platform>-<os>.yml` with build params
1. **Dockerfile per OS**: `docker/vllm/Dockerfile.amzn2023` — SGLang would need `docker/sglang/Dockerfile.amzn2023`
1. **OS-specific scripts dir**: `scripts/vllm/amzn2023/` contains setup + example test scripts with OS-specific adjustments (venv detection, uv flags)
1. **Shared scripts**: `scripts/vllm/vllm_model_smoke_test.sh` pattern — start server, health check, send request, assert response
1. **Reusable workflows**: Parameterized with `setup-script` and `example-test-script` inputs for OS-variant flexibility
1. **SageMaker tests**: Python pytest in `test/<framework>/sagemaker/` with fixtures for model creation, endpoint deployment, inference validation
1. **Model test matrix**: Separate config file (`.github/config/vllm-model-tests.yml`) with S3 paths and serve args per model
