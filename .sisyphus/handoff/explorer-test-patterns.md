# Test Patterns in deep-learning-containers

## 1. Directory Structure

```
test/
├── conftest.py                          # Root: --image-uri, --region, aws_session fixture
├── requirements.txt                     # boto3, botocore, fabric, pytest
├── test_utils/
│   ├── __init__.py                      # random_suffix_name(), clean_string(), wait_for_status()
│   ├── aws.py                           # AWSSessionManager (EC2, SageMaker, SSH, key pairs, SGs)
│   ├── constants.py                     # DEFAULT_REGION, EC2_INSTANCE_ROLE_NAME, SAGEMAKER_ROLE, INFERENCE_AMI_VERSION
│   ├── docker_helper.py                 # parse_image_uri(), get_docker_labels()
│   ├── huggingface_helper.py            # get_hf_token() from Secrets Manager
│   └── logger.py
├── vllm/
│   └── sagemaker/
│       ├── requirements.txt             # sagemaker>=2,<3
│       └── test_sm_endpoint.py          # SageMaker endpoint integration test
├── sglang/
│   └── sagemaker/
│       ├── requirements.txt             # sagemaker>=2,<3
│       └── test_sm_endpoint.py          # SageMaker endpoint integration test
├── pytorch/
│   ├── conftest.py                      # Empty (docstring only: "tests run inside container")
│   ├── pytest.ini                       # (empty)
│   ├── unit/                            # CPU-only tests run inside container
│   │   ├── test_imports.py
│   │   ├── test_environment.py
│   │   ├── test_filesystem.py
│   │   ├── test_ssh_config.py
│   │   ├── test_versions.py
│   │   └── test_eks.py
│   ├── single_gpu/                      # Single GPU functional tests
│   │   ├── test_cuda.py
│   │   ├── test_flash_attn.py
│   │   ├── test_runtime.py
│   │   ├── test_training_smoke.py
│   │   └── test_transformer_engine.py
│   ├── multi_gpu/                       # Multi-GPU functional tests
│   │   ├── test_ddp.py
│   │   ├── test_deepspeed.py
│   │   └── test_fsdp.py
│   └── multi_node/
│       ├── test_multinode_ddp.py
│       └── test_nccl_efa.py
├── sanity/scripts/
│   └── test_sanity_vllm_sglang.py       # GPU-free sanity (unittest-based, runs inside container)
├── telemetry/
│   ├── conftest.py                      # EC2 instance lifecycle fixtures, SSH connection
│   └── test_telemetry.py                # EC2 telemetry integration tests
├── security/                            # ECR scan allowlists
├── docs/                                # Documentation generation tests
└── dlc_tests/ec2/                       # Legacy EC2 tests
```

## 2. vLLM Test Files

### test/vllm/sagemaker/test_sm_endpoint.py

- **Type**: SageMaker integration test (runs from CI host, not inside container)
- **Fixtures**: `model_id`, `instance_type` (indirect parametrize), `model_package`, `model_endpoint`
- **Test**: `test_vllm_sagemaker_endpoint` — deploys model to SageMaker endpoint, sends chat completion request, asserts non-empty response
- **Model**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` on `ml.g5.12xlarge`
- **Env var**: `SM_VLLM_MODEL` for model ID
- **Payload**: OpenAI-compatible chat format with messages/max_tokens/temperature/top_p/top_k
- **Cleanup**: Deletes model, endpoint, and endpoint config in finally blocks

### test/sanity/scripts/test_sanity_vllm_sglang.py

- **Type**: GPU-free sanity tests (unittest.TestCase, runs inside container)
- **4 test classes**:
  1. `TestCudaJitDependencies` — verifies nvcc, ptxas, cuobjdump, fatbinary, nvlink exist; deep_gemm/flashinfer/triton import
  1. `TestEntrypointArgHandling` — dry-runs sagemaker_entrypoint.sh, verifies env→CLI arg translation (booleans, model autodetect, HF_MODEL_ID fallback)
  1. `TestPackageVersionConsistency` — checks vllm version, python version, CUDA version match image tag; no duplicate pip packages
  1. `TestEntrypointContract` — entrypoint exists, is executable, invokes correct server, defaults to port 8080
- **Framework-aware**: Auto-detects vLLM vs SGLang from entrypoint content (SM_VLLM\_ vs SM_SGLANG\_ prefix)
- **Can run as**: `python3 test_sanity_vllm_sglang.py` or `pytest test_sanity_vllm_sglang.py -v`

### Upstream vLLM tests (via scripts/)

- `scripts/vllm/vllm_test_setup.sh` — installs vllm test deps, pytest, hf_transfer
- `scripts/vllm/vllm_regression_test.sh` — runs `pytest -v -s test_regression.py` from vllm source
- `scripts/vllm/vllm_cuda_test.sh` — runs `pytest -v -s cuda/test_cuda_context.py`
- `scripts/vllm/vllm_ec2_examples_test.sh` — runs vllm example scripts (offline inference, spec decode, etc.)
- `scripts/vllm/vllm_model_smoke_test.sh` — starts vllm serve, health check, sends /v1/completions request

## 3. SGLang Test Files

### test/sglang/sagemaker/test_sm_endpoint.py

- **Type**: SageMaker integration test (nearly identical to vllm version)
- **Differences from vllm**:
  - Env var: `SM_SGLANG_MODEL_PATH` (not `SM_VLLM_MODEL`)
  - Payload includes `"model": model_id` field (OpenAI-compatible)
  - Model: `Qwen/Qwen3-0.6B` on `ml.g5.12xlarge`
  - Resource names prefixed `sglang-` instead of `vllm-`
  - Test function takes `model_id` as parameter (for payload construction)

### Upstream SGLang tests (via workflows)

- `reusable-sglang-upstream-tests.yml` runs:
  1. `local-benchmark-test` — `sglang.bench_serving` with ShareGPT dataset
  1. `srt-backend-test` — checks out sglang source, runs `python3 run_suite.py --hw cuda --suite stage-a-test-1`

## 4. Shared Test Utilities

### test/conftest.py (root)

```python
def pytest_addoption(parser):
    parser.addoption("--image-uri", ...)
    parser.addoption("--region", default="us-west-2", ...)

@pytest.fixture(scope="session")
def image_uri(request): ...

@pytest.fixture(scope="session")
def region(request): ...

@pytest.fixture(scope="session")
def aws_session(region):
    return AWSSessionManager(region)
```

### test/test_utils/__init__.py

- `random_suffix_name(resource_name, max_length)` — appends random alphanumeric suffix
- `clean_string(text, symbols_to_remove)` — replaces symbols with dashes
- `wait_for_status(expected, periods, length, get_status_fn, *args)` — polling loop

### test/test_utils/aws.py — AWSSessionManager

- Wraps boto3 session with clients for: ec2, sagemaker, ecr, s3, sts, secretsmanager, etc.
- EC2 lifecycle: `launch_instance()`, `terminate_instance()`, `wait_for_instance_ready()`, `get_public_ip()`
- SSH: `create_key_pair()`, `delete_key_pair()`, `get_ssh_connection()` (returns Fabric LoggedConnection)
- Security groups: `create_ssh_security_group()`, `delete_security_group()`
- AMI: `get_latest_ami()` via SSM parameter

### test/test_utils/huggingface_helper.py

- `get_hf_token(aws_session)` — retrieves HF token from Secrets Manager at `test/hf_token`

### test/test_utils/docker_helper.py

- `parse_image_uri(uri)` → `ImageURI(full_uri, account_id, region, repository, image_tag)`
- `get_docker_labels(uri)` → dict from `docker inspect`

## 5. Test Invocation Patterns

### From CI workflows

```bash
# SageMaker tests (run from CI host)
cd test/
python3 -m pytest -vs -rA --image-uri <URI> vllm/sagemaker
python3 -m pytest -vs -rA --image-uri <URI> sglang/sagemaker

# Upstream tests (run inside container via docker exec)
docker exec $CID scripts/vllm/vllm_regression_test.sh
docker exec $CID scripts/vllm/vllm_cuda_test.sh

# Sanity tests (run inside container)
docker run --rm --entrypoint pytest <image> /tests/test_sanity_vllm_sglang.py -v
```

### Pytest conventions

- `pytest -vs -rA` — verbose, no capture, show all test results
- `--image-uri` custom option via root conftest.py
- `@pytest.mark.parametrize("param", [...], indirect=True)` for fixture parametrization
- Function-scoped fixtures for model/endpoint lifecycle with try/finally cleanup
- Session-scoped fixtures for AWS session and image URI

### Two test paradigms

1. **Inside-container tests** (pytorch/unit, pytorch/single_gpu, sanity): Plain Python/pytest, assume they're already in the right environment. No Docker fixtures.
1. **Outside-container tests** (vllm/sagemaker, sglang/sagemaker, telemetry): Run from CI host, manage AWS resources (endpoints, EC2 instances) via fixtures.

## 6. Test Dependencies

### test/requirements.txt (root)

```
boto3
botocore
fabric
pytest
```

### test/vllm/sagemaker/requirements.txt

```
sagemaker>=2,<3
```

### test/sglang/sagemaker/requirements.txt

```
sagemaker>=2,<3
```

## 7. Key Patterns for New Tests

### Naming

- Test files: `test_<what>.py`
- Test functions: `test_<framework>_<what>` (e.g., `test_vllm_sagemaker_endpoint`)
- Fixtures: descriptive nouns (`model_package`, `model_endpoint`, `ec2_instance`)

### Fixture lifecycle

- Use `yield` in fixtures for setup/teardown
- Always clean up AWS resources in `finally` blocks
- Scope: `session` for shared resources (aws_session), `function` for per-test resources

### Inside-container tests (pytorch pattern)

- Pure pytest, no Docker orchestration
- Use `@pytest.mark.parametrize` for data-driven tests
- Group related tests in classes (e.g., `TestContainerEnv`, `TestBinaries`)
- Test the contract: imports work, CUDA available, training converges

### Outside-container tests (sagemaker pattern)

- Indirect parametrize for model_id and instance_type
- Fixture chain: `aws_session` → `model_package` → `model_endpoint` → test function
- Assert on non-empty response (not specific content)

### vLLM vs SGLang differences

| Aspect            | vLLM                                 | SGLang                       |
| ----------------- | ------------------------------------ | ---------------------------- |
| Model env var     | `SM_VLLM_MODEL`                      | `SM_SGLANG_MODEL_PATH`       |
| Payload           | No `model` field                     | Includes `"model": model_id` |
| Entrypoint prefix | `SM_VLLM_`                           | `SM_SGLANG_`                 |
| Server module     | `vllm.entrypoints.openai.api_server` | `sglang.launch_server`       |

### What's missing (no tests yet for vllm/sglang)

- **No unit tests** — no `test/vllm/unit/` or `test/sglang/unit/` directories
- **No EC2 functional tests** — no `test/vllm/ec2/` or `test/sglang/ec2/` directories
- **No in-container pytest tests** — all vllm/sglang GPU tests are shell scripts, not pytest
- The sanity test (`test_sanity_vllm_sglang.py`) uses `unittest.TestCase`, not pytest style
