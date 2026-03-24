# SGLang Test Infrastructure — Complete Inventory

## 1. Workflow YAML Files (8 files)

### PR Workflows (4)

| File                                                 | Triggers                                                      | Dockerfile                                                    | Tests Run                                                                      |
| ---------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `.github/workflows/pr-sglang-ec2.yml`                | PR paths `**sglang**` (excl docs, amzn2023)                   | `docker/sglang/Dockerfile` target `sglang-ec2`                | sanity, security, telemetry, upstream (benchmark + srt-backend)                |
| `.github/workflows/pr-sglang-sagemaker.yml`          | PR paths `**sglang**` (excl docs, amzn2023)                   | `docker/sglang/Dockerfile` target `sglang-sagemaker`          | sanity, security, telemetry, upstream (benchmark + srt-backend), endpoint-test |
| `.github/workflows/pr-sglang-ec2-amzn2023.yml`       | PR paths `docker/sglang/Dockerfile.amzn2023`, scripts, config | `docker/sglang/Dockerfile.amzn2023` target `sglang-ec2`       | sanity, security, telemetry, local-benchmark (inline, no upstream reusable)    |
| `.github/workflows/pr-sglang-sagemaker-amzn2023.yml` | PR paths `docker/sglang/Dockerfile.amzn2023`, scripts, config | `docker/sglang/Dockerfile.amzn2023` target `sglang-sagemaker` | sanity, security, telemetry, local-benchmark (inline), endpoint-test           |

### Reusable Workflows (2)

| File                                                    | Purpose                            | Key Details                                                                                                                                                                                                 |
| ------------------------------------------------------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `.github/workflows/reusable-sglang-upstream-tests.yml`  | Runs benchmark + srt-backend tests | 2 jobs: `local-benchmark-test` (g6xl runner, bench_serving with ShareGPT), `srt-backend-test` (g6exl runner, checks out sglang source at version tag, runs `run_suite.py --hw cuda --suite stage-a-test-1`) |
| `.github/workflows/reusable-sglang-sagemaker-tests.yml` | Runs SageMaker endpoint test       | 1 job: installs test deps, runs `pytest -vs -rA --image-uri <uri> sglang/sagemaker`                                                                                                                         |

### Auto-Release Workflows (2)

| File                                                  | Schedule                    |
| ----------------------------------------------------- | --------------------------- |
| `.github/workflows/auto-release-sglang-ec2.yml`       | Tue/Thu 10AM PDT, or manual |
| `.github/workflows/auto-release-sglang-sagemaker.yml` | Tue/Thu 10AM PDT, or manual |

## 2. Build Configuration Files (4 files)

| File                                           | OS          | Platform  | Version | Prod Image                          | Release              |
| ---------------------------------------------- | ----------- | --------- | ------- | ----------------------------------- | -------------------- |
| `.github/config/sglang-ec2.yml`                | ubuntu24.04 | ec2       | 0.5.9   | `sglang:0.5-gpu-py312-ec2`          | **yes** (production) |
| `.github/config/sglang-sagemaker.yml`          | ubuntu24.04 | sagemaker | 0.5.9   | `sglang:0.5-gpu-py312`              | **yes** (production) |
| `.github/config/sglang-ec2-amzn2023.yml`       | amzn2023    | ec2       | 0.5.9   | `sglang:0.5-gpu-py312-amzn2023-ec2` | **no** (gamma)       |
| `.github/config/sglang-sagemaker-amzn2023.yml` | amzn2023    | sagemaker | 0.5.9   | `sglang:0.5-gpu-py312-amzn2023`     | **no** (gamma)       |

All configs: framework=sglang, python=py312, cuda=cu129, arch=x86, device=gpu.

## 3. Test Files

### SGLang-Specific Tests (2 files)

#### `test/sglang/sagemaker/test_sm_endpoint.py`

- **Type**: Integration test (SageMaker endpoint deployment)
- **Fixtures**: `model_id` (parametrized: `Qwen/Qwen3-0.6B`), `instance_type` (parametrized: `ml.g5.12xlarge`), `model_package`, `model_endpoint`
- **Test function**: `test_sglang_sagemaker_endpoint` — deploys model via SageMaker, sends OpenAI-compatible chat completion request, asserts non-empty response
- **Dependencies**: `sagemaker>=2,<3`, `test_utils` (clean_string, random_suffix_name, wait_for_status, SAGEMAKER_ROLE, INFERENCE_AMI_VERSION, get_hf_token)
- **Env vars used**: `SM_SGLANG_MODEL_PATH`, `HF_TOKEN`

#### `test/sglang/sagemaker/requirements.txt`

- Content: `sagemaker>=2,<3`

### Shared Sanity Tests (1 file, shared with vLLM)

#### `test/sanity/scripts/test_sanity_vllm_sglang.py`

- **Type**: Unit tests (GPU-free, run inside container)
- **Framework**: `unittest` (not pytest)
- **Auto-detects** vLLM vs SGLang from entrypoint content (`SM_VLLM_` vs `SM_SGLANG_` prefix)
- **4 test classes**:
  1. `TestCudaJitDependencies` — verifies nvcc, ptxas, cuobjdump, fatbinary, nvlink; tests deep_gemm, flashinfer, triton imports
  1. `TestEntrypointArgHandling` — dry-runs sagemaker_entrypoint.sh, tests string/numeric/boolean env var translation, default port 8080, model auto-detect
  1. `TestPackageVersionConsistency` — vllm/python/CUDA version vs IMAGE_TAG, torch↔CUDA agreement, no duplicate pip packages
  1. `TestEntrypointContract` — entrypoint exists, is executable, invokes correct server module (`sglang.launch_server`), defaults to port 8080

### Security Allowlist (1 file)

#### `test/security/data/ecr_scan_allowlist/sglang/framework_allowlist.json`

- 3 CVE allowlist entries for sglang 0.5.9 (jsonwebtoken, grpc, tar — all build-time/bundled deps)

## 4. Dockerfiles & Entrypoint Scripts

### Dockerfiles (2 files)

| File                                | Base Image                                    | Build Strategy                                                                                                                                          |
| ----------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `docker/sglang/Dockerfile`          | `lmsysorg/sglang:v0.5.9` (upstream pre-built) | Overlay DLC scripts on upstream image, multi-target (sagemaker/ec2)                                                                                     |
| `docker/sglang/Dockerfile.amzn2023` | `nvidia/cuda:12.9.1-devel-amzn2023`           | Full from-source build (sglang, sgl-kernel, flashinfer, DeepEP, mooncake, sgl-model-gateway), multi-stage builder→runtime, multi-target (sagemaker/ec2) |

### Entrypoint Scripts (2 files)

| File                                     | Purpose           | Key Behavior                                                                                                                                                                                |
| ---------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scripts/sglang/sagemaker_entrypoint.sh` | SageMaker serving | Translates `SM_SGLANG_*` env vars → CLI args, boolean handling (true=flag, false=omit), defaults: port 8080, host 0.0.0.0, model-path /opt/ml/model, runs `python3 -m sglang.launch_server` |
| `scripts/sglang/dockerd_entrypoint.sh`   | EC2 serving       | Simple passthrough: `python3 -m sglang.launch_server "$@"`                                                                                                                                  |

## 5. Shared Test Utilities Used by SGLang

### `test/conftest.py` (root)

- Fixtures: `image_uri` (from `--image-uri` CLI option), `region`, `aws_session` (AWSSessionManager)

### `test/test_utils/` (6 files)

| File                    | Exports Used by SGLang                                        |
| ----------------------- | ------------------------------------------------------------- |
| `__init__.py`           | `random_suffix_name()`, `clean_string()`, `wait_for_status()` |
| `aws.py`                | `AWSSessionManager` class                                     |
| `constants.py`          | `INFERENCE_AMI_VERSION`, `SAGEMAKER_ROLE`, `DEFAULT_REGION`   |
| `huggingface_helper.py` | `get_hf_token()`                                              |
| `docker_helper.py`      | Not directly used by SGLang tests                             |
| `logger.py`             | Not directly used by SGLang tests                             |

## 6. Key Observations

### What Exists

- **SageMaker endpoint test**: Full integration test deploying Qwen3-0.6B via SageMaker
- **Sanity tests**: Comprehensive unit tests shared with vLLM (entrypoint, CUDA, packages, versions)
- **Upstream tests**: Runs SGLang's own `run_suite.py --suite stage-a-test-1` from sglang source
- **Benchmark tests**: `sglang.bench_serving` with ShareGPT dataset (1000 prompts)
- **Security scanning**: ECR scan with framework-specific allowlist

### What Does NOT Exist

- **No `test/sglang/conftest.py`** — SGLang tests rely on root `test/conftest.py`
- **No EC2 functional tests** — No `test/sglang/ec2/` directory; EC2 testing is only via upstream reusable workflow (benchmark + srt-backend)
- **No SGLang-specific unit tests** — All unit tests are shared with vLLM in `test/sanity/scripts/test_sanity_vllm_sglang.py`
- **No `test/sglang/ec2/` directory at all** — EC2 tests are entirely inline in workflows or delegated to upstream
- **AMZN2023 workflows lack upstream srt-backend tests** — They only run local-benchmark (inline), not the reusable upstream workflow

### Comparison with vLLM

vLLM has an identical structure: `test/vllm/sagemaker/test_sm_endpoint.py` + `requirements.txt`. Both frameworks share the sanity test file. Neither has dedicated EC2 test files.
