# TensorFlow Inference Image Build Runbook

This runbook documents the process for creating new TensorFlow Inference Docker images for AWS Deep Learning Containers. It captures lessons learned and common issues encountered during builds.

## Table of Contents
1. [Using This Runbook with AI Assistance](#using-this-runbook-with-ai-assistance)
2. [Prerequisites](#prerequisites)
3. [File Structure](#file-structure)
4. [Step-by-Step Build Process](#step-by-step-build-process)
5. [Common Issues and Solutions](#common-issues-and-solutions)
6. [Testing](#testing)
7. [Checklist](#checklist)

---

## Using This Runbook with AI Assistance

This runbook is designed to be used with AI coding assistants (like Cline). Simply reference this runbook in your prompt, and the AI will follow the documented process.

### Quick Start Prompt

For a new TensorFlow inference image, use this prompt format:

```
Create TensorFlow X.Y inference images following the runbook at `tensorflow/inference/RUNBOOK.md`
```

### Recommended Prompt Format

For more control, include these details:

```
Create TensorFlow X.Y inference SageMaker images (CPU and GPU with CUDA Z.Z) 
following `tensorflow/inference/RUNBOOK.md`. 
Base it on the TF 2.20 images.
```

### Example Prompts

**Basic:**
> "Create TensorFlow 2.21 inference images following the runbook at `tensorflow/inference/RUNBOOK.md`"

**With CUDA version:**
> "Create TensorFlow 2.21 inference images with CUDA 12.6 following `tensorflow/inference/RUNBOOK.md`"

**With Python version:**
> "Create TensorFlow 2.21 inference images with Python 3.13 following `tensorflow/inference/RUNBOOK.md`"

**Full specification:**
> "Create TensorFlow 2.21 inference SageMaker images (CPU and GPU with CUDA 12.6, Python 3.13) following `tensorflow/inference/RUNBOOK.md`. Base it on the TF 2.20 images we created."

### What the AI Will Do

When you provide a prompt referencing this runbook, the AI will:

1. **Read this runbook** to understand the process
2. **Check availability** of TF Serving, TF Serving API, and license files
3. **Create all necessary files**:
   - Buildspec YAML (with proper version quoting)
   - CPU Dockerfile
   - GPU Dockerfile
4. **Apply CVE fixes** proactively (wheel, setuptools, nvjpeg)
5. **Update test allowlists** if TF Serving version differs from TF version
6. **Format Python files** with black
7. **Run through the checklist** before completion

### Tips for Efficient Prompts

- Always include the target TensorFlow version (e.g., "2.21")
- Specify CUDA version for GPU images if different from default
- Mention if you want to base it on an existing version
- Include any special requirements upfront

---

## Prerequisites

Before starting a new TensorFlow inference image build:

1. **Check TensorFlow Serving availability**: Visit [TensorFlow Serving releases](https://github.com/tensorflow/serving/releases) to verify if TF Serving for your target version exists
   - If TF Serving X.Y.Z doesn't exist, use the latest available version (usually X.Y-1.Z)
   
2. **Check TensorFlow Serving API availability**: Verify on PyPI
   - CPU: `tensorflow-serving-api==X.Y.Z`
   - GPU: `tensorflow-serving-api-gpu==X.Y.Z`

3. **Check license file availability**: Verify `s3://aws-dlc-licenses/tensorflow-X.Y/license.txt` exists
   - If not, use the previous version's license file

4. **Identify CUDA/cuDNN versions**: For GPU images, check TensorFlow's tested configurations

---

## File Structure

For a new TensorFlow inference version (e.g., 2.20), create:

```
tensorflow/inference/
├── buildspec-2-20-sm.yml           # SageMaker buildspec
├── buildspec-2-20-ec2.yml          # EC2 buildspec (if needed)
├── docker/
│   └── 2.20/
│       └── py3/
│           ├── Dockerfile.cpu      # CPU Dockerfile
│           └── cu125/              # CUDA version directory
│               └── Dockerfile.gpu  # GPU Dockerfile
└── sagemaker/                      # SageMaker serving code (copied to image)
```

---

## Step-by-Step Build Process

### Step 1: Create Buildspec File

Create `tensorflow/inference/buildspec-X-Y-sm.yml`:

```yaml
account_id: &ACCOUNT_ID <+ACCOUNT_ID+>
region: &REGION <+REGION+>
framework: &FRAMEWORK tensorflow
version: &VERSION "X.Y.0"  # IMPORTANT: Quote version numbers like "2.20" to prevent YAML parsing issues
short_version: &SHORT_VERSION "X.Y"  # IMPORTANT: Must be quoted!

repository_info:
  inference_repository: &INFERENCE_REPOSITORY
    image_type: &INFERENCE_IMAGE_TYPE inference
    root: !join [ *FRAMEWORK, "/", *INFERENCE_IMAGE_TYPE ]
    repository_name: &REPOSITORY_NAME !join [pr, "-", *FRAMEWORK, "-", *INFERENCE_IMAGE_TYPE]
    repository: &REPOSITORY !join [ *ACCOUNT_ID, .dkr.ecr., *REGION, .amazonaws.com/, *REPOSITORY_NAME ]

context:
  inference_context: &INFERENCE_CONTEXT
    sagemaker:
      source: docker/X.Y/py3/sagemaker
      target: sagemaker
    bash_telemetry:  # Required for telemetry tests
      source: ../../miscellaneous_scripts/bash_telemetry.sh
      target: bash_telemetry.sh

images:
  BuildTFInferenceCPUPy3DockerImage:
    <<: *INFERENCE_REPOSITORY
    build: &TENSORFLOW_CPU_INFERENCE_PY3 false
    image_size_baseline: 4000
    device_type: &DEVICE_TYPE cpu
    python_version: &DOCKER_PYTHON_VERSION py312
    tag_python_version: &TAG_PYTHON_VERSION py312
    os_version: &OS_VERSION ubuntu22.04
    tag: !join [ *VERSION, "-", *DEVICE_TYPE, "-", *TAG_PYTHON_VERSION, "-", *OS_VERSION, "-sagemaker" ]
    docker_file: !join [ docker/, *SHORT_VERSION, /py3/Dockerfile.cpu ]
    target: sagemaker
    context:
      <<: *INFERENCE_CONTEXT

  # Add GPU image configuration similarly
```

**Key Points:**
- Always quote version numbers like `"2.20"` to prevent YAML float parsing
- Include `bash_telemetry.sh` in context for telemetry tests
- Set `build: false` initially for testing

### Step 2: Create CPU Dockerfile

Create `tensorflow/inference/docker/X.Y/py3/Dockerfile.cpu`:

```dockerfile
# Note: Using TF Serving X.Y-1.0 as TF Serving X.Y.0 is not yet released (if applicable)
FROM tensorflow/serving:X.Y.0-devel as build_image
# OR if TF Serving doesn't exist:
FROM tensorflow/serving:X.Y-1.0-devel as build_image

FROM ubuntu:22.04 AS base_image
# ... base configuration ...

FROM base_image AS ec2
# ... EC2 stage ...

# Key sections to include:

# 1. Python installation from source
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
 && tar -xvf Python-${PYTHON_VERSION}.tgz \
 && cd Python-${PYTHON_VERSION} \
 && ./configure && make && make install \
 && rm -rf ../Python-${PYTHON_VERSION}*

# 2. Pip/setuptools upgrade with CVE fixes
RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    "setuptools>=75.8.2" \
    "wheel>=0.46.2" \
 && rm -rf /usr/local/lib/python*/site-packages/setuptools/_vendor/wheel*

# 3. TF Serving API installation (use matching TF Serving version)
RUN ${PIP} install --no-dependencies --no-cache-dir \
    tensorflow-serving-api=="X.Y.0"

# 4. Copy TF model server binary
COPY --from=build_image /usr/local/bin/tensorflow_model_server /usr/bin/tensorflow_model_server

# 5. Entrypoint script (IMPORTANT: use >> for appending)
RUN echo '#!/bin/bash \n\n' > /usr/bin/tf_serving_entrypoint.sh \
 && echo 'bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true' >> /usr/bin/tf_serving_entrypoint.sh \
 && echo '/usr/bin/tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} "$@"' >> /usr/bin/tf_serving_entrypoint.sh \
 && chmod +x /usr/bin/tf_serving_entrypoint.sh

# 6. License file (use previous version if current not available)
RUN curl https://aws-dlc-licenses.s3.amazonaws.com/tensorflow-X.Y/license.txt -o /license.txt
# OR if not available:
RUN curl https://aws-dlc-licenses.s3.amazonaws.com/tensorflow-X.Y-1/license.txt -o /license.txt

# SageMaker stage
FROM ec2 AS sagemaker
# ... SageMaker specific configurations ...
```

### Step 3: Create GPU Dockerfile

Create `tensorflow/inference/docker/X.Y/py3/cu125/Dockerfile.gpu`:

Key differences from CPU:
- Base image: `nvidia/cuda:12.5.0-base-ubuntu22.04`
- Build image: `tensorflow/serving:X.Y.0-devel-gpu`
- Install CUDA toolkit and cuDNN packages
- Use `tensorflow-serving-api-gpu` instead of `tensorflow-serving-api`
- Add CUDA compat scripts for SageMaker

**cuDNN Package Naming:**
```dockerfile
# For CUDA 12.x with cuDNN 9.x:
libcudnn9-cuda-12=${CUDNN_VERSION}
libcudnn9-dev-cuda-12=${CUDNN_VERSION}
```

### Step 4: Update Test Allowlists (if needed)

If using a different TF Serving version than the TF version, update `test/dlc_tests/sanity/test_pre_release.py`:

```python
# In test_tf_serving_version_cpu and _test_framework_and_cuda_version:
# TF Serving X.Y is not yet released, so TF X.Y images use TF Serving X.Y-1
expected_serving_version = tag_framework_version
if Version(tag_framework_version) >= Version("X.Y.0") and Version(tag_framework_version) < Version("X.Y+1.0"):
    expected_serving_version = "X.Y-1"

# In test_license_file:
# TF X.Y license is not yet available in S3, use TF X.Y-1 license
if framework == "tensorflow" and short_version == "X.Y":
    short_version = "X.Y-1"
```

### Step 5: Configure Build

Update `dlc_developer_config.toml`:

```toml
[build]
build_frameworks = ["tensorflow"]
build_job_types = ["inference"]

[buildspec_override]
tensorflow-inference-cpu = "buildspec-X-Y-sm.yml"
tensorflow-inference-gpu = "buildspec-X-Y-sm.yml"
```

---

## Common Issues and Solutions

### 1. YAML Parsing Error: `expected '<document start>'`
**Cause:** Version number `2.20` is parsed as float `2.2`
**Solution:** Quote version numbers: `"2.20"`

### 2. Build Error: `COPY failed: file not found`
**Cause:** Missing context files in buildspec
**Solution:** Add required files to buildspec context section (e.g., `bash_telemetry.sh`)

### 3. TF Serving Image Not Found
**Cause:** TF Serving version not yet released
**Solution:** Use previous TF Serving version that's compatible

### 4. cuDNN Package Not Found
**Cause:** Package naming changed between CUDA versions
**Solution:** Use correct naming: `libcudnn9-cuda-12` instead of `libcudnn9-cuda-12-5`

### 5. Exec Format Error
**Cause:** Shell script missing shebang due to `>` overwriting instead of `>>` appending
**Solution:** Use `>>` for all lines after the first:
```dockerfile
RUN echo '#!/bin/bash \n\n' > /script.sh \
 && echo 'line2' >> /script.sh \    # Note: >> not >
 && echo 'line3' >> /script.sh
```

### 6. License File Not Found (404)
**Cause:** License file for new version not uploaded to S3
**Solution:** Use previous version's license file temporarily

### 7. CVE in Vendored wheel (setuptools)
**Cause:** setuptools bundles an old wheel version
**Solution:** Remove vendored wheel after install:
```dockerfile
RUN rm -rf /usr/local/lib/python*/site-packages/setuptools/_vendor/wheel*
```

### 8. nvjpeg CVE
**Cause:** Vulnerable nvjpeg version in base CUDA image
**Solution:** Patch nvjpeg with latest available version:
```dockerfile
RUN wget https://developer.download.nvidia.com/compute/cuda/redist/libnvjpeg/linux-x86_64/libnvjpeg-linux-x86_64-12.4.0.76-archive.tar.xz \
 && # Extract and replace files
```

### 9. Black Formatting Failures
**Cause:** Python files not formatted with project's black config
**Solution:** Format with: `black --verbose -l 100 <file>` or `pip install black==24.8.0 && black <file>`

---

## Testing

### Local Testing

1. Build image locally:
```bash
docker build -t tf-inf-test -f tensorflow/inference/docker/X.Y/py3/Dockerfile.cpu --target sagemaker .
```

2. Run basic tests:
```bash
docker run -it tf-inf-test tensorflow_model_server --version
docker run -it tf-inf-test python -c "import tensorflow_serving"
```

### CI/CD Testing

Tests run automatically include:
- `test_tf_serving_version_cpu`: Verifies TF Serving version
- `test_tf_serving_api_version`: Verifies TF Serving API version
- `test_license_file`: Verifies license file matches S3
- `test_ecr_enhanced_scan`: Security vulnerability scanning
- `test_ec2_tensorflow_inference_telemetry`: Telemetry functionality

---

## Checklist

Before submitting PR:

- [ ] Version numbers quoted in YAML (`"2.20"` not `2.20`)
- [ ] `bash_telemetry.sh` added to buildspec context
- [ ] TF Serving version exists (or using compatible earlier version)
- [ ] TF Serving API version matches TF Serving version
- [ ] cuDNN package names correct for CUDA version
- [ ] Entrypoint script uses `>>` for appending (not `>`)
- [ ] License URL points to existing file in S3
- [ ] CVE fixes applied (wheel >= 0.46.2, setuptools >= 75.8.2, remove vendored wheel)
- [ ] nvjpeg patched if needed
- [ ] Test allowlists updated for TF Serving version mismatch
- [ ] Python files formatted with `black --verbose -l 100`
- [ ] Build tested locally

---

## Future Automation Opportunities

1. **Version Availability Check Script**: Auto-check TF Serving, TF Serving API, and license file availability
2. **Template Generation**: Auto-generate Dockerfiles from templates with version substitution
3. **CVE Auto-patching**: Script to automatically apply known CVE fixes
4. **Buildspec Generator**: Generate buildspec from template with version parameters
5. **Test Updater**: Auto-update test allowlists when using different TF Serving version

---

## References

- [TensorFlow Serving Releases](https://github.com/tensorflow/serving/releases)
- [TensorFlow Serving API on PyPI](https://pypi.org/project/tensorflow-serving-api/)
- [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [AWS DLC License Bucket](https://aws-dlc-licenses.s3.amazonaws.com/)
