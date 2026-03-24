# SGLang Existing Files

## File: .github/workflows/pr-sglang-ec2-amzn2023.yml

name: PR - SGLang EC2 AMZN2023

on:
pull_request:
branches: [main]
types: [opened, reopened, synchronize]
paths:
\- "docker/sglang/Dockerfile.amzn2023"
\- "scripts/sglang/dockerd_entrypoint.sh"
\- "scripts/sglang/sagemaker_entrypoint.sh"
\- "scripts/common/**"
\- "scripts/telemetry/**"
\- ".github/config/sglang-ec2-amzn2023.yml"
\- ".github/workflows/pr-sglang-ec2-amzn2023.yml"
\- "test/sanity/**"
\- "test/telemetry/**"

permissions:
contents: read
pull-requests: read

env:
FORCE_COLOR: "1"
CONFIG_FILE: ".github/config/sglang-ec2-amzn2023.yml"

jobs:
gatekeeper:
runs-on: ubuntu-latest
concurrency:
group: ${{ github.workflow }}-gate-${{ github.event.pull_request.number }}
cancel-in-progress: true
steps:
\- name: Checkout base branch (safe)
uses: actions/checkout@v5
with:
ref: ${{ github.event.pull_request.base.sha }}
fetch-depth: 1

```
  - name: Run permission gate (from base)
    uses: ./.github/actions/pr-permission-gate
```

load-config:
needs: [gatekeeper]
if: success()
runs-on: ubuntu-latest
outputs:
framework: ${{ steps.parse.outputs.framework }}
framework-version: ${{ steps.parse.outputs.framework-version }}
python-version: ${{ steps.parse.outputs.python-version }}
cuda-version: ${{ steps.parse.outputs.cuda-version }}
os-version: ${{ steps.parse.outputs.os-version }}
container-type: ${{ steps.parse.outputs.container-type }}
device-type: ${{ steps.parse.outputs.device-type }}
arch-type: ${{ steps.parse.outputs.arch-type }}
contributor: ${{ steps.parse.outputs.contributor }}
customer-type: ${{ steps.parse.outputs.customer-type }}
prod-image: ${{ steps.parse.outputs.prod-image }}
steps:
\- name: Checkout code
uses: actions/checkout@v5

```
  - name: Load configuration
    id: load
    uses: ./.github/actions/load-config
    with:
      config-file: ${{ env.CONFIG_FILE }}

  - name: Parse configuration
    id: parse
    run: |
      echo '${{ steps.load.outputs.config }}' > config.json
      echo "framework=$(jq -r '.common.framework' config.json)" >> $GITHUB_OUTPUT
      echo "framework-version=$(jq -r '.common.framework_version' config.json)" >> $GITHUB_OUTPUT
      echo "python-version=$(jq -r '.common.python_version' config.json)" >> $GITHUB_OUTPUT
      echo "cuda-version=$(jq -r '.common.cuda_version' config.json)" >> $GITHUB_OUTPUT
      echo "os-version=$(jq -r '.common.os_version' config.json)" >> $GITHUB_OUTPUT
      echo "container-type=$(jq -r '.common.job_type' config.json)" >> $GITHUB_OUTPUT
      echo "device-type=$(jq -r '.common.device_type // "gpu"' config.json)" >> $GITHUB_OUTPUT
      echo "arch-type=$(jq -r '.common.arch_type // "x86"' config.json)" >> $GITHUB_OUTPUT
      echo "contributor=$(jq -r '.common.contributor // "None"' config.json)" >> $GITHUB_OUTPUT
      echo "customer-type=$(jq -r '.common.customer_type // ""' config.json)" >> $GITHUB_OUTPUT
      echo "prod-image=$(jq -r '.common.prod_image' config.json)" >> $GITHUB_OUTPUT
```

check-changes:
needs: [gatekeeper]
if: success()
runs-on: ubuntu-latest
concurrency:
group: ${{ github.workflow }}-check-changes-${{ github.event.pull_request.number }}
cancel-in-progress: true
outputs:
build-change: ${{ steps.changes.outputs.build-change }}
sanity-test-change: ${{ steps.changes.outputs.sanity-test-change }}
telemetry-test-change: ${{ steps.changes.outputs.telemetry-test-change }}
steps:
\- name: Checkout DLC source
uses: actions/checkout@v5

```
  - name: Setup python
    uses: actions/setup-python@v6
    with:
      python-version: "3.12"

  - name: Run pre-commit
    uses: pre-commit/action@v3.0.1
    with:
      extra_args: --all-files

  - name: Detect file changes
    id: changes
    uses: dorny/paths-filter@v3
    with:
      filters: |
        build-change:
          - "docker/sglang/Dockerfile.amzn2023"
          - "scripts/sglang/dockerd_entrypoint.sh"
          - "scripts/sglang/sagemaker_entrypoint.sh"
          - "scripts/common/**"
          - "scripts/telemetry/**"
          - ".github/config/sglang-ec2-amzn2023.yml"
        sanity-test-change:
          - "test/sanity/**"
        telemetry-test-change:
          - "test/telemetry/**"
```

build-image:
needs: [check-changes, load-config]
if: needs.check-changes.outputs.build-change == 'true'
runs-on:
\- codebuild-runner-${{ github.run_id }}-${{ github.run_attempt }}
fleet:x86-sglang-build-runner
buildspec-override:true
timeout-minutes: 720
concurrency:
group: ${{ github.workflow }}-build-image-${{ github.event.pull_request.number }}
cancel-in-progress: true
outputs:
ci-image: ${{ steps.build.outputs.image-uri }}
steps:
\- name: Checkout code
uses: actions/checkout@v5

```
  - name: Build image
    id: build
    uses: ./.github/actions/build-image
    with:
      framework: ${{ needs.load-config.outputs.framework }}
      target: sglang-ec2
      base-image: nvidia/cuda:12.9.1-devel-amzn2023
      framework-version: ${{ needs.load-config.outputs.framework-version }}
      container-type: ${{ needs.load-config.outputs.container-type }}
      aws-account-id: ${{ vars.CI_AWS_ACCOUNT_ID }}
      aws-region: ${{ vars.AWS_REGION }}
      tag-pr: ${{ needs.load-config.outputs.framework }}-${{ needs.load-config.outputs.framework-version }}-gpu-${{ needs.load-config.outputs.python-version }}-${{ needs.load-config.outputs.cuda-version }}-${{ needs.load-config.outputs.os-version }}-ec2-pr-${{ github.event.pull_request.number }}
      dockerfile-path: docker/${{ needs.load-config.outputs.framework }}/Dockerfile.amzn2023
      arch-type: ${{ needs.load-config.outputs.arch-type }}
      device-type: ${{ needs.load-config.outputs.device-type }}
      cuda-version: ${{ needs.load-config.outputs.cuda-version }}
      python-version: ${{ needs.load-config.outputs.python-version }}
      os-version: ${{ needs.load-config.outputs.os-version }}
      contributor: ${{ needs.load-config.outputs.contributor }}
      customer-type: ${{ needs.load-config.outputs.customer-type }}
```

sanity-test:
needs: [check-changes, build-image, load-config]
if: |
always() && !failure() && !cancelled() &&
(needs.check-changes.outputs.build-change == 'true' || needs.check-changes.outputs.sanity-test-change == 'true')
concurrency:
group: ${{ github.workflow }}-sanity-test-${{ github.event.pull_request.number }}
cancel-in-progress: true
uses: ./.github/workflows/reusable-sanity-tests.yml
with:
image-uri: ${{ needs.build-image.result == 'success' && needs.build-image.outputs.ci-image || format('{0}.dkr.ecr.{1}.amazonaws.com/{2}', vars.PROD_AWS_ACCOUNT_ID, vars.AWS_REGION, needs.load-config.outputs.prod-image) }}
aws-account-id: ${{ needs.build-image.result == 'success' && vars.CI_AWS_ACCOUNT_ID || vars.PROD_AWS_ACCOUNT_ID }}
aws-region: ${{ vars.AWS_REGION }}
framework: ${{ needs.load-config.outputs.framework }}
framework-version: ${{ needs.load-config.outputs.framework-version }}
python-version: ${{ needs.load-config.outputs.python-version }}
cuda-version: ${{ needs.load-config.outputs.cuda-version }}
os-version: ${{ needs.load-config.outputs.os-version }}
customer-type: ${{ needs.load-config.outputs.customer-type }}
arch-type: ${{ needs.load-config.outputs.arch-type }}
device-type: ${{ needs.load-config.outputs.device-type }}
contributor: ${{ needs.load-config.outputs.contributor }}
container-type: ${{ needs.load-config.outputs.container-type }}

security-test:
needs: [build-image, load-config]
if: success()
concurrency:
group: ${{ github.workflow }}-security-test-${{ github.event.pull_request.number }}
cancel-in-progress: true
uses: ./.github/workflows/reusable-security-tests.yml
with:
image-uri: ${{ needs.build-image.result == 'success' && needs.build-image.outputs.ci-image || format('{0}.dkr.ecr.{1}.amazonaws.com/{2}', vars.PROD_AWS_ACCOUNT_ID, vars.AWS_REGION, needs.load-config.outputs.prod-image) }}
aws-account-id: ${{ needs.build-image.result == 'success' && vars.CI_AWS_ACCOUNT_ID || vars.PROD_AWS_ACCOUNT_ID }}
aws-region: ${{ vars.AWS_REGION }}
framework: ${{ needs.load-config.outputs.framework }}
framework-version: ${{ needs.load-config.outputs.framework-version }}

telemetry-test:
needs: [check-changes, build-image, load-config]
if: |
always() && !failure() && !cancelled() &&
(needs.check-changes.outputs.build-change == 'true' || needs.check-changes.outputs.telemetry-test-change == 'true')
concurrency:
group: ${{ github.workflow }}-telemetry-test-${{ github.event.pull_request.number }}
cancel-in-progress: false
uses: ./.github/workflows/reusable-telemetry-tests.yml
with:
image-uri: ${{ needs.build-image.result == 'success' && needs.build-image.outputs.ci-image || format('{0}.dkr.ecr.{1}.amazonaws.com/{2}', vars.PROD_AWS_ACCOUNT_ID, vars.AWS_REGION, needs.load-config.outputs.prod-image) }}
aws-account-id: ${{ needs.build-image.result == 'success' && vars.CI_AWS_ACCOUNT_ID || vars.PROD_AWS_ACCOUNT_ID }}
aws-region: ${{ vars.AWS_REGION }}
framework: ${{ needs.load-config.outputs.framework }}
framework-version: ${{ needs.load-config.outputs.framework-version }}
container-type: ${{ needs.load-config.outputs.container-type }}

local-benchmark-test:
needs: [build-image, load-config]
if: success()
concurrency:
group: ${{ github.workflow }}-local-benchmark-test-${{ github.event.pull_request.number }}
cancel-in-progress: true
runs-on:
\- codebuild-runner-${{ github.run_id }}-${{ github.run_attempt }}
fleet:x86-g6xl-runner
buildspec-override:true
env:
TEST_ARTIFACTS_DIRECTORY: "/test_artifacts/sglang"
steps:
\- name: Checkout DLC source
uses: actions/checkout@v5

```
  - name: Container pull
    uses: ./.github/actions/ecr-authenticate
    with:
      aws-account-id: ${{ needs.build-image.result == 'success' && vars.CI_AWS_ACCOUNT_ID || vars.PROD_AWS_ACCOUNT_ID }}
      aws-region: ${{ vars.AWS_REGION }}
      image-uri: ${{ needs.build-image.result == 'success' && needs.build-image.outputs.ci-image || format('{0}.dkr.ecr.{1}.amazonaws.com/{2}', vars.PROD_AWS_ACCOUNT_ID, vars.AWS_REGION, needs.load-config.outputs.prod-image) }}

  - name: Setup for SGLang datasets
    run: |
      mkdir -p ${TEST_ARTIFACTS_DIRECTORY}/dataset
      if [ ! -f ${TEST_ARTIFACTS_DIRECTORY}/dataset/ShareGPT_V3_unfiltered_cleaned_split.json ]; then
          echo "Downloading ShareGPT dataset..."
          wget -P ${TEST_ARTIFACTS_DIRECTORY}/dataset https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
      else
          echo "ShareGPT dataset already exists. Skipping download."
      fi

  - name: Start container
    env:
      IMAGE: ${{ needs.build-image.result == 'success' && needs.build-image.outputs.ci-image || format('{0}.dkr.ecr.{1}.amazonaws.com/{2}', vars.PROD_AWS_ACCOUNT_ID, vars.AWS_REGION, needs.load-config.outputs.prod-image) }}
      HF_TOKEN: ${{ secrets.HUGGING_FACE_HUB_TOKEN }}
    run: |
      CONTAINER_ID=$(docker run -d -it --rm --gpus=all \
        -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
        -v ${TEST_ARTIFACTS_DIRECTORY}/dataset:/dataset \
        -p 30000:30000 \
        -e HF_TOKEN=${HF_TOKEN} \
        ${IMAGE} \
        --model-path Qwen/Qwen3-0.6B \
        --reasoning-parser qwen3 \
        --host 127.0.0.1 \
        --port 30000)
      echo "CONTAINER_ID=${CONTAINER_ID}" >> ${GITHUB_ENV}
      echo "Waiting for serving endpoint startup ..."
      sleep 120s
      docker logs ${CONTAINER_ID}

  - name: Run SGLang benchmark
    run: |
      docker exec ${CONTAINER_ID} python3 -m sglang.bench_serving \
      --backend sglang \
      --host 127.0.0.1 --port 30000 \
      --num-prompts 1000 \
      --model Qwen/Qwen3-0.6B \
      --dataset-name sharegpt \
      --dataset-path /dataset/ShareGPT_V3_unfiltered_cleaned_split.json
```

______________________________________________________________________

## File: .github/workflows/pr-sglang-sagemaker-amzn2023.yml

name: PR - SGLang SageMaker AMZN2023

on:
pull_request:
branches: [main]
types: [opened, reopened, synchronize]
paths:
\- "docker/sglang/Dockerfile.amzn2023"
\- "scripts/sglang/dockerd_entrypoint.sh"
\- "scripts/sglang/sagemaker_entrypoint.sh"
\- "scripts/common/**"
\- "scripts/telemetry/**"
\- ".github/config/sglang-sagemaker-amzn2023.yml"
\- ".github/workflows/pr-sglang-sagemaker-amzn2023.yml"
\- "test/sanity/**"
\- "test/telemetry/**"
\- "test/sglang/sagemaker/\*\*"

permissions:
contents: read
pull-requests: read

env:
FORCE_COLOR: "1"
CONFIG_FILE: ".github/config/sglang-sagemaker-amzn2023.yml"

jobs:
gatekeeper:
runs-on: ubuntu-latest
concurrency:
group: ${{ github.workflow }}-gate-${{ github.event.pull_request.number }}
cancel-in-progress: true
steps:
\- name: Checkout base branch (safe)
uses: actions/checkout@v5
with:
ref: ${{ github.event.pull_request.base.sha }}
fetch-depth: 1

```
  - name: Run permission gate (from base)
    uses: ./.github/actions/pr-permission-gate
```

load-config:
needs: [gatekeeper]
if: success()
runs-on: ubuntu-latest
outputs:
framework: ${{ steps.parse.outputs.framework }}
framework-version: ${{ steps.parse.outputs.framework-version }}
python-version: ${{ steps.parse.outputs.python-version }}
cuda-version: ${{ steps.parse.outputs.cuda-version }}
os-version: ${{ steps.parse.outputs.os-version }}
container-type: ${{ steps.parse.outputs.container-type }}
device-type: ${{ steps.parse.outputs.device-type }}
arch-type: ${{ steps.parse.outputs.arch-type }}
contributor: ${{ steps.parse.outputs.contributor }}
customer-type: ${{ steps.parse.outputs.customer-type }}
prod-image: ${{ steps.parse.outputs.prod-image }}
steps:
\- name: Checkout code
uses: actions/checkout@v5

```
  - name: Load configuration
    id: load
    uses: ./.github/actions/load-config
    with:
      config-file: ${{ env.CONFIG_FILE }}

  - name: Parse configuration
    id: parse
    run: |
      echo '${{ steps.load.outputs.config }}' > config.json
      echo "framework=$(jq -r '.common.framework' config.json)" >> $GITHUB_OUTPUT
      echo "framework-version=$(jq -r '.common.framework_version' config.json)" >> $GITHUB_OUTPUT
      echo "python-version=$(jq -r '.common.python_version' config.json)" >> $GITHUB_OUTPUT
      echo "cuda-version=$(jq -r '.common.cuda_version' config.json)" >> $GITHUB_OUTPUT
      echo "os-version=$(jq -r '.common.os_version' config.json)" >> $GITHUB_OUTPUT
      echo "container-type=$(jq -r '.common.job_type' config.json)" >> $GITHUB_OUTPUT
      echo "device-type=$(jq -r '.common.device_type // "gpu"' config.json)" >> $GITHUB_OUTPUT
      echo "arch-type=$(jq -r '.common.arch_type // "x86"' config.json)" >> $GITHUB_OUTPUT
      echo "contributor=$(jq -r '.common.contributor // "None"' config.json)" >> $GITHUB_OUTPUT
      echo "customer-type=$(jq -r '.common.customer_type // ""' config.json)" >> $GITHUB_OUTPUT
      echo "prod-image=$(jq -r '.common.prod_image' config.json)" >> $GITHUB_OUTPUT
```

check-changes:
needs: [gatekeeper]
if: success()
runs-on: ubuntu-latest
concurrency:
group: ${{ github.workflow }}-check-changes-${{ github.event.pull_request.number }}
cancel-in-progress: true
outputs:
build-change: ${{ steps.changes.outputs.build-change }}
sanity-test-change: ${{ steps.changes.outputs.sanity-test-change }}
sagemaker-test-change: ${{ steps.changes.outputs.sagemaker-test-change }}
telemetry-test-change: ${{ steps.changes.outputs.telemetry-test-change }}
steps:
\- name: Checkout DLC source
uses: actions/checkout@v5

```
  - name: Setup python
    uses: actions/setup-python@v6
    with:
      python-version: "3.12"

  - name: Run pre-commit
    uses: pre-commit/action@v3.0.1
    with:
      extra_args: --all-files

  - name: Detect file changes
    id: changes
    uses: dorny/paths-filter@v3
    with:
      filters: |
        build-change:
          - "docker/sglang/Dockerfile.amzn2023"
          - "scripts/sglang/dockerd_entrypoint.sh"
          - "scripts/sglang/sagemaker_entrypoint.sh"
          - "scripts/common/**"
          - "scripts/telemetry/**"
          - ".github/config/sglang-sagemaker-amzn2023.yml"
        sanity-test-change:
          - "test/sanity/**"
        telemetry-test-change:
          - "test/telemetry/**"
        sagemaker-test-change:
          - "test/sglang/sagemaker/**"
```

build-image:
needs: [check-changes, load-config]
if: needs.check-changes.outputs.build-change == 'true'
runs-on:
\- codebuild-runner-${{ github.run_id }}-${{ github.run_attempt }}
fleet:x86-sglang-build-runner
buildspec-override:true
timeout-minutes: 720
concurrency:
group: ${{ github.workflow }}-build-image-${{ github.event.pull_request.number }}
cancel-in-progress: true
outputs:
ci-image: ${{ steps.build.outputs.image-uri }}
steps:
\- name: Checkout code
uses: actions/checkout@v5

```
  - name: Build image
    id: build
    uses: ./.github/actions/build-image
    with:
      framework: ${{ needs.load-config.outputs.framework }}
      target: sglang-sagemaker
      base-image: nvidia/cuda:12.9.1-devel-amzn2023
      framework-version: ${{ needs.load-config.outputs.framework-version }}
      container-type: ${{ needs.load-config.outputs.container-type }}
      aws-account-id: ${{ vars.CI_AWS_ACCOUNT_ID }}
      aws-region: ${{ vars.AWS_REGION }}
      tag-pr: ${{ needs.load-config.outputs.framework }}-${{ needs.load-config.outputs.framework-version }}-gpu-${{ needs.load-config.outputs.python-version }}-${{ needs.load-config.outputs.cuda-version }}-${{ needs.load-config.outputs.os-version }}-sagemaker-pr-${{ github.event.pull_request.number }}
      dockerfile-path: docker/${{ needs.load-config.outputs.framework }}/Dockerfile.amzn2023
      arch-type: ${{ needs.load-config.outputs.arch-type }}
      device-type: ${{ needs.load-config.outputs.device-type }}
      cuda-version: ${{ needs.load-config.outputs.cuda-version }}
      python-version: ${{ needs.load-config.outputs.python-version }}
      os-version: ${{ needs.load-config.outputs.os-version }}
      contributor: ${{ needs.load-config.outputs.contributor }}
      customer-type: ${{ needs.load-config.outputs.customer-type }}
```

sanity-test:
needs: [check-changes, build-image, load-config]
if: |
always() && !failure() && !cancelled() &&
(needs.check-changes.outputs.build-change == 'true' || needs.check-changes.outputs.sanity-test-change == 'true')
concurrency:
group: ${{ github.workflow }}-sanity-test-${{ github.event.pull_request.number }}
cancel-in-progress: true
uses: ./.github/workflows/reusable-sanity-tests.yml
with:
image-uri: ${{ needs.build-image.result == 'success' && needs.build-image.outputs.ci-image || format('{0}.dkr.ecr.{1}.amazonaws.com/{2}', vars.PROD_AWS_ACCOUNT_ID, vars.AWS_REGION, needs.load-config.outputs.prod-image) }}
aws-account-id: ${{ needs.build-image.result == 'success' && vars.CI_AWS_ACCOUNT_ID || vars.PROD_AWS_ACCOUNT_ID }}
aws-region: ${{ vars.AWS_REGION }}
framework: ${{ needs.load-config.outputs.framework }}
framework-version: ${{ needs.load-config.outputs.framework-version }}
python-version: ${{ needs.load-config.outputs.python-version }}
cuda-version: ${{ needs.load-config.outputs.cuda-version }}
os-version: ${{ needs.load-config.outputs.os-version }}
customer-type: ${{ needs.load-config.outputs.customer-type }}
arch-type: ${{ needs.load-config.outputs.arch-type }}
device-type: ${{ needs.load-config.outputs.device-type }}
contributor: ${{ needs.load-config.outputs.contributor }}
container-type: ${{ needs.load-config.outputs.container-type }}

security-test:
needs: [build-image, load-config]
if: success()
concurrency:
group: ${{ github.workflow }}-security-test-${{ github.event.pull_request.number }}
cancel-in-progress: true
uses: ./.github/workflows/reusable-security-tests.yml
with:
image-uri: ${{ needs.build-image.result == 'success' && needs.build-image.outputs.ci-image || format('{0}.dkr.ecr.{1}.amazonaws.com/{2}', vars.PROD_AWS_ACCOUNT_ID, vars.AWS_REGION, needs.load-config.outputs.prod-image) }}
aws-account-id: ${{ needs.build-image.result == 'success' && vars.CI_AWS_ACCOUNT_ID || vars.PROD_AWS_ACCOUNT_ID }}
aws-region: ${{ vars.AWS_REGION }}
framework: ${{ needs.load-config.outputs.framework }}
framework-version: ${{ needs.load-config.outputs.framework-version }}

telemetry-test:
needs: [check-changes, build-image, load-config]
if: |
always() && !failure() && !cancelled() &&
(needs.check-changes.outputs.build-change == 'true' || needs.check-changes.outputs.telemetry-test-change == 'true')
concurrency:
group: ${{ github.workflow }}-telemetry-test-${{ github.event.pull_request.number }}
cancel-in-progress: false
uses: ./.github/workflows/reusable-telemetry-tests.yml
with:
image-uri: ${{ needs.build-image.result == 'success' && needs.build-image.outputs.ci-image || format('{0}.dkr.ecr.{1}.amazonaws.com/{2}', vars.PROD_AWS_ACCOUNT_ID, vars.AWS_REGION, needs.load-config.outputs.prod-image) }}
aws-account-id: ${{ needs.build-image.result == 'success' && vars.CI_AWS_ACCOUNT_ID || vars.PROD_AWS_ACCOUNT_ID }}
aws-region: ${{ vars.AWS_REGION }}
framework: ${{ needs.load-config.outputs.framework }}
framework-version: ${{ needs.load-config.outputs.framework-version }}
container-type: ${{ needs.load-config.outputs.container-type }}

local-benchmark-test:
needs: [build-image, load-config]
if: success()
concurrency:
group: ${{ github.workflow }}-local-benchmark-test-${{ github.event.pull_request.number }}
cancel-in-progress: true
runs-on:
\- codebuild-runner-${{ github.run_id }}-${{ github.run_attempt }}
fleet:x86-g6xl-runner
buildspec-override:true
env:
TEST_ARTIFACTS_DIRECTORY: "/test_artifacts/sglang"
steps:
\- name: Checkout DLC source
uses: actions/checkout@v5

```
  - name: Container pull
    uses: ./.github/actions/ecr-authenticate
    with:
      aws-account-id: ${{ needs.build-image.result == 'success' && vars.CI_AWS_ACCOUNT_ID || vars.PROD_AWS_ACCOUNT_ID }}
      aws-region: ${{ vars.AWS_REGION }}
      image-uri: ${{ needs.build-image.result == 'success' && needs.build-image.outputs.ci-image || format('{0}.dkr.ecr.{1}.amazonaws.com/{2}', vars.PROD_AWS_ACCOUNT_ID, vars.AWS_REGION, needs.load-config.outputs.prod-image) }}

  - name: Setup for SGLang datasets
    run: |
      mkdir -p ${TEST_ARTIFACTS_DIRECTORY}/dataset
      if [ ! -f ${TEST_ARTIFACTS_DIRECTORY}/dataset/ShareGPT_V3_unfiltered_cleaned_split.json ]; then
          echo "Downloading ShareGPT dataset..."
          wget -P ${TEST_ARTIFACTS_DIRECTORY}/dataset https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
      else
          echo "ShareGPT dataset already exists. Skipping download."
      fi

  - name: Start container
    env:
      IMAGE: ${{ needs.build-image.result == 'success' && needs.build-image.outputs.ci-image || format('{0}.dkr.ecr.{1}.amazonaws.com/{2}', vars.PROD_AWS_ACCOUNT_ID, vars.AWS_REGION, needs.load-config.outputs.prod-image) }}
      HF_TOKEN: ${{ secrets.HUGGING_FACE_HUB_TOKEN }}
    run: |
      CONTAINER_ID=$(docker run -d -it --rm --gpus=all \
        -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
        -v ${TEST_ARTIFACTS_DIRECTORY}/dataset:/dataset \
        -p 30000:30000 \
        -e SM_SGLANG_MODEL_PATH=Qwen/Qwen3-0.6B \
        -e SM_SGLANG_REASONING_PARSER=qwen3 \
        -e SM_SGLANG_HOST=127.0.0.1 \
        -e SM_SGLANG_PORT=30000 \
        -e HF_TOKEN=${HF_TOKEN} \
        ${IMAGE})
      echo "CONTAINER_ID=${CONTAINER_ID}" >> ${GITHUB_ENV}
      echo "Waiting for serving endpoint startup ..."
      sleep 120s
      docker logs ${CONTAINER_ID}

  - name: Run SGLang benchmark
    run: |
      docker exec ${CONTAINER_ID} python3 -m sglang.bench_serving \
      --backend sglang \
      --host 127.0.0.1 --port 30000 \
      --num-prompts 1000 \
      --model Qwen/Qwen3-0.6B \
      --dataset-name sharegpt \
      --dataset-path /dataset/ShareGPT_V3_unfiltered_cleaned_split.json
```

endpoint-test:
needs: [check-changes, build-image, load-config]
if: |
always() && !failure() && !cancelled() &&
(needs.check-changes.outputs.build-change == 'true' || needs.check-changes.outputs.sagemaker-test-change == 'true')
concurrency:
group: ${{ github.workflow }}-endpoint-test-${{ github.event.pull_request.number }}
cancel-in-progress: false
uses: ./.github/workflows/reusable-sglang-sagemaker-tests.yml
with:
image-uri: ${{ needs.build-image.result == 'success' && needs.build-image.outputs.ci-image || format('{0}.dkr.ecr.{1}.amazonaws.com/{2}', vars.PROD_AWS_ACCOUNT_ID, vars.AWS_REGION, needs.load-config.outputs.prod-image) }}

______________________________________________________________________

## File: .github/workflows/reusable-sglang-upstream-tests.yml

name: Reusable SGLang Upstream Tests

permissions:
contents: read

on:
workflow_call:
inputs:
image-uri:
description: 'Image URI to test'
required: true
type: string
aws-account-id:
description: 'AWS account ID for ECR authentication'
required: true
type: string
aws-region:
description: 'AWS region for ECR authentication'
required: true
type: string
framework-version:
description: 'SGLang framework version (e.g., 0.5.9)'
required: true
type: string
benchmark-start-command:
description: 'Docker run command to start the benchmark container (must output CONTAINER_ID)'
required: true
type: string
run-srt-backend-test:
description: 'Whether to run the SRT backend test suite (default: true)'
required: false
type: boolean
default: true

env:
TEST_ARTIFACTS_DIRECTORY: "/test_artifacts/sglang"

jobs:
local-benchmark-test:
runs-on:
\- codebuild-runner-${{ github.run_id }}-${{ github.run_attempt }}
fleet:x86-g6xl-runner
buildspec-override:true
steps:
\- name: Checkout DLC source
uses: actions/checkout@v5

```
  - name: Container pull
    uses: ./.github/actions/ecr-authenticate
    with:
      aws-account-id: ${{ inputs.aws-account-id }}
      aws-region: ${{ inputs.aws-region }}
      image-uri: ${{ inputs.image-uri }}

  - name: Setup for SGLang datasets
    run: |
      mkdir -p ${TEST_ARTIFACTS_DIRECTORY}/dataset
      if [ ! -f ${TEST_ARTIFACTS_DIRECTORY}/dataset/ShareGPT_V3_unfiltered_cleaned_split.json ]; then
          echo "Downloading ShareGPT dataset..."
          wget -P ${TEST_ARTIFACTS_DIRECTORY}/dataset https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
      else
          echo "ShareGPT dataset already exists. Skipping download."
      fi

  - name: Start container
    env:
      IMAGE: ${{ inputs.image-uri }}
      HF_TOKEN: ${{ secrets.HUGGING_FACE_HUB_TOKEN }}
    run: |
      CONTAINER_ID=$(${{ inputs.benchmark-start-command }})
      echo "CONTAINER_ID=${CONTAINER_ID}" >> ${GITHUB_ENV}
      echo "Waiting for serving endpoint startup ..."
      sleep 120s
      docker logs ${CONTAINER_ID}

  - name: Run SGLang benchmark
    run: |
      docker exec ${CONTAINER_ID} python3 -m sglang.bench_serving \
      --backend sglang \
      --host 127.0.0.1 --port 30000 \
      --num-prompts 1000 \
      --model Qwen/Qwen3-0.6B \
      --dataset-name sharegpt \
      --dataset-path /dataset/ShareGPT_V3_unfiltered_cleaned_split.json
```

srt-backend-test:
if: ${{ inputs.run-srt-backend-test }}
runs-on:
\- codebuild-runner-${{ github.run_id }}-${{ github.run_attempt }}
fleet:x86-g6exl-runner
buildspec-override:true
steps:
\- name: Checkout DLC source
uses: actions/checkout@v5

```
  - name: Container pull
    uses: ./.github/actions/ecr-authenticate
    with:
      aws-account-id: ${{ inputs.aws-account-id }}
      aws-region: ${{ inputs.aws-region }}
      image-uri: ${{ inputs.image-uri }}

  - name: Checkout SGLang tests
    uses: actions/checkout@v5
    with:
      repository: sgl-project/sglang
      ref: v${{ inputs.framework-version }}
      path: sglang_source

  - name: Start container
    run: |
      CONTAINER_ID=$(docker run -d -it --rm --gpus=all --entrypoint /bin/bash \
        -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
        -v ./sglang_source:/workdir --workdir /workdir \
        -e HF_TOKEN=${{ secrets.HUGGING_FACE_HUB_TOKEN }} \
        ${{ inputs.image-uri }})
      echo "CONTAINER_ID=${CONTAINER_ID}" >> ${GITHUB_ENV}

  - name: Setup for SGLang tests
    run: |
      docker exec ${CONTAINER_ID} sh -c '
        set -eux
        # https://github.com/sgl-project/sglang/blob/v0.5.7/scripts/ci/ci_install_dependency.sh#L78C8-L80
        # our CI suffers the same issue
        export IS_BLACKWELL=1

        bash scripts/ci/cuda/ci_install_dependency.sh
      '

  - name: Run SGLang tests
    run: |
      docker exec ${CONTAINER_ID} sh -c '
        set -eux
        nvidia-smi

        # SRT backend Test with increased timeout
        cd /workdir/test
        export SGLANG_TIMEOUT=600
        python3 run_suite.py --hw cuda --suite stage-a-test-1
      '
```

______________________________________________________________________

## File: .github/config/sglang-ec2-amzn2023.yml

# SGLang EC2 AL2023 Image Configuration

image:
name: "sglang-ec2-amzn2023"
description: "SGLang for EC2 instances (AL2023, built from source)"
common:
framework: "sglang"
framework_version: "0.5.9"
job_type: "general"
python_version: "py312"
cuda_version: "cu129"
os_version: "amzn2023"
customer_type: "ec2"
arch_type: "x86"
prod_image: "sglang:0.5-gpu-py312-amzn2023-ec2"
device_type: "gpu"
contributor: "None"
release:
release: false
force_release: false
public_registry: false
private_registry: false
enable_soci: false
environment: gamma

______________________________________________________________________

## File: .github/config/sglang-sagemaker-amzn2023.yml

# SGLang SageMaker AL2023 Image Configuration

image:
name: "sglang-sagemaker-amzn2023"
description: "SGLang for SageMaker on Amazon Linux 2023"
common:
framework: "sglang"
framework_version: "0.5.9"
job_type: "general"
python_version: "py312"
cuda_version: "cu129"
os_version: "amzn2023"
customer_type: "sagemaker"
arch_type: "x86"
prod_image: "sglang:0.5-gpu-py312-amzn2023"
device_type: "gpu"
contributor: "None"
release:
release: false
force_release: false
public_registry: false
private_registry: false
enable_soci: false
environment: gamma

______________________________________________________________________

## File: scripts/sglang/sagemaker_entrypoint.sh

#!/bin/bash

# Check if telemetry file exists before executing

# Execute telemetry script if it exists, suppress errors

bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

if command -v nvidia-smi >/dev/null 2>&1 && command -v nvcc >/dev/null 2>&1; then
bash /usr/local/bin/start_cuda_compat.sh
fi

echo "Starting server"

PREFIX="SM_SGLANG\_"
ARG_PREFIX="--"

ARGS=()

while IFS='=' read -r key value; do
arg_name=$(echo "${key#"${PREFIX}"}" | tr '[:upper:]' '[:lower:]' | tr '\_' '-')

```
# Handle boolean flags: true -> flag only, false -> skip entirely
lower_value=$(echo "$value" | tr '[:upper:]' '[:lower:]')
if [ "$lower_value" = "true" ]; then
    ARGS+=("${ARG_PREFIX}${arg_name}")
elif [ "$lower_value" = "false" ]; then
    continue
else
    ARGS+=("${ARG_PREFIX}${arg_name}")
    if [ -n "$value" ]; then
        ARGS+=("$value")
    fi
fi
```

done < \<(env | grep "^${PREFIX}")

# Add default port only if not already set

if ! \[\[ " ${ARGS[@]} " =~ " --port " \]\]; then
ARGS+=(--port "${SM_SGLANG_PORT:-8080}")
fi

# Add default host only if not already set

if ! \[\[ " ${ARGS[@]} " =~ " --host " \]\]; then
ARGS+=(--host "${SM_SGLANG_HOST:-0.0.0.0}")
fi

# Add default model-path only if not already set

if ! \[\[ " ${ARGS[@]} " =~ " --model-path " \]\]; then
ARGS+=(--model-path "${SM_SGLANG_MODEL_PATH:-/opt/ml/model}")
fi

echo "Running command: exec python3 -m sglang.launch_server ${ARGS[@]}"
exec python3 -m sglang.launch_server "${ARGS[@]}"

______________________________________________________________________

## File: scripts/sglang/dockerd_entrypoint.sh

#!/usr/bin/env bash

# Check if telemetry file exists before executing

# Execute telemetry script if it exists, suppress errors

bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

python3 -m sglang.launch_server "$@"
