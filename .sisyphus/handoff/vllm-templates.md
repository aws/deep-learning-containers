# vLLM AMZN2023 Test Infrastructure Templates

______________________________________________________________________

## File: .github/workflows/pr-vllm-ec2-amzn2023.yml

```yaml
name: PR - vLLM EC2 AMZN2023

on:
  pull_request:
    branches: [main]
    types: [opened, reopened, synchronize]
    paths:
      - "docker/vllm/Dockerfile.amzn2023"
      - "scripts/vllm/amzn2023/**"
      - "scripts/vllm/dockerd_entrypoint.sh"
      - "scripts/vllm/sagemaker_entrypoint.sh"
      - "scripts/common/**"
      - "scripts/telemetry/**"
      - ".github/config/vllm-ec2-amzn2023.yml"
      - ".github/workflows/pr-vllm-ec2-amzn2023.yml"
      - ".github/workflows/reusable-vllm-upstream-tests.yml"
      - ".github/workflows/reusable-vllm-model-tests.yml"
      - "test/sanity/**"
      - "test/telemetry/**"

permissions:
  contents: read
  pull-requests: read

env:
  FORCE_COLOR: "1"
  CONFIG_FILE: ".github/config/vllm-ec2-amzn2023.yml"

jobs:
  gatekeeper:
    runs-on: ubuntu-latest
    concurrency:
      group: ${{ github.workflow }}-gate-${{ github.event.pull_request.number }}
      cancel-in-progress: true
    steps:
      - name: Checkout base branch (safe)
        uses: actions/checkout@v5
        with:
          ref: ${{ github.event.pull_request.base.sha }}
          fetch-depth: 1

      - name: Run permission gate (from base)
        uses: ./.github/actions/pr-permission-gate

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
      - name: Checkout code
        uses: actions/checkout@v5

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
      - name: Checkout DLC source
        uses: actions/checkout@v5

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
              - "docker/vllm/Dockerfile.amzn2023"
              - "scripts/vllm/amzn2023/**"
              - "scripts/vllm/dockerd_entrypoint.sh"
              - "scripts/vllm/sagemaker_entrypoint.sh"
              - "scripts/common/**"
              - "scripts/telemetry/**"
              - ".github/config/vllm-ec2-amzn2023.yml"
            sanity-test-change:
              - "test/sanity/**"
            telemetry-test-change:
              - "test/telemetry/**"

  build-image:
    needs: [check-changes, load-config]
    if: needs.check-changes.outputs.build-change == 'true'
    runs-on:
      - codebuild-runner-${{ github.run_id }}-${{ github.run_attempt }}
        fleet:x86-vllm-build-runner
        buildspec-override:true
    timeout-minutes: 720
    concurrency:
      group: ${{ github.workflow }}-build-image-${{ github.event.pull_request.number }}
      cancel-in-progress: true
    outputs:
      ci-image: ${{ steps.build.outputs.image-uri }}
    steps:
      - name: Checkout code
        uses: actions/checkout@v5

      - name: Build image
        id: build
        uses: ./.github/actions/build-image
        with:
          framework: ${{ needs.load-config.outputs.framework }}
          target: vllm-ec2-amzn2023
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

  upstream-tests:
    needs: [build-image, load-config]
    if: success()
    concurrency:
      group: ${{ github.workflow }}-upstream-tests-${{ github.event.pull_request.number }}
      cancel-in-progress: true
    uses: ./.github/workflows/reusable-vllm-upstream-tests.yml
    with:
      image-uri: ${{ needs.build-image.result == 'success' && needs.build-image.outputs.ci-image || format('{0}.dkr.ecr.{1}.amazonaws.com/{2}', vars.PROD_AWS_ACCOUNT_ID, vars.AWS_REGION, needs.load-config.outputs.prod-image) }}
      aws-account-id: ${{ needs.build-image.result == 'success' && vars.CI_AWS_ACCOUNT_ID || vars.PROD_AWS_ACCOUNT_ID }}
      aws-region: ${{ vars.AWS_REGION }}
      framework-version: ${{ needs.load-config.outputs.framework-version }}
      setup-script: scripts/vllm/amzn2023/vllm_test_setup.sh
      example-test-script: scripts/vllm/amzn2023/vllm_ec2_examples_test.sh
    secrets: inherit

  model-smoke-tests:
    needs: [build-image, load-config]
    if: success()
    concurrency:
      group: ${{ github.workflow }}-model-smoke-tests-${{ github.event.pull_request.number }}
      cancel-in-progress: true
    uses: ./.github/workflows/reusable-vllm-model-tests.yml
    with:
      image-uri: ${{ needs.build-image.result == 'success' && needs.build-image.outputs.ci-image || format('{0}.dkr.ecr.{1}.amazonaws.com/{2}', vars.PROD_AWS_ACCOUNT_ID, vars.AWS_REGION, needs.load-config.outputs.prod-image) }}
      aws-account-id: ${{ needs.build-image.result == 'success' && vars.CI_AWS_ACCOUNT_ID || vars.PROD_AWS_ACCOUNT_ID }}
      aws-region: ${{ vars.AWS_REGION }}
    secrets: inherit

```

______________________________________________________________________

## File: .github/workflows/pr-vllm-sagemaker-amzn2023.yml

```yaml
name: PR - vLLM SageMaker AMZN2023

on:
  pull_request:
    branches: [main]
    types: [opened, reopened, synchronize]
    paths:
      - "docker/vllm/Dockerfile.amzn2023"
      - "scripts/vllm/amzn2023/**"
      - "scripts/vllm/dockerd_entrypoint.sh"
      - "scripts/vllm/sagemaker_entrypoint.sh"
      - "scripts/common/**"
      - "scripts/telemetry/**"
      - ".github/config/vllm-sagemaker-amzn2023.yml"
      - ".github/workflows/pr-vllm-sagemaker-amzn2023.yml"
      - ".github/workflows/reusable-vllm-upstream-tests.yml"
      - ".github/workflows/reusable-vllm-sagemaker-tests.yml"
      - "test/sanity/**"
      - "test/telemetry/**"
      - "test/vllm/sagemaker/**"

permissions:
  contents: read
  pull-requests: read

env:
  FORCE_COLOR: "1"
  CONFIG_FILE: ".github/config/vllm-sagemaker-amzn2023.yml"

jobs:
  gatekeeper:
    runs-on: ubuntu-latest
    concurrency:
      group: ${{ github.workflow }}-gate-${{ github.event.pull_request.number }}
      cancel-in-progress: true
    steps:
      - name: Checkout base branch (safe)
        uses: actions/checkout@v5
        with:
          ref: ${{ github.event.pull_request.base.sha }}
          fetch-depth: 1

      - name: Run permission gate (from base)
        uses: ./.github/actions/pr-permission-gate

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
      - name: Checkout code
        uses: actions/checkout@v5

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
      - name: Checkout DLC source
        uses: actions/checkout@v5

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
              - "docker/vllm/Dockerfile.amzn2023"
              - "scripts/vllm/amzn2023/**"
              - "scripts/vllm/dockerd_entrypoint.sh"
              - "scripts/vllm/sagemaker_entrypoint.sh"
              - "scripts/common/**"
              - "scripts/telemetry/**"
              - ".github/config/vllm-sagemaker-amzn2023.yml"
            sanity-test-change:
              - "test/sanity/**"
            telemetry-test-change:
              - "test/telemetry/**"
            sagemaker-test-change:
              - "test/vllm/sagemaker/**"

  build-image:
    needs: [check-changes, load-config]
    if: needs.check-changes.outputs.build-change == 'true'
    runs-on:
      - codebuild-runner-${{ github.run_id }}-${{ github.run_attempt }}
        fleet:x86-vllm-build-runner
        buildspec-override:true
    timeout-minutes: 720
    concurrency:
      group: ${{ github.workflow }}-build-image-${{ github.event.pull_request.number }}
      cancel-in-progress: true
    outputs:
      ci-image: ${{ steps.build.outputs.image-uri }}
    steps:
      - name: Checkout code
        uses: actions/checkout@v5

      - name: Build image
        id: build
        uses: ./.github/actions/build-image
        with:
          framework: ${{ needs.load-config.outputs.framework }}
          target: vllm-sagemaker-amzn2023
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

  upstream-tests:
    needs: [build-image, load-config]
    if: success()
    concurrency:
      group: ${{ github.workflow }}-upstream-tests-${{ github.event.pull_request.number }}
      cancel-in-progress: true
    uses: ./.github/workflows/reusable-vllm-upstream-tests.yml
    with:
      image-uri: ${{ needs.build-image.result == 'success' && needs.build-image.outputs.ci-image || format('{0}.dkr.ecr.{1}.amazonaws.com/{2}', vars.PROD_AWS_ACCOUNT_ID, vars.AWS_REGION, needs.load-config.outputs.prod-image) }}
      aws-account-id: ${{ needs.build-image.result == 'success' && vars.CI_AWS_ACCOUNT_ID || vars.PROD_AWS_ACCOUNT_ID }}
      aws-region: ${{ vars.AWS_REGION }}
      framework-version: ${{ needs.load-config.outputs.framework-version }}
      setup-script: scripts/vllm/amzn2023/vllm_test_setup.sh
      example-test-script: scripts/vllm/amzn2023/vllm_sagemaker_examples_test.sh
    secrets: inherit

  endpoint-test:
    needs: [check-changes, build-image, load-config]
    if: |
      always() && !failure() && !cancelled() &&
      (needs.check-changes.outputs.build-change == 'true' || needs.check-changes.outputs.sagemaker-test-change == 'true')
    concurrency:
      group: ${{ github.workflow }}-endpoint-test-${{ github.event.pull_request.number }}
      cancel-in-progress: false
    uses: ./.github/workflows/reusable-vllm-sagemaker-tests.yml
    with:
      image-uri: ${{ needs.build-image.result == 'success' && needs.build-image.outputs.ci-image || format('{0}.dkr.ecr.{1}.amazonaws.com/{2}', vars.PROD_AWS_ACCOUNT_ID, vars.AWS_REGION, needs.load-config.outputs.prod-image) }}

```

______________________________________________________________________

## File: .github/workflows/reusable-vllm-model-tests.yml

```yaml
name: Reusable vLLM Model Smoke Tests

permissions:
  contents: read

on:
  workflow_call:
    inputs:
      image-uri:
        description: "Image URI to test"
        required: true
        type: string
      aws-account-id:
        description: "AWS account ID for ECR authentication"
        required: true
        type: string
      aws-region:
        description: "AWS region for ECR authentication"
        required: true
        type: string

jobs:
  load-models:
    runs-on: ubuntu-latest
    outputs:
      codebuild-fleet-matrix: ${{ steps.parse.outputs.codebuild-fleet }}
      runner-scale-sets-matrix: ${{ steps.parse.outputs.runner-scale-sets }}
    steps:
      - name: Checkout code
        uses: actions/checkout@v5

      - name: Parse model config
        id: parse
        run: |
          python3 -c "
          import yaml, json
          with open('.github/config/vllm-model-tests.yml') as f:
              cfg = yaml.safe_load(f)
          print('codebuild-fleet=' + json.dumps(cfg.get('codebuild-fleet', [])))
          print('runner-scale-sets=' + json.dumps(cfg.get('runner-scale-sets', [])))
          " > parsed.txt
          grep '^codebuild-fleet=' parsed.txt | sed 's/^codebuild-fleet=//' > cb.json
          grep '^runner-scale-sets=' parsed.txt | sed 's/^runner-scale-sets=//' > rss.json
          echo "codebuild-fleet=$(cat cb.json)" >> $GITHUB_OUTPUT
          echo "runner-scale-sets=$(cat rss.json)" >> $GITHUB_OUTPUT

  test-model-codebuild-fleet:
    name: test-model (${{ matrix.name }})
    if: ${{ fromJson(needs.load-models.outputs.codebuild-fleet-matrix)[0] != null }}
    needs: [load-models]
    strategy:
      fail-fast: false
      matrix:
        include: ${{ fromJson(needs.load-models.outputs.codebuild-fleet-matrix) }}
    runs-on:
      - codebuild-runner-${{ github.run_id }}-${{ github.run_attempt }}
        fleet:${{ matrix.fleet }}
        buildspec-override:true
    steps:
      - name: Checkout code
        uses: actions/checkout@v5

      - name: ECR authenticate
        uses: ./.github/actions/ecr-authenticate
        with:
          aws-account-id: ${{ inputs.aws-account-id }}
          aws-region: ${{ inputs.aws-region }}

      - name: Download model from S3
        run: |
          MODEL_DIR="/dlc-models/${{ matrix.name }}"
          mkdir -p "${MODEL_DIR}"
          aws s3 cp "${{ matrix.s3_path }}" "/dlc-models/${{ matrix.name }}.tar.gz"
          tar xzf "/dlc-models/${{ matrix.name }}.tar.gz" -C "${MODEL_DIR}"
          rm -f "/dlc-models/${{ matrix.name }}.tar.gz"
          SUBDIRS=("${MODEL_DIR}"/*)
          if [ ${#SUBDIRS[@]} -eq 1 ] && [ -d "${SUBDIRS[0]}" ]; then
            mv "${SUBDIRS[0]}"/* "${MODEL_DIR}"/
            rmdir "${SUBDIRS[0]}"
          fi

      - name: Start container
        run: |
          docker pull ${{ inputs.image-uri }}
          CONTAINER_ID=$(docker run -d -it --gpus all --entrypoint /bin/bash \
            -v /dlc-models:/models \
            ${{ inputs.image-uri }})
          echo "CONTAINER_ID=$CONTAINER_ID" >> $GITHUB_ENV

      - name: Copy scripts into container
        run: |
          docker cp scripts/vllm/vllm_model_smoke_test.sh ${CONTAINER_ID}:/models/

      - name: Run model smoke test
        run: |
          docker exec ${CONTAINER_ID} bash /models/vllm_model_smoke_test.sh \
            "/models/${{ matrix.name }}" \
            "${{ matrix.name }}" \
            ${{ matrix.extra_args }}

      - name: Cleanup
        if: always()
        run: |
          docker stop ${CONTAINER_ID} 2>/dev/null || true
          docker rm -f ${CONTAINER_ID} 2>/dev/null || true
          docker rmi ${{ inputs.image-uri }} 2>/dev/null || true
          rm -rf /dlc-models

  test-model-runner-scale-sets:
    name: test-model (${{ matrix.name }})
    if: ${{ fromJson(needs.load-models.outputs.runner-scale-sets-matrix)[0] != null }}
    needs: [load-models]
    strategy:
      fail-fast: false
      matrix:
        include: ${{ fromJson(needs.load-models.outputs.runner-scale-sets-matrix) }}
    runs-on: gpu-efa-runners
    steps:
      - name: Checkout code
        uses: actions/checkout@v5

      - name: ECR authenticate
        uses: ./.github/actions/ecr-authenticate
        with:
          aws-account-id: ${{ inputs.aws-account-id }}
          aws-region: ${{ inputs.aws-region }}

      - name: Download model from S3
        run: |
          MODEL_DIR="/dlc-models/${{ matrix.name }}"
          mkdir -p "${MODEL_DIR}"
          aws s3 cp "${{ matrix.s3_path }}" "/dlc-models/${{ matrix.name }}.tar.gz"
          tar xzf "/dlc-models/${{ matrix.name }}.tar.gz" -C "${MODEL_DIR}"
          rm -f "/dlc-models/${{ matrix.name }}.tar.gz"
          SUBDIRS=("${MODEL_DIR}"/*)
          if [ ${#SUBDIRS[@]} -eq 1 ] && [ -d "${SUBDIRS[0]}" ]; then
            mv "${SUBDIRS[0]}"/* "${MODEL_DIR}"/
            rmdir "${SUBDIRS[0]}"
          fi

      - name: Start container
        run: |
          docker pull ${{ inputs.image-uri }}
          CONTAINER_ID=$(docker run -d -it --gpus all --entrypoint /bin/bash \
            --ipc=host --shm-size=10g \
            ${{ inputs.image-uri }})
          echo "CONTAINER_ID=$CONTAINER_ID" >> $GITHUB_ENV

      # on runner scales sets, container run inside container cannot mount
      # we need to copy models to the container
      - name: Copy files into container
        run: |
          docker exec ${CONTAINER_ID} mkdir -p /models
          docker cp /dlc-models/${{ matrix.name }} ${CONTAINER_ID}:/models/${{ matrix.name }}
          docker cp scripts/vllm/vllm_model_smoke_test.sh ${CONTAINER_ID}:/models/
          rm -rf /dlc-models

      - name: Run model smoke test
        run: |
          docker exec ${CONTAINER_ID} bash /models/vllm_model_smoke_test.sh \
            "/models/${{ matrix.name }}" \
            "${{ matrix.name }}" \
            ${{ matrix.extra_args }}

      - name: Cleanup
        if: always()
        run: |
          docker stop ${CONTAINER_ID} 2>/dev/null || true
          docker rm -f ${CONTAINER_ID} 2>/dev/null || true
          docker rmi ${{ inputs.image-uri }} 2>/dev/null || true
          rm -rf /dlc-models

```

______________________________________________________________________

## File: .github/workflows/reusable-vllm-upstream-tests.yml

```yaml
name: Reusable vLLM Upstream Tests

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
        description: 'vLLM framework version (e.g., 0.15.1)'
        required: true
        type: string
      setup-script:
        description: 'Path to test setup script (e.g., scripts/vllm/vllm_test_setup.sh)'
        required: true
        type: string
      example-test-script:
        description: 'Path to example test script (e.g., scripts/vllm/vllm_ec2_examples_test.sh)'
        required: true
        type: string

jobs:
  regression-test:
    runs-on:
      - codebuild-runner-${{ github.run_id }}-${{ github.run_attempt }}
        fleet:x86-g6xl-runner
        buildspec-override:true
    steps:
      - name: Checkout DLC source
        uses: actions/checkout@v5

      - name: Container pull
        uses: ./.github/actions/ecr-authenticate
        with:
          aws-account-id: ${{ inputs.aws-account-id }}
          aws-region: ${{ inputs.aws-region }}
          image-uri: ${{ inputs.image-uri }}

      - name: Checkout vLLM tests
        uses: actions/checkout@v5
        with:
          repository: vllm-project/vllm
          ref: v${{ inputs.framework-version }}
          path: vllm_source

      - name: Start container
        run: |
          CONTAINER_ID=$(docker run -d -it --rm --gpus=all --entrypoint /bin/bash \
            -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
            -v ${HOME}/.cache/vllm:/root/.cache/vllm \
            -v .:/workdir --workdir /workdir \
            -e HF_TOKEN=${{ secrets.HUGGING_FACE_HUB_TOKEN }} \
            ${{ inputs.image-uri }})
          echo "CONTAINER_ID=$CONTAINER_ID" >> $GITHUB_ENV

      - name: Setup for vLLM tests
        run: |
          docker exec ${CONTAINER_ID} ${{ inputs.setup-script }}

      - name: Run vLLM regression tests
        run: |
          docker exec ${CONTAINER_ID} scripts/vllm/vllm_regression_test.sh

  cuda-test:
    runs-on:
      - codebuild-runner-${{ github.run_id }}-${{ github.run_attempt }}
        fleet:x86-g6xl-runner
        buildspec-override:true
    steps:
      - name: Checkout DLC source
        uses: actions/checkout@v5

      - name: Container pull
        uses: ./.github/actions/ecr-authenticate
        with:
          aws-account-id: ${{ inputs.aws-account-id }}
          aws-region: ${{ inputs.aws-region }}
          image-uri: ${{ inputs.image-uri }}

      - name: Checkout vLLM tests
        uses: actions/checkout@v5
        with:
          repository: vllm-project/vllm
          ref: v${{ inputs.framework-version }}
          path: vllm_source

      - name: Start container
        run: |
          CONTAINER_ID=$(docker run -d -it --rm --gpus=all --entrypoint /bin/bash \
            -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
            -v ${HOME}/.cache/vllm:/root/.cache/vllm \
            -v .:/workdir --workdir /workdir \
            -e HF_TOKEN=${{ secrets.HUGGING_FACE_HUB_TOKEN }} \
            ${{ inputs.image-uri }})
          echo "CONTAINER_ID=$CONTAINER_ID" >> $GITHUB_ENV

      - name: Setup for vLLM tests
        run: |
          docker exec ${CONTAINER_ID} ${{ inputs.setup-script }}

      - name: Run vLLM CUDA tests
        run: |
          docker exec ${CONTAINER_ID} scripts/vllm/vllm_cuda_test.sh

  example-test:
    runs-on:
      - codebuild-runner-${{ github.run_id }}-${{ github.run_attempt }}
        fleet:x86-g6xl-runner
        buildspec-override:true
    steps:
      - name: Checkout DLC source
        uses: actions/checkout@v5

      - name: Container pull
        uses: ./.github/actions/ecr-authenticate
        with:
          aws-account-id: ${{ inputs.aws-account-id }}
          aws-region: ${{ inputs.aws-region }}
          image-uri: ${{ inputs.image-uri }}

      - name: Checkout vLLM tests
        uses: actions/checkout@v5
        with:
          repository: vllm-project/vllm
          ref: v${{ inputs.framework-version }}
          path: vllm_source

      - name: Start container
        run: |
          CONTAINER_ID=$(docker run -d -it --rm --gpus=all --entrypoint /bin/bash \
            -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
            -v ${HOME}/.cache/vllm:/root/.cache/vllm \
            -v .:/workdir --workdir /workdir \
            -e HF_TOKEN=${{ secrets.HUGGING_FACE_HUB_TOKEN }} \
            ${{ inputs.image-uri }})
          echo "CONTAINER_ID=$CONTAINER_ID" >> $GITHUB_ENV

      - name: Setup for vLLM tests
        run: |
          docker exec ${CONTAINER_ID} ${{ inputs.setup-script }}

      - name: Run vLLM example tests
        run: |
          docker exec ${CONTAINER_ID} ${{ inputs.example-test-script }}

```

______________________________________________________________________

## File: .github/config/vllm-model-tests.yml

```yaml
# vLLM Model Smoke Test Configuration
# Shared across all vLLM PR workflows
#
# codebuild-fleet: Runs on CodeBuild fleet runners (single GPU instances)
# runner-scale-sets: Runs on self-hosted runner scale sets (multi-GPU instances)
#
# extra_args: passed directly to vllm serve (includes --tensor-parallel-size)

codebuild-fleet:
  - name: "gpt-oss-20b"
    s3_path: "s3://dlc-cicd-models/vllm_models/gpt-oss-20b.tar.gz"
    fleet: "x86-g6exl-runner"
    extra_args: "--tensor-parallel-size 1 --max-model-len 4096 --dtype bfloat16"

  - name: "llama-3.3-70b"
    s3_path: "s3://dlc-cicd-models/vllm_models/llama-3.3-70b.tar.gz"
    fleet: "x86-g6e12xl-runner"
    extra_args: "--tensor-parallel-size 4 --max-model-len 4096"

runner-scale-sets:
  - name: "qwen3-32b"
    s3_path: "s3://dlc-cicd-models/vllm_models/qwen3-32b.tar.gz"
    extra_args: "--tensor-parallel-size 4 --max-model-len 8192"

# upstream
# facebook/opt-125m
# meta-llama/Llama-3.2-1B-Instruct
# Qwen/Qwen3-0.6B
# fixie-ai/ultravox-v0_5-llama-3_2-1b
# llava-hf/llava-1.5-7b-hf
# microsoft/Phi-3.5-vision-instruct
# openai/whisper-large-v3-turbo
# jason9693/Qwen2.5-1.5B-apeach
# intfloat/e5-small
# BAAI/bge-reranker-v2-m3
# meta-llama/Llama-3.1-8B-Instruct

```

______________________________________________________________________

## File: scripts/vllm/amzn2023/vllm_test_setup.sh

```bash
#!/bin/bash
set -eux

# Use --system when not in a virtualenv (Ubuntu image), omit when venv is active (AL2023)
UV_FLAGS=""
if [ -z "${VIRTUAL_ENV:-}" ]; then
  UV_FLAGS="--system"
fi

# delete old test dependencies file and regen
rm vllm_source/requirements/test.txt
uv pip compile vllm_source/requirements/test.in -o vllm_source/requirements/test.txt --index-strategy unsafe-best-match --torch-backend cu129 --python-platform x86_64-manylinux_2_28 --python-version 3.12
# uv pip install $UV_FLAGS -r vllm_source/requirements/common.txt --torch-backend=auto
uv pip install $UV_FLAGS -r vllm_source/requirements/dev.txt
uv pip install $UV_FLAGS pytest pytest-asyncio
uv pip install $UV_FLAGS -e vllm_source/tests/vllm_test_utils
uv pip install $UV_FLAGS hf_transfer
cd vllm_source
mkdir src
mv vllm src/vllm
```

______________________________________________________________________

## File: scripts/vllm/amzn2023/vllm_ec2_examples_test.sh

```bash
#!/bin/bash
set -eux

nvidia-smi
cd vllm_source/examples
pip install tensorizer # for tensorizer test
python3 offline_inference/basic/generate.py --model facebook/opt-125m
# python3 offline_inference/basic/generate.py --model meta-llama/Llama-2-13b-chat-hf --cpu-offload-gb 10
python3 offline_inference/basic/chat.py
python3 offline_inference/prefix_caching.py
python3 offline_inference/llm_engine_example.py
python3 offline_inference/audio_language.py --seed 0
python3 offline_inference/vision_language.py --seed 0
python3 offline_inference/vision_language_multi_image.py --seed 0
python3 others/tensorize_vllm_model.py --model facebook/opt-125m serialize --serialized-directory /tmp/ --suffix v1 && python3 others/tensorize_vllm_model.py --model facebook/opt-125m deserialize --path-to-tensors /tmp/vllm/facebook/opt-125m/v1/model.tensors
python3 offline_inference/encoder_decoder_multimodal.py --model-type whisper --seed 0
python3 offline_inference/basic/classify.py
python3 offline_inference/basic/embed.py
python3 offline_inference/basic/score.py
python3 offline_inference/spec_decode.py --test --method eagle --num_spec_tokens 3 --dataset-name hf --dataset-path philschmid/mt-bench --num-prompts 80 --temp 0 --top-p 1.0 --top-k -1 --tp 1 --enable-chunked-prefill --max-model-len 2048
# https://github.com/vllm-project/vllm/pull/26682 uses slightly more memory in PyTorch 2.9+ causing this test to OOM in 1xL4 GPU
python3 offline_inference/spec_decode.py --test --method eagle3 --num_spec_tokens 3 --dataset-name hf --dataset-path philschmid/mt-bench --num-prompts 80 --temp 0 --top-p 1.0 --top-k -1 --tp 1 --enable-chunked-prefill --max-model-len 1536

```

______________________________________________________________________

## File: scripts/vllm/amzn2023/vllm_sagemaker_examples_test.sh

```bash
#!/bin/bash
set -eux
nvidia-smi

cd vllm_source

# Test LoRA adapter loading/unloading via SageMaker endpoints
pytest tests/entrypoints/sagemaker/test_sagemaker_lora_adapters.py -v

# Test stateful session management
pytest tests/entrypoints/sagemaker/test_sagemaker_stateful_sessions.py -v

# Test sagemaker custom middleware
pytest tests/entrypoints/sagemaker/test_sagemaker_middleware_integration.py -v

# Test sagemaker endpoint overrides
pytest tests/entrypoints/sagemaker/test_sagemaker_handler_overrides.py -v

# Test LoRA adapter loading/unloading via original OpenAI API server endpoints
pytest tests/entrypoints/openai/test_lora_adapters.py -v

cd examples
pip install tensorizer # for tensorizer test
python3 offline_inference/basic/generate.py --model facebook/opt-125m
# python3 offline_inference/basic/generate.py --model meta-llama/Llama-2-13b-chat-hf --cpu-offload-gb 10
python3 offline_inference/basic/chat.py
python3 offline_inference/prefix_caching.py
python3 offline_inference/llm_engine_example.py
python3 offline_inference/audio_language.py --seed 0
python3 offline_inference/vision_language.py --seed 0
python3 offline_inference/vision_language_multi_image.py --seed 0
python3 others/tensorize_vllm_model.py --model facebook/opt-125m serialize --serialized-directory /tmp/ --suffix v1 && python3 others/tensorize_vllm_model.py --model facebook/opt-125m deserialize --path-to-tensors /tmp/vllm/facebook/opt-125m/v1/model.tensors
python3 offline_inference/encoder_decoder_multimodal.py --model-type whisper --seed 0
python3 offline_inference/basic/classify.py
python3 offline_inference/basic/embed.py
python3 offline_inference/basic/score.py
python3 offline_inference/spec_decode.py --test --method eagle --num_spec_tokens 3 --dataset-name hf --dataset-path philschmid/mt-bench --num-prompts 80 --temp 0 --top-p 1.0 --top-k -1 --tp 1 --enable-chunked-prefill --max-model-len 2048
# https://github.com/vllm-project/vllm/pull/26682 uses slightly more memory in PyTorch 2.9+ causing this test to OOM in 1xL4 GPU
python3 offline_inference/spec_decode.py --test --method eagle3 --num_spec_tokens 3 --dataset-name hf --dataset-path philschmid/mt-bench --num-prompts 80 --temp 0 --top-p 1.0 --top-k -1 --tp 1 --enable-chunked-prefill --max-model-len 1536

```

______________________________________________________________________

## File: scripts/vllm/vllm_model_smoke_test.sh

```bash
#!/bin/bash
set -euo pipefail

# vLLM Model Smoke Test
# Usage: vllm_model_smoke_test.sh <model_dir> <model_name> [extra_args...]

MODEL_DIR="${1:?Usage: $0 <model_dir> <model_name> [extra_args...]}"
MODEL_NAME="${2:?Usage: $0 <model_dir> <model_name> [extra_args...]}"
shift 2
EXTRA_ARGS="$*"

VLLM_PORT=8000
HEALTH_TIMEOUT=600
HEALTH_INTERVAL=10

echo "=== Model directory: ${MODEL_DIR} ==="
ls -la "${MODEL_DIR}"

echo "=== Starting vLLM server ==="
# shellcheck disable=SC2086
vllm serve "${MODEL_DIR}" \
  --port "${VLLM_PORT}" \
  ${EXTRA_ARGS} &
VLLM_PID=$!

cleanup() {
  echo "=== Stopping vLLM server ==="
  kill "${VLLM_PID}" 2>/dev/null || true
  wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Waiting for health check ==="
elapsed=0
while [ "${elapsed}" -lt "${HEALTH_TIMEOUT}" ]; do
  if curl -sf http://localhost:${VLLM_PORT}/health >/dev/null 2>&1; then
    echo "Server healthy after ${elapsed}s"
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "ERROR: vLLM process died"
    exit 1
  fi
  sleep "${HEALTH_INTERVAL}"
  elapsed=$((elapsed + HEALTH_INTERVAL))
done

if [ "${elapsed}" -ge "${HEALTH_TIMEOUT}" ]; then
  echo "ERROR: Health check timed out after ${HEALTH_TIMEOUT}s"
  exit 1
fi

echo "=== Running completion test ==="
RESPONSE=$(curl -sf http://localhost:${VLLM_PORT}/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${MODEL_DIR}\", \"prompt\": \"Hello\", \"max_tokens\": 16}")

echo "Response: ${RESPONSE}"

if echo "${RESPONSE}" | python3 -c "import sys,json; c=json.load(sys.stdin)['choices']; assert len(c)>0 and len(c[0]['text'].strip())>0"; then
  echo "=== PASSED: ${MODEL_NAME} ==="
else
  echo "=== FAILED: ${MODEL_NAME} - empty or invalid response ==="
  exit 1
fi

```
