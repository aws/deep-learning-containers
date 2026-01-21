# DLC GitHub Actions

This directory contains reusable GitHub Actions for building and releasing Deep Learning Container (DLC) images.

## Available Actions

### release-image
Complete DLC release workflow that executes three steps sequentially:
1. Publish DLC images to ECR
2. Generate release information and upload to S3
3. Publish SNS notifications and update SHA values

**Documentation:** [RELEASE_ACTION_USAGE.md](../RELEASE_ACTION_USAGE.md)

**Example:**
```yaml
- uses: ./.github/actions/release-image
  with:
    release-spec-content: |
      framework: vllm
      version: 0.13.0
      source_image_uri: <image-uri>
      target_regions:
        - us-west-2
        - us-east-1
    release-package-s3-bucket: dlc-release-logic-v2
    release-package-s3-prefix: DLContainersReleaseLogicV2/DLContainersReleaseLogicV2
    aws-region: us-west-2
    source-stage: private
    target-stage: gamma
```

### build-image
Build DLC images using Docker.

### ecr-authenticate
Authenticate to Amazon ECR.

### pr-permission-gate
Check PR permissions before allowing workflow execution.

## Documentation

- [Release Action Usage Guide](../RELEASE_ACTION_USAGE.md) - Complete guide for the release-image action
- [Release Package Overview](../RELEASE_PACKAGE_OVERVIEW.md) - Information about the DLContainersReleaseLogicV2 package
- [Runner Configuration Guide](../RUNNER_CONFIGURATION_GUIDE.md) - How to configure CodeBuild runners
- [AWS Authentication Guide](../AWS_AUTHENTICATION_GUIDE.md) - AWS authentication setup

## Example Workflows

See [../workflows/release-example.yml](../workflows/release-example.yml) for complete working examples.
