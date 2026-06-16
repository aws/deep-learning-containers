"""SageMaker integration test configuration.

Tests launch real SageMaker training jobs against the CI-built image. The
GitHub Actions workflow injects:

  TEST_IMAGE_URI  — ECR URI of the image under test
  SM_ROLE_ARN     — execution role for the training job
  PYTHONPATH      — set to <repo>/test so `test_utils` is importable

The tests read these env vars directly; no fixtures needed.
"""
