"""Verify key Python packages import successfully."""

import importlib

import pytest

PACKAGES = [
    "ray",
    "ray.train",
    "ray.tune",
    "ray.data",
    "torch",
    "torchvision",
    "pytorch_lightning",
    "transformers",
    "datasets",
    "accelerate",
    "deepspeed",
    "aiohttp_cors",  # ships only with ray[default] (dashboard + job server)
    "boto3",
    "yaml",
]


@pytest.mark.parametrize("pkg", PACKAGES)
def test_import(pkg):
    importlib.import_module(pkg)


def test_job_submission_client_importable():
    """ray[default] provides the job-submission API that `ray job submit` uses."""
    from ray.job_submission import JobSubmissionClient  # noqa: F401


def test_ray_serve_absent():
    """Training-scoped image: ray[serve] must not be installed (ray.serve deps missing)."""
    with pytest.raises(ImportError):
        import ray.serve  # noqa: F401
