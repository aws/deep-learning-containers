"""Network isolation training test — parquet training via script mode
with `enable_network_isolation=True`. Verifies the container completes training
without needing external network access.
"""

from .conftest import data_uri, run_training_job


def _tarball_slug(sklearn_version):
    """Turn '1.9.0' into '1.9-0' — the naming convention used for the S3 fixture."""
    parts = sklearn_version.split(".")
    return f"{parts[0]}.{parts[1]}-{parts[2]}"


class TestNetworkIsolation:
    def test_script_mode(self, image_uri, role, sklearn_version):
        slug = _tarball_slug(sklearn_version)
        code_tarball_key = f"code/pandas-parquet-file-{slug}.tar.gz"
        code_tarball_local = f"/opt/ml/input/data/code/pandas-parquet-file-{slug}.tar.gz"
        hp = {
            "sagemaker_program": "train.py",
            "sagemaker_submit_directory": code_tarball_local,
        }
        _, _, desc = run_training_job(
            image_uri=image_uri,
            role=role,
            hyperparameters=hp,
            train_s3_key="data/train.parquet",
            content_type="application/x-parquet",
            test_name="netiso",
            volume_size=20,
            enable_network_isolation=True,
            extra_channels={"code": (data_uri(code_tarball_key), "text/plain")},
        )
        assert desc["TrainingJobStatus"] == "Completed"
