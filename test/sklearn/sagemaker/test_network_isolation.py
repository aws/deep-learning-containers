"""Network isolation training test — parquet training via script mode
with `enable_network_isolation=True`. Verifies the container completes training
without needing external network access.
"""

from .conftest import data_uri, run_training_job

CODE_TARBALL_KEY = "code/pandas-parquet-file-1.4-2.tar.gz"
CODE_TARBALL_LOCAL = "/opt/ml/input/data/code/pandas-parquet-file-1.4-2.tar.gz"


class TestNetworkIsolation:
    def test_script_mode(self, image_uri, role):
        hp = {
            "sagemaker_program": "train.py",
            "sagemaker_submit_directory": CODE_TARBALL_LOCAL,
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
            extra_channels={"code": (data_uri(CODE_TARBALL_KEY), "text/plain")},
        )
        assert desc["TrainingJobStatus"] == "Completed"
