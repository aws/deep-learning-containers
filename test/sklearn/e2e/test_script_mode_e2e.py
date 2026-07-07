"""Script mode training with `requirements.txt` — verifies the container
installs extra Python dependencies from a user-provided requirements.txt
alongside the training script.
"""

from .conftest import data_uri, run_training_job


CODE_TARBALL_KEY = "code/requirements.tar.gz"


class TestScriptModeE2E:
    def test_requirements_install(self, image_uri, role):
        hp = {
            "sagemaker_program": "train.py",
            "sagemaker_submit_directory": data_uri(CODE_TARBALL_KEY),
        }
        _, _, desc = run_training_job(
            image_uri=image_uri,
            role=role,
            hyperparameters=hp,
            train_s3_key="data/train.parquet",
            test_name="script-reqs",
            volume_size=20,
        )
        assert desc["TrainingJobStatus"] == "Completed"
