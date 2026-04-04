"""Training tests with libsvm content type.

Migrated from SMFrameworksXGBoost3_0-5Tests/src/integration_tests/test_training_libsvm.py
"""

from .conftest import E2E_TEST_BUCKET, run_training_job, s3_uri

BASE_HP = {
    "max_depth": "5",
    "eta": "0.2",
    "gamma": "4",
    "min_child_weight": "6",
    "subsample": "0.7",
    "verbosity": "3",
    "objective": "reg:squarederror",
    "num_round": "10",
}


def _checkpoint_uri(name):
    return s3_uri(E2E_TEST_BUCKET, f"e2e-output/checkpoints/{name}")


class TestTrainingLibsvm:
    def test_single_instance(self, image_uri, role):
        _, duration, desc = run_training_job(
            image_uri=image_uri,
            role=role,
            hyperparameters=BASE_HP,
            train_s3_key="train",
            validation_s3_key="test",
            content_type="text/libsvm",
            test_name="libsvm-single",
        )
        assert desc["TrainingJobStatus"] == "Completed"
        assert 1 <= duration <= 1800

    def test_distributed(self, image_uri, role):
        hp = {**BASE_HP, "tree_method": "hist"}
        _, _, desc = run_training_job(
            image_uri=image_uri,
            role=role,
            hyperparameters=hp,
            train_s3_key="train",
            validation_s3_key="test",
            content_type="text/libsvm",
            test_name="libsvm-dist",
            instance_count=2,
        )
        assert desc["TrainingJobStatus"] == "Completed"

    def test_checkpoint_single_instance(self, image_uri, role):
        _, _, desc = run_training_job(
            image_uri=image_uri,
            role=role,
            hyperparameters=BASE_HP,
            train_s3_key="train",
            validation_s3_key="test",
            content_type="text/libsvm",
            test_name="libsvm-ckpt",
            checkpoint_s3_uri=_checkpoint_uri("libsvm-ckpt"),
        )
        assert desc["TrainingJobStatus"] == "Completed"

    def test_checkpoint_distributed(self, image_uri, role):
        hp = {**BASE_HP, "tree_method": "hist"}
        _, _, desc = run_training_job(
            image_uri=image_uri,
            role=role,
            hyperparameters=hp,
            train_s3_key="train",
            validation_s3_key="test",
            content_type="text/libsvm",
            test_name="libsvm-ckpt-d",
            instance_count=2,
            checkpoint_s3_uri=_checkpoint_uri("libsvm-ckpt-dist"),
        )
        assert desc["TrainingJobStatus"] == "Completed"

    def test_gpu_single_instance(self, image_uri, role):
        hp = {**BASE_HP, "tree_method": "gpu_hist"}
        _, _, desc = run_training_job(
            image_uri=image_uri,
            role=role,
            hyperparameters=hp,
            train_s3_key="train",
            validation_s3_key="test",
            content_type="text/libsvm",
            test_name="libsvm-gpu",
            instance_type="ml.g4dn.2xlarge",
        )
        assert desc["TrainingJobStatus"] == "Completed"

    def test_gpu_checkpoint(self, image_uri, role):
        hp = {**BASE_HP, "tree_method": "gpu_hist"}
        _, _, desc = run_training_job(
            image_uri=image_uri,
            role=role,
            hyperparameters=hp,
            train_s3_key="train",
            validation_s3_key="test",
            content_type="text/libsvm",
            test_name="libsvm-gpuck",
            instance_type="ml.g4dn.2xlarge",
            checkpoint_s3_uri=_checkpoint_uri("libsvm-gpu-ckpt"),
        )
        assert desc["TrainingJobStatus"] == "Completed"
