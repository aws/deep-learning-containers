"""Training tests with parquet content type.

Migrated from SMFrameworksXGBoost3_0-5Tests/src/integration_tests/test_training_pq.py
"""

from .conftest import run_training_job

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


class TestTrainingParquet:
    def test_single_instance(self, image_uri, role):
        _, duration, desc = run_training_job(
            image_uri=image_uri, role=role, hyperparameters=BASE_HP,
            train_s3_key="parquet/train", validation_s3_key="parquet/test",
            content_type="application/x-parquet", test_name="pq-single",
            instance_type="ml.m5.2xlarge",
        )
        assert desc["TrainingJobStatus"] == "Completed"
        assert 1 <= duration <= 1800

    def test_distributed(self, image_uri, role):
        hp = {**BASE_HP, "tree_method": "hist"}
        _, _, desc = run_training_job(
            image_uri=image_uri, role=role, hyperparameters=hp,
            train_s3_key="parquet/train", validation_s3_key="parquet/test",
            content_type="application/x-parquet", test_name="pq-dist",
            instance_count=2,
        )
        assert desc["TrainingJobStatus"] == "Completed"

    def test_pipe_mode_single_instance(self, image_uri, role):
        _, _, desc = run_training_job(
            image_uri=image_uri, role=role, hyperparameters=BASE_HP,
            train_s3_key="parquet/train", validation_s3_key="parquet/test",
            content_type="application/x-parquet", test_name="pq-pipe",
            input_mode="Pipe",
        )
        assert desc["TrainingJobStatus"] == "Completed"

    def test_pipe_mode_distributed(self, image_uri, role):
        hp = {**BASE_HP, "tree_method": "hist"}
        _, _, desc = run_training_job(
            image_uri=image_uri, role=role, hyperparameters=hp,
            train_s3_key="parquet/train", validation_s3_key="parquet/test",
            content_type="application/x-parquet", test_name="pq-pipe-dist",
            input_mode="Pipe", instance_count=2,
        )
        assert desc["TrainingJobStatus"] == "Completed"

    def test_dask_gpu_single(self, image_uri, role):
        hp = {**BASE_HP, "tree_method": "gpu_hist", "use_dask_gpu_training": "true"}
        _, _, desc = run_training_job(
            image_uri=image_uri, role=role, hyperparameters=hp,
            train_s3_key="parquet/train", validation_s3_key="parquet/test",
            content_type="application/x-parquet", test_name="pq-dask-gpu",
            instance_type="ml.g4dn.2xlarge",
            train_distribution="FullyReplicated",
        )
        assert desc["TrainingJobStatus"] == "Completed"

    def test_dask_gpu_multi_instance(self, image_uri, role):
        hp = {**BASE_HP, "tree_method": "gpu_hist", "use_dask_gpu_training": "true"}
        _, _, desc = run_training_job(
            image_uri=image_uri, role=role, hyperparameters=hp,
            train_s3_key="parquet/train", validation_s3_key="parquet/test",
            content_type="application/x-parquet", test_name="pq-dask-2x",
            instance_type="ml.g4dn.2xlarge", instance_count=2,
            train_distribution="FullyReplicated",
        )
        assert desc["TrainingJobStatus"] == "Completed"
