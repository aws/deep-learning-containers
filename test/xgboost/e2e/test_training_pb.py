"""Training tests with recordio-protobuf content type.

Migrated from SMFrameworksXGBoost3_0-5Tests/src/integration_tests/test_training_pb.py
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


class TestTrainingProtobuf:
    def test_single_instance(self, image_uri, role):
        _, duration, desc = run_training_job(
            image_uri=image_uri, role=role, hyperparameters=BASE_HP,
            train_s3_key="recordio-protobuf/train", validation_s3_key="recordio-protobuf/test",
            content_type="application/x-recordio-protobuf", test_name="pb-single",
        )
        assert desc["TrainingJobStatus"] == "Completed"
        assert 1 <= duration <= 1800

    def test_distributed(self, image_uri, role):
        hp = {**BASE_HP, "tree_method": "hist"}
        _, _, desc = run_training_job(
            image_uri=image_uri, role=role, hyperparameters=hp,
            train_s3_key="recordio-protobuf/train", validation_s3_key="recordio-protobuf/test",
            content_type="application/x-recordio-protobuf", test_name="pb-dist",
            instance_count=2,
        )
        assert desc["TrainingJobStatus"] == "Completed"

    def test_pipe_mode_single_instance(self, image_uri, role):
        _, _, desc = run_training_job(
            image_uri=image_uri, role=role, hyperparameters=BASE_HP,
            train_s3_key="recordio-protobuf/train", validation_s3_key="recordio-protobuf/test",
            content_type="application/x-recordio-protobuf", test_name="pb-pipe",
            input_mode="Pipe",
        )
        assert desc["TrainingJobStatus"] == "Completed"

    def test_pipe_mode_distributed(self, image_uri, role):
        hp = {**BASE_HP, "tree_method": "hist"}
        _, _, desc = run_training_job(
            image_uri=image_uri, role=role, hyperparameters=hp,
            train_s3_key="recordio-protobuf/train", validation_s3_key="recordio-protobuf/test",
            content_type="application/x-recordio-protobuf", test_name="pb-pipe-dist",
            input_mode="Pipe", instance_count=2,
        )
        assert desc["TrainingJobStatus"] == "Completed"

    def test_sparse_single_instance(self, image_uri, role):
        _, _, desc = run_training_job(
            image_uri=image_uri, role=role, hyperparameters=BASE_HP,
            train_s3_key="recordio-protobuf/sparse/train", validation_s3_key="recordio-protobuf/sparse/test",
            content_type="application/x-recordio-protobuf", test_name="pb-sparse",
        )
        assert desc["TrainingJobStatus"] == "Completed"
