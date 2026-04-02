"""Batch transform tests.

Migrated from SMFrameworksXGBoost3_0-5Tests/src/integration_tests/test_transform.py
"""

import pytest

from .conftest import data_uri, run_batch_transform


@pytest.fixture(scope="module")
def model_data():
    return data_uri("model_1.0/models/model.tar.gz")


class TestTransform:
    def test_batch_inference_libsvm(self, image_uri, role, model_data):
        desc = run_batch_transform(
            image_uri=image_uri,
            role=role,
            model_data=model_data,
            input_s3_uri=data_uri("testdata/abalone_test.libsvm"),
            content_type="text/libsvm",
            split_type="Line",
            accept="text/csv",
        )
        assert desc["TransformJobStatus"] == "Completed"
