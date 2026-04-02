"""Inference tests using pre-trained models.

Migrated from SMFrameworksXGBoost3_0-5Tests/src/integration_tests/test_inference.py
"""

import pytest

from .conftest import data_uri, delete_endpoint, deploy_endpoint


@pytest.fixture(scope="module")
def model_data():
    return data_uri("model_1.0/models/model.tar.gz")


class TestInference:
    def test_libsvm_inference(self, image_uri, role, model_data):
        endpoint_name = None
        try:
            predictor, endpoint_name = deploy_endpoint(
                image_uri=image_uri, role=role,
                model_data=model_data, test_name="infer-libsvm",
            )
            predictor.content_type = "text/libsvm"
            predictor.accept = "text/csv"

            payload = "3 1:0.5 2:0.3"
            response = predictor.predict(payload)
            assert response is not None
        finally:
            if endpoint_name:
                delete_endpoint(endpoint_name)

    def test_csv_inference(self, image_uri, role, model_data):
        endpoint_name = None
        try:
            predictor, endpoint_name = deploy_endpoint(
                image_uri=image_uri, role=role,
                model_data=model_data, test_name="infer-csv",
            )
            predictor.content_type = "text/csv"
            predictor.accept = "text/csv"

            payload = "0.5,0.3"
            response = predictor.predict(payload)
            assert response is not None
        finally:
            if endpoint_name:
                delete_endpoint(endpoint_name)
