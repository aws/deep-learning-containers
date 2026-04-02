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
                image_uri=image_uri, role=role, model_data=model_data,
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
                image_uri=image_uri, role=role, model_data=model_data,
            )
            predictor.content_type = "text/csv"
            predictor.accept = "text/csv"

            payload = "0.5,0.3"
            response = predictor.predict(payload)
            assert response is not None
        finally:
            if endpoint_name:
                delete_endpoint(endpoint_name)

    def test_protobuf_inference(self, image_uri, role, model_data):
        endpoint_name = None
        try:
            predictor, endpoint_name = deploy_endpoint(
                image_uri=image_uri, role=role, model_data=model_data,
            )
            predictor.content_type = "application/x-recordio-protobuf"
            predictor.accept = "text/csv"

            # Read protobuf payload from resources
            import os
            pbr_path = os.path.join(
                os.path.dirname(__file__), "resources", "testdata", "mnist_test.pbr"
            )
            if os.path.exists(pbr_path):
                with open(pbr_path, "rb") as f:
                    payload = f.read()
                response = predictor.predict(payload)
                assert response is not None
            else:
                pytest.skip("mnist_test.pbr not available in resources")
        finally:
            if endpoint_name:
                delete_endpoint(endpoint_name)
