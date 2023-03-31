import os
import pytest
import logging
from . import utils
from sagemaker.tensorflow import TensorFlowModel

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

@pytest.mark.sagemaker_canary
def test_endpoint(image_uri, sagemaker_region, instance_type, framework_version, py_version, model_dir, model_script):

    prefix = 'sagemaker-tensorflow-serving/models'
    sagemaker_session = utils.get_sagemaker_session(sagemaker_region)
    endpoint_name = utils.create_endpoint_name("sagemaker-tensorflow-serving")
    instance_type = instance_type or "ml.p2.xlarge"
    try:
        model_data = sagemaker_session.upload_data(path=model_dir, key_prefix=prefix)
        model = TensorFlowModel(
            model_data=model_data,
            role="SageMakerRole",
            entry_point=model_script,
            framework_version=framework_version,
            # py_version=py_version,
            image_uri=image_uri,
            sagemaker_session=sagemaker_session
        )
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
        )
        LOGGER.info(f"Successfully deployed endpoint: {endpoint_name}.")
    except Exception as e:
        LOGGER.error(f"Error while setting up SageMaker Endpoint:{endpoint_name}. {e}")
    finally:
        if sagemaker_session:
            sagemaker_session.delete_endpoint(endpoint_name)
            LOGGER.info(f"Deleted endpoint: {endpoint_name}.")
