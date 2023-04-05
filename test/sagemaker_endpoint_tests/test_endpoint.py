import pytest
import logging
import utils as utils
from sagemaker.mxnet.model import MXNetModel
from sagemaker.pytorch import PyTorchModel
from sagemaker.tensorflow import TensorFlowModel

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

@pytest.mark.sagemaker_endpoint("mxnet")
def test_mxnet_endpoint(image_uri, sagemaker_region, instance_type, framework_version, py_version, mx_model_dir, mx_model_script):
    prefix = 'sagemaker-mxnet-serving/models'
    sagemaker_session = utils.get_sagemaker_session(sagemaker_region)
    endpoint_name = utils.create_endpoint_name("sagemaker-mxnet-serving")
    instance_type = instance_type or "ml.g4dn.xlarge"
    
    try:
        model_data = sagemaker_session.upload_data(path=mx_model_dir, key_prefix=prefix)
        
        model = MXNetModel(
            model_data,
            'SageMakerRole',
            mx_model_script,
            framework_version=framework_version,
            py_version=py_version,
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


@pytest.mark.sagemaker_endpoint("pytorch")
def test_pytorch_endpoint(image_uri, sagemaker_region, instance_type, framework_version, py_version, pt_model_dir, pt_model_script):
    prefix = 'sagemaker-pytorch-serving/models'
    sagemaker_session = utils.get_sagemaker_session(sagemaker_region)
    endpoint_name = utils.create_endpoint_name("sagemaker-pytorch-serving")
    instance_type = instance_type or "ml.g4dn.xlarge"
    try:
        model_data = sagemaker_session.upload_data(path=pt_model_dir, key_prefix=prefix)
        model = PyTorchModel(
            model_data=model_data,
            role="SageMakerRole",
            entry_point=pt_model_script,
            framework_version=framework_version,
            py_version=py_version,
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


@pytest.mark.sagemaker_endpoint("tensorflow")
def test_tensorflow_endpoint(image_uri, sagemaker_region, instance_type, framework_version, tf_model_dir, tf_model_script):

    prefix = 'sagemaker-tensorflow-serving/models'
    sagemaker_session = utils.get_sagemaker_session(sagemaker_region)
    endpoint_name = utils.create_endpoint_name("sagemaker-tensorflow-serving")
    instance_type = instance_type or "ml.g4dn.xlarge"
    try:
        model_data = sagemaker_session.upload_data(path=tf_model_dir, key_prefix=prefix)
        model = TensorFlowModel(
            model_data=model_data,
            role="SageMakerRole",
            entry_point=tf_model_script,
            framework_version=framework_version,
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
