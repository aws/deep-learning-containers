import unittest.mock as mock
import pytest
from tensorflow_serving.config import model_server_config_pb2

from container.sagemaker import proxy_client


@pytest.fixture(autouse=True)
def create_sagemaker_folder(tmpdir):
    tmpdir.join('sagemaker').ensure(dir=True)

    proxy_client.MODEL_CONFIG_FILE = str(tmpdir) + proxy_client.MODEL_CONFIG_FILE
    proxy_client.DEFAULT_LOCK_FILE = str(tmpdir) + proxy_client.DEFAULT_LOCK_FILE


def test_grpc_add_model_no_config_file():
    client = proxy_client.GRPCProxyClient(port='9090')

    with pytest.raises(FileNotFoundError) as e:
        assert client.add_model('my-model', '/opt/ml/model_path')
    assert 'No such file or directory' in str(e.value)


@mock.patch('tensorflow_serving.apis.model_management_pb2.ReloadConfigRequest')
@mock.patch('grpc.insecure_channel')
def test_grpc_add_model_call(channel, ReloadConfigRequest):
    config = 'model_config_list: {\n}\n'
    with open(proxy_client.MODEL_CONFIG_FILE, 'w') as f:
        f.write(config)

    client = proxy_client.GRPCProxyClient(port='9090')
    client.add_model('my-model', '/opt/ml/model_path')

    calls = [mock.call('0.0.0.0:9090'),
             mock.call().unary_unary('/tensorflow.serving.ModelService/GetModelStatus',
                                     request_serializer=mock.ANY, response_deserializer=mock.ANY),
             mock.call().unary_unary('/tensorflow.serving.ModelService/HandleReloadConfigRequest',
                                     request_serializer=mock.ANY, response_deserializer=mock.ANY),
             mock.call().unary_unary()(ReloadConfigRequest())
             ]

    channel.assert_has_calls(calls)

    config_list = model_server_config_pb2.ModelConfigList()
    new_model_config = config_list.config.add()
    new_model_config.name = 'my-model'
    new_model_config.base_path = '/opt/ml/model_path'
    new_model_config.model_platform = 'tensorflow'

    model_server_config = model_server_config_pb2.ModelServerConfig()
    model_server_config.model_config_list.MergeFrom(config_list)

    ReloadConfigRequest().config.CopyFrom.assert_called_with(model_server_config)

    expected = 'model_config_list: {\n'
    expected += '  config: {\n'
    expected += '    name: "my-model",\n'
    expected += '    base_path: "/opt/ml/model_path",\n'
    expected += '    model_platform: "tensorflow"\n'
    expected += '  }\n'
    expected += '}\n'

    with open(proxy_client.MODEL_CONFIG_FILE, 'r') as file:
        assert file.read() == expected


@mock.patch('tensorflow_serving.apis.model_management_pb2.ReloadConfigRequest')
@mock.patch('grpc.insecure_channel')
def test_grpc_delete_model_call(channel, ReloadConfigRequest):
    config = 'model_config_list: {\n'
    config += '  config: {\n'
    config += '    name: "my-model",\n'
    config += '    base_path: "/opt/ml/model_path",\n'
    config += '    model_platform: "tensorflow"\n'
    config += '  }\n'
    config += '}\n'
    with open(proxy_client.MODEL_CONFIG_FILE, 'w') as f:
        f.write(config)

    client = proxy_client.GRPCProxyClient(port='9090')
    client.delete_model('my-model', '/opt/ml/model_path')

    calls = [mock.call('0.0.0.0:9090'),
             mock.call().unary_unary('/tensorflow.serving.ModelService/GetModelStatus',
                                     request_serializer=mock.ANY, response_deserializer=mock.ANY),
             mock.call().unary_unary('/tensorflow.serving.ModelService/HandleReloadConfigRequest',
                                     request_serializer=mock.ANY, response_deserializer=mock.ANY),
             mock.call().unary_unary()(ReloadConfigRequest())
             ]

    channel.assert_has_calls(calls)

    config_list = model_server_config_pb2.ModelConfigList()
    model_server_config = model_server_config_pb2.ModelServerConfig()
    model_server_config.model_config_list.MergeFrom(config_list)

    ReloadConfigRequest().config.CopyFrom.assert_called_with(model_server_config)

    expected = 'model_config_list: {\n'
    expected += '}\n'

    with open(proxy_client.MODEL_CONFIG_FILE, 'r') as file:
        assert file.read() == expected
