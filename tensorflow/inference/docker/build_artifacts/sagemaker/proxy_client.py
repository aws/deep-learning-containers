# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import grpc
from google.protobuf import text_format
from tensorflow_serving.apis import model_management_pb2
from tensorflow_serving.apis import model_service_pb2_grpc
from tensorflow_serving.config import model_server_config_pb2

from multi_model_utils import lock, DEFAULT_LOCK_FILE, MODEL_CONFIG_FILE, MultiModelException

GRPC_REQUEST_TIMEOUT_IN_SECONDS = 5


class GRPCProxyClient(object):
    def __init__(self, port, host='0.0.0.0'):
        self.channel = grpc.insecure_channel('{}:{}'.format(host, port))
        self.stub = model_service_pb2_grpc.ModelServiceStub(self.channel)

    def add_model(self, model_name, base_path, model_platform='tensorflow'):
        # read model configs from existing model config file
        model_server_config = model_server_config_pb2.ModelServerConfig()
        config_list = model_server_config_pb2.ModelConfigList()

        with lock(DEFAULT_LOCK_FILE):
            try:
                config_file = self._read_model_config(MODEL_CONFIG_FILE)
                model_server_config = text_format.Parse(text=config_file,
                                                        message=model_server_config)

                new_model_config = config_list.config.add()
                new_model_config.name = model_name
                new_model_config.base_path = base_path
                new_model_config.model_platform = model_platform

                # send HandleReloadConfigRequest to tensorflow model server
                model_server_config.model_config_list.MergeFrom(config_list)
                req = model_management_pb2.ReloadConfigRequest()
                req.config.CopyFrom(model_server_config)

                self.stub.HandleReloadConfigRequest(request=req,
                                                    timeout=GRPC_REQUEST_TIMEOUT_IN_SECONDS,
                                                    wait_for_ready=True)
                self._add_model_to_config_file(model_name, base_path, model_platform)
            except grpc.RpcError as e:
                if e.code() is grpc.StatusCode.INVALID_ARGUMENT:
                    raise MultiModelException(409, e.details())
                elif e.code() is grpc.StatusCode.DEADLINE_EXCEEDED:
                    raise MultiModelException(408, e.details())
                raise MultiModelException(500, e.details())

        return 'Successfully loaded model {}'.format(model_name)

    def delete_model(self, model_name):
        model_server_config = model_server_config_pb2.ModelServerConfig()
        config_list = model_server_config_pb2.ModelConfigList()

        with lock(DEFAULT_LOCK_FILE):
            try:
                config_file = self._read_model_config(MODEL_CONFIG_FILE)
                config_list_text = config_file.strip('\n').strip('}').strip('model_config_list: {')
                config_list = text_format.Parse(text=config_list_text, message=config_list)

                for config in config_list.config:
                    if config.name == model_name:
                        config_list.config.remove(config)
                        model_server_config.model_config_list.CopyFrom(config_list)
                        req = model_management_pb2.ReloadConfigRequest()
                        req.config.CopyFrom(model_server_config)
                        self.stub.HandleReloadConfigRequest(request=req,
                                                            timeout=GRPC_REQUEST_TIMEOUT_IN_SECONDS,
                                                            wait_for_ready=True)
                        return self._delete_model_from_config_file(model_server_config)

                # no such model exists
                raise FileNotFoundError
            except grpc.RpcError as e:
                if e.code() is grpc.StatusCode.DEADLINE_EXCEEDED:
                    raise MultiModelException(408, e.details())
                raise MultiModelException(500, e.details())

        return 'Model {} unloaded.'.format(model_name)

    def _read_model_config(self, model_config_file):
        with open(model_config_file, 'r') as f:
            model_config = f.read()
        return model_config

    def _add_model_to_config_file(self, model_name, base_path, model_platform):
        # read existing model-config.cfg, excluding the closing bracket
        config = ''
        with open(MODEL_CONFIG_FILE, 'r') as f:
            line = f.readline()
            while line:
                if not line.startswith('}'):  # closing bracket of model-config.cfg
                    config += line
                line = f.readline()

        # add new model config, append the closing bracket
        config += '  config: {\n'
        config += '    name: "{}",\n'.format(model_name)
        config += '    base_path: "{}",\n'.format(base_path)
        config += '    model_platform: "{}"\n'.format(model_platform)
        config += '  }\n'
        config += '}\n'

        # write back to model-config.cfg
        with open(MODEL_CONFIG_FILE, 'w') as f:
            f.write(config)

    def _delete_model_from_config_file(self, model_server_config):
        if model_server_config:
            with open(MODEL_CONFIG_FILE, 'w') as f:
                f.write(str(model_server_config))
