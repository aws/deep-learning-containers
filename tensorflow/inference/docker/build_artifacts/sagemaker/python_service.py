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
import signal
from contextlib import contextmanager

import importlib.util
import json
import logging
import os
import re
from collections import namedtuple
from time import sleep

import falcon
import requests
from proxy_client import GRPCProxyClient

from multi_model_utils import MultiModelException

INFERENCE_SCRIPT_PATH = '/opt/ml/model/code/inference.py'
MODEL_CONFIG_FILE_PATH = '/sagemaker/model-config.cfg'
TFS_GRPC_PORT = os.environ.get('TFS_GRPC_PORT')
TFS_REST_PORT = os.environ.get('TFS_REST_PORT')

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DEFAULT_CONTENT_TYPE = 'application/json'
DEFAULT_ACCEPT_HEADER = 'application/json'
CUSTOM_ATTRIBUTES_HEADER = 'X-Amzn-SageMaker-Custom-Attributes'

Context = namedtuple('Context',
                     'model_name, model_version, method, rest_uri, grpc_port, '
                     'custom_attributes, request_content_type, accept_header, content_length')


class InvocationResource(object):

    def __init__(self):
        self._tfs_default_model_name = os.environ['TFS_DEFAULT_MODEL_NAME']
        self._tfs_grpc_port = TFS_GRPC_PORT
        self._tfs_rest_port = TFS_REST_PORT

        self._handler, self._input_handler, self._output_handler = self._import_handlers()
        self._handlers = self._make_handler(self._handler,
                                            self._input_handler,
                                            self._output_handler)

    def on_post(self, req, res):
        data, context = self._parse_request(req)
        try:
            res.status = falcon.HTTP_200
            res.body, res.content_type = self._handlers(data, context)
        except Exception as e:  # pylint: disable=broad-except
            log.exception('exception handling request: {}'.format(e))
            res.status = falcon.HTTP_500
            res.body = json.dumps({
                'error': str(e)
            }).encode('utf-8')  # pylint: disable=E1101

    def _import_handlers(self):
        spec = importlib.util.spec_from_file_location('inference', INFERENCE_SCRIPT_PATH)
        inference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inference)

        _custom_handler, _custom_input_handler, _custom_output_handler = None, None, None
        if hasattr(inference, 'handler'):
            _custom_handler = inference.handler
        elif hasattr(inference, 'input_handler') and hasattr(inference, 'output_handler'):
            _custom_input_handler = inference.input_handler
            _custom_output_handler = inference.output_handler
        else:
            raise NotImplementedError('Handlers are not implemented correctly in user script.')

        return _custom_handler, _custom_input_handler, _custom_output_handler

    def _make_handler(self, custom_handler, custom_input_handler, custom_output_handler):
        if custom_handler:
            return custom_handler

        def handler(data, context):
            processed_input = custom_input_handler(data, context)
            response = requests.post(context.rest_uri, data=processed_input)
            return custom_output_handler(response, context)

        return handler

    def _parse_request(self, req):
        tfs_attributes = self._parse_tfs_custom_attributes(req)
        tfs_uri = self._tfs_uri(self._tfs_rest_port, tfs_attributes)

        context = Context(tfs_attributes.get('tfs-model-name'),
                          tfs_attributes.get('tfs-model-version'),
                          tfs_attributes.get('tfs-method'),
                          tfs_uri,
                          self._tfs_grpc_port,
                          req.get_header(CUSTOM_ATTRIBUTES_HEADER),
                          req.get_header('Content-Type') or DEFAULT_CONTENT_TYPE,
                          req.get_header('Accept') or DEFAULT_ACCEPT_HEADER,
                          req.content_length)

        data = req.stream
        return data, context

    def _parse_tfs_custom_attributes(self, req):
        attributes = {}
        header = req.get_header(CUSTOM_ATTRIBUTES_HEADER)
        if header:
            for attribute in re.findall(r'(tfs-[a-z\-]+=[^,]+)', header):
                k, v = attribute.split('=')
                attributes[k] = v
        return attributes

    def _tfs_uri(self, port, attributes):
        tfs_model_name = attributes.get('tfs-model-name', self._tfs_default_model_name)
        tfs_model_version = attributes.get('tfs-model-version')
        tfs_method = attributes.get('tfs-method', 'predict')

        uri = 'http://localhost:{}/v1/models/{}'.format(port, tfs_model_name)
        if tfs_model_version:
            uri += '/versions/' + tfs_model_version
        uri += ':' + tfs_method
        return uri


class PingResource(object):
    def on_get(self, req, res):  # pylint: disable=W0613
        res.status = falcon.HTTP_200


class ModelManagerResource(object):

    def __init__(self):
        self.grpc_client = GRPCProxyClient(TFS_GRPC_PORT)

    def on_get(self, req, res, model_name=None):  # pylint: disable=W0613
        try:
            models = self._read_model_config()
            if model_name:
                for model in models:
                    if model['modelName'] == model_name:
                        res.body = json.dumps({
                            'modelName': model_name,
                            'modelUrl': model['modelUrl']
                        })
                res.status = falcon.HTTP_404
                res.body = json.dumps({
                    'error': '{} is not loaded.'.format(model_name)
                })
            else:
                res.status = falcon.HTTP_200
                res.body = json.dumps({
                    'models': models
                })
        except ValueError as e:
            log.exception('exception handling request: {}'.format(e))
            res.status = falcon.HTTP_500
            res.body = json.dumps({
                'error': str(e)
            }).encode('utf-8')

    def on_post(self, req, res):
        data = json.loads(req.stream.read()
                          .decode('utf-8'))
        model_name = data['model_name']
        base_path = data['url']
        if self.validate_model_dir(base_path):
            try:
                msg = self.grpc_client.add_model(model_name, base_path)

                # make sure all versions of model is ready before returning the call
                with self._timeout(seconds=60):
                    while True:
                        model_status = self._get_model_status(model_name)
                        if self._check_all_versions_available(model_status):
                            break
                        sleep(1)

                res.body = msg
                res.status = falcon.HTTP_200
            except MultiModelException as multi_model_exception:
                if multi_model_exception.code == 409:
                    res.status = falcon.HTTP_409
                    res.body = multi_model_exception.msg
                elif multi_model_exception.code == 408:
                    res.status = falcon.HTTP_408
                    res.body = multi_model_exception.msg
                else:
                    raise MultiModelException(falcon.HTTP_500, multi_model_exception.msg)
        else:
            res.status = falcon.HTTP_404
            res.body = 'Could not find valid base path {} for servable {}'.format(base_path,
                                                                                  model_name)

    def on_delete(self, req, res, model_name):  # pylint: disable=W0613
        try:
            msg = self.grpc_client.delete_model(model_name)

            # make sure all versions of model is unloaded before returning the call
            with self._timeout(seconds=60):
                while True:
                    model_status = self._get_model_status(model_name)
                    if self._check_all_versions_unloaded(model_status):
                        break
                    sleep(1)

            res.body = msg
            res.status = falcon.HTTP_200
        except FileNotFoundError:
            res.status = falcon.HTTP_404
            res.body = '{} not loaded yet.'.format(model_name)
        except MultiModelException as multi_model_exception:
            if multi_model_exception.code == 408:
                res.status = falcon.HTTP_408
                res.body = multi_model_exception.msg
            else:
                raise MultiModelException(falcon.HTTP_500, multi_model_exception.msg)

    def _read_model_config(self):
        models = []
        name_key = re.compile(r'([ \t]*)name:(.*)')
        uri_key = re.compile(r'([ \t]*)base_path:(.*)')
        pattern = r'"([A-Za-z0-9_\./\\-]*)"'
        with open(MODEL_CONFIG_FILE_PATH, 'r') as f:
            line = f.readline()
            while line:
                if name_key.search(line):
                    model_name = re.search(pattern, line).group().strip('\"')
                    line = f.readline()
                    if uri_key.search(line):
                        uri = re.search(pattern, line).group().strip('\"')
                        models.append(json.dumps({
                            'modelName': model_name,
                            'modelUrl': uri
                        }))
                    else:
                        raise ValueError('Malformed model-config.cfg file.')
                line = f.readline()
        return models

    def validate_model_dir(self, model_path):
        # model base path doesn't exits
        if not os.path.exists(model_path):
            return False
        # model versions doesn't exist
        versions = []
        for _, dirs, _ in os.walk(model_path):
            for dirname in dirs:
                if dirname.isdigit():
                    versions.append(dirname)
        return self.validate_model_versions(versions)

    def validate_model_versions(self, versions):
        if not versions:
            return False
        for v in versions:
            if v.isdigit():
                # TensorFlow model server will succeed with any versions found
                # even if there are directories that's not a valid model version,
                # the loading will succeed.
                return True
        return False

    def _get_model_status(self, model_name):
        url = 'http://localhost:{}/v1/models/{}'.format(TFS_REST_PORT, model_name)
        try:
            response = requests.get(url).json()
            version_states = []
            for version in response['model_version_status']:
                version_states.append({
                    'version': version['version'],
                    'state': version['state']
                })
            return version_states
        except Exception as e:  # pylint: disable=W0703
            raise Exception(502, str(e))

    def _check_all_versions_available(self, model_status):
        if not model_status:
            raise Exception(404, 'No model versions found')
        for version in model_status:
            if version['state'] != 'AVAILABLE':
                return False
        return True

    def _check_all_versions_unloaded(self, model_status):
        if not model_status:
            raise Exception(404, 'No model versions found')
        for version in model_status:
            if version['state'] != 'END':
                return False
        return True

    @contextmanager
    def _timeout(self, seconds):
        def _raise_timeout_error(signum, frame):
            raise Exception(408, 'Timed our after {} seconds'.format(seconds))

        try:
            signal.signal(signal.SIGALRM, _raise_timeout_error)
            signal.alarm(seconds)
            yield
        finally:
            signal.alarm(0)


class ServiceResources(object):
    def __init__(self):
        self._enable_python_service = os.path.exists(INFERENCE_SCRIPT_PATH)
        self._enable_model_manager = os.environ.get('SAGEMAKER_MULTI_MODEL')

    def add_routes(self, application):
        if self._enable_python_service:
            ping_resource = PingResource()
            invocation_resource = InvocationResource()
            application.add_route('/ping', ping_resource)
            application.add_route('/invocations', invocation_resource)

        if self._enable_model_manager:
            model_manager_resource = ModelManagerResource()
            application.add_route('/models', model_manager_resource)
            application.add_route('/models/{model_name}', model_manager_resource)


app = falcon.API()
resources = ServiceResources()
resources.add_routes(app)
