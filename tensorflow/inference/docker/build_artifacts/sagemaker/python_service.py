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
import bisect
import importlib.util
import json
import logging
import os
import subprocess
import time
import sys

import falcon
import requests

from urllib3.util.retry import Retry

from multi_model_utils import lock, timeout, MultiModelException
import tfs_utils

SAGEMAKER_MULTI_MODEL_ENABLED = os.environ.get('SAGEMAKER_MULTI_MODEL', 'false').lower() == 'true'
INFERENCE_SCRIPT_PATH = '/opt/ml/model/code/inference.py'

SAGEMAKER_BATCHING_ENABLED = os.environ.get('SAGEMAKER_TFS_ENABLE_BATCHING', 'false').lower()
MODEL_CONFIG_FILE_PATH = '/sagemaker/model-config.cfg'
TFS_GRPC_PORT = os.environ.get('TFS_GRPC_PORT')
TFS_REST_PORT = os.environ.get('TFS_REST_PORT')
SAGEMAKER_TFS_PORT_RANGE = os.environ.get('SAGEMAKER_SAFE_PORT_RANGE')


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

CUSTOM_ATTRIBUTES_HEADER = 'X-Amzn-SageMaker-Custom-Attributes'


def default_handler(data, context):
    """A default inference request handler that directly send post request to TFS rest port with
    un-processed data and return un-processed response

    :param data: input data
    :param context: context instance that contains tfs_rest_uri
    :return: inference response from TFS model server
    """
    data = data.read().decode("utf-8")
    if not isinstance(data, str):
        data = json.loads(data)
    response = requests.post(context.rest_uri, data=data)
    return response.content, context.accept_header


class PythonServiceResource:

    def __init__(self):
        if SAGEMAKER_MULTI_MODEL_ENABLED:
            self._model_tfs_rest_port = {}
            self._model_tfs_grpc_port = {}
            self._model_tfs_pid = {}
            self._tfs_ports = self._parse_sagemaker_port_range(SAGEMAKER_TFS_PORT_RANGE)
            # If Multi-Model mode is enabled, dependencies/handlers will be imported
            # during the _handle_load_model_post()
            self.model_handlers = {}
        else:
            self._tfs_grpc_port = TFS_GRPC_PORT
            self._tfs_rest_port = TFS_REST_PORT

            if os.path.exists(INFERENCE_SCRIPT_PATH):
                self._handler, self._input_handler, self._output_handler = self._import_handlers()
                self._handlers = self._make_handler(self._handler,
                                                    self._input_handler,
                                                    self._output_handler)
            else:
                self._handlers = default_handler

        self._tfs_enable_batching = SAGEMAKER_BATCHING_ENABLED == 'true'
        self._tfs_default_model_name = os.environ.get('TFS_DEFAULT_MODEL_NAME', "None")

    def on_post(self, req, res, model_name=None):
        log.info(req.uri)
        if model_name or "invocations" in req.uri:
            self._handle_invocation_post(req, res, model_name)
        else:
            data = json.loads(req.stream.read().decode('utf-8'))
            self._handle_load_model_post(res, data)

    def _parse_sagemaker_port_range(self, port_range):
        lower, upper = port_range.split('-')
        lower = int(lower)
        upper = lower + int((int(upper) - lower) * 0.9)  # only utilizing 90% of the ports
        rest_port = lower
        grpc_port = (lower + upper) // 2
        tfs_ports = {
            'rest_port': [port for port in range(rest_port, grpc_port)],
            'grpc_port': [port for port in range(grpc_port, upper)],
        }
        return tfs_ports

    def _ports_available(self):
        with lock():
            rest_ports = self._tfs_ports['rest_port']
            grpc_ports = self._tfs_ports['grpc_port']
        return len(rest_ports) > 0 and len(grpc_ports) > 0

    def _handle_load_model_post(self, res, data):  # noqa: C901
        model_name = data['model_name']
        base_path = data['url']

        # model is already loaded
        if model_name in self._model_tfs_pid:
            res.status = falcon.HTTP_409
            res.body = json.dumps({
                'error': 'Model {} is already loaded.'.format(model_name)
            })

        # check if there are available ports
        if not self._ports_available():
            res.status = falcon.HTTP_507
            res.body = json.dumps({
                'error': 'Memory exhausted: no available ports to load the model.'
            })
        with lock():
            self._model_tfs_rest_port[model_name] = self._tfs_ports['rest_port'].pop()
            self._model_tfs_grpc_port[model_name] = self._tfs_ports['grpc_port'].pop()

        # validate model files are in the specified base_path
        if self.validate_model_dir(base_path):
            try:
                # install custom dependencies, import handlers
                self._import_custom_modules(model_name)

                tfs_config = tfs_utils.create_tfs_config_individual_model(model_name, base_path)
                tfs_config_file = '/sagemaker/tfs-config/{}/model-config.cfg'.format(model_name)
                log.info('tensorflow serving model config: \n%s\n', tfs_config)
                os.makedirs(os.path.dirname(tfs_config_file))
                with open(tfs_config_file, 'w') as f:
                    f.write(tfs_config)

                batching_config_file = '/sagemaker/batching/{}/batching-config.cfg'.format(
                    model_name)
                if self._tfs_enable_batching:
                    tfs_utils.create_batching_config(batching_config_file)

                cmd = tfs_utils.tfs_command(
                    self._model_tfs_grpc_port[model_name],
                    self._model_tfs_rest_port[model_name],
                    tfs_config_file,
                    self._tfs_enable_batching,
                    batching_config_file,
                )
                p = subprocess.Popen(cmd.split())
                self._wait_for_model(model_name)

                log.info('started tensorflow serving (pid: %d)', p.pid)
                # update model name <-> tfs pid map
                self._model_tfs_pid[model_name] = p

                res.status = falcon.HTTP_200
                res.body = json.dumps({
                    'success':
                        'Successfully loaded model {}, '
                        'listening on rest port {} '
                        'and grpc port {}.'.format(model_name,
                                                   self._model_tfs_rest_port,
                                                   self._model_tfs_grpc_port,)
                })
            except MultiModelException as multi_model_exception:
                self._cleanup_config_file(tfs_config_file)
                self._cleanup_config_file(batching_config_file)
                if multi_model_exception.code == 409:
                    res.status = falcon.HTTP_409
                    res.body = multi_model_exception.msg
                elif multi_model_exception.code == 408:
                    res.status = falcon.HTTP_408
                    res.body = multi_model_exception.msg
                else:
                    raise MultiModelException(falcon.HTTP_500, multi_model_exception.msg)
            except FileExistsError as e:
                res.status = falcon.HTTP_409
                res.body = json.dumps({
                    'error': 'Model {} is already loaded. {}'.format(model_name, str(e))
                })
            except OSError as os_error:
                self._cleanup_config_file(tfs_config_file)
                self._cleanup_config_file(batching_config_file)
                if os_error.errno == 12:
                    raise MultiModelException(falcon.HTTP_507,
                                              'Memory exhausted: '
                                              'not enough memory to start TFS instance')
                else:
                    raise MultiModelException(falcon.HTTP_500, os_error.strerror)
        else:
            res.status = falcon.HTTP_404
            res.body = json.dumps({
                'error':
                    'Could not find valid base path {} for servable {}'.format(base_path,
                                                                               model_name)
            })

    def _import_custom_modules(self, model_name):
        inference_script_path = "/opt/ml/models/{}/model/code/inference.py".format(model_name)
        requirements_file_path = "/opt/ml/models/{}/model/code/requirements.txt".format(model_name)
        python_lib_path = "/opt/ml/models/{}/model/code/lib".format(model_name)

        if os.path.exists(requirements_file_path):
            log.info("pip install dependencies from requirements.txt")
            pip_install_cmd = "pip3 install -r {}".format(requirements_file_path)
            try:
                subprocess.check_call(pip_install_cmd.split())
            except subprocess.CalledProcessError:
                log.error('failed to install required packages, exiting.')
                raise ChildProcessError('failed to install required packages.')

        if os.path.exists(python_lib_path):
            log.info("add Python code library path")
            sys.path.append(python_lib_path)

        if os.path.exists(inference_script_path):
            handler, input_handler, output_handler = self._import_handlers(model_name)
            model_handlers = self._make_handler(handler,
                                                input_handler,
                                                output_handler)
            self.model_handlers[model_name] = model_handlers
        else:
            self.model_handlers[model_name] = default_handler

    def _cleanup_config_file(self, config_file):
        if os.path.exists(config_file):
            os.remove(config_file)

    def _wait_for_model(self, model_name):
        url = "http://localhost:{}/v1/models/{}".format(self._model_tfs_rest_port[model_name],
                                                        model_name)
        with timeout():
            while True:
                time.sleep(0.5)
                try:
                    session = requests.Session()
                    retries = Retry(total=9,
                                    backoff_factor=0.1)
                    session.mount('http://', requests.adapters.HTTPAdapter(max_retries=retries))
                    response = session.get(url)
                    if response.status_code == 200:
                        versions = json.loads(response.content)['model_version_status']
                        if all(version["state"] == "AVAILABLE" for version in versions):
                            break
                except ConnectionError:
                    log.exception("Failed to load models.")

    def _handle_invocation_post(self, req, res, model_name=None):
        if SAGEMAKER_MULTI_MODEL_ENABLED:
            if model_name:
                if model_name not in self._model_tfs_rest_port:
                    res.status = falcon.HTTP_404
                    res.body = json.dumps({
                        'error': "Model {} is not loaded yet.".format(model_name)
                    })
                    return
                else:
                    log.info("model name: {}".format(model_name))
                    rest_port = self._model_tfs_rest_port[model_name]
                    log.info("rest port: {}".format(str(self._model_tfs_rest_port[model_name])))
                    grpc_port = self._model_tfs_grpc_port[model_name]
                    log.info("grpc port: {}".format(str(self._model_tfs_grpc_port[model_name])))
                    data, context = tfs_utils.parse_request(req, rest_port, grpc_port,
                                                            self._tfs_default_model_name,
                                                            model_name)
            else:
                res.status = falcon.HTTP_400
                res.body = json.dumps({
                    'error': 'Invocation request does not contain model name.'
                })
        else:
            data, context = tfs_utils.parse_request(req, self._tfs_rest_port, self._tfs_grpc_port,
                                                    self._tfs_default_model_name)

        try:
            res.status = falcon.HTTP_200
            if SAGEMAKER_MULTI_MODEL_ENABLED:
                with lock():
                    handlers = self.model_handlers[model_name]
                    res.body, res.content_type = handlers(data, context)
            else:
                res.body, res.content_type = self._handlers(data, context)
        except Exception as e:  # pylint: disable=broad-except
            log.exception('exception handling request: {}'.format(e))
            res.status = falcon.HTTP_500
            res.body = json.dumps({
                'error': str(e)
            }).encode('utf-8')  # pylint: disable=E1101

    def _import_handlers(self, model_name=None):
        inference_script = INFERENCE_SCRIPT_PATH
        if model_name:
            inference_script = "/opt/ml/models/{}/model/code/inference.py".format(model_name)
        spec = importlib.util.spec_from_file_location('inference', inference_script)
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

    def on_get(self, req, res, model_name=None):  # pylint: disable=W0613
        if model_name is None:
            models_info = {}
            uri = 'http://localhost:{}/v1/models/{}'
            for model, port in self._model_tfs_rest_port.items():
                try:
                    info = json.loads(requests.get(uri.format(port, model)).content)
                    models_info[model] = info
                except ValueError as e:
                    log.exception('exception handling request: {}'.format(e))
                    res.status = falcon.HTTP_500
                    res.body = json.dumps({
                        'error': str(e)
                    }).encode('utf-8')
            res.status = falcon.HTTP_200
            res.body = json.dumps(models_info)
        else:
            if model_name not in self._model_tfs_rest_port:
                res.status = falcon.HTTP_404
                res.body = json.dumps({
                    'error': 'Model {} is loaded yet.'.format(model_name)
                }).encode('utf-8')
            else:
                port = self._model_tfs_rest_port[model_name]
                uri = 'http://localhost:{}/v1/models/{}'.format(port, model_name)
                try:
                    info = requests.get(uri)
                    res.status = falcon.HTTP_200
                    res.body = json.dumps({
                        'model': info
                    }).encode('utf-8')
                except ValueError as e:
                    log.exception('exception handling GET models request.')
                    res.status = falcon.HTTP_500
                    res.body = json.dumps({
                        'error': str(e)
                    }).encode('utf-8')

    def on_delete(self, req, res, model_name):  # pylint: disable=W0613
        if model_name not in self._model_tfs_pid:
            res.status = falcon.HTTP_404
            res.body = json.dumps({
                'error': 'Model {} is not loaded yet'.format(model_name)
            })
        else:
            try:
                self._model_tfs_pid[model_name].kill()
                os.remove('/sagemaker/tfs-config/{}/model-config.cfg'.format(model_name))
                os.rmdir('/sagemaker/tfs-config/{}'.format(model_name))
                release_rest_port = self._model_tfs_rest_port[model_name]
                release_grpc_port = self._model_tfs_grpc_port[model_name]
                with lock():
                    bisect.insort(self._tfs_ports['rest_port'], release_rest_port)
                    bisect.insort(self._tfs_ports['grpc_port'], release_grpc_port)
                del self._model_tfs_rest_port[model_name]
                del self._model_tfs_grpc_port[model_name]
                del self._model_tfs_pid[model_name]
                res.status = falcon.HTTP_200
                res.body = json.dumps({
                    'success': 'Successfully unloaded model {}.'.format(model_name)
                })
            except OSError as error:
                res.status = falcon.HTTP_500
                res.body = json.dumps({
                    'error': str(error)
                }).encode('utf-8')

    def validate_model_dir(self, model_path):
        # model base path doesn't exits
        if not os.path.exists(model_path):
            return False
        versions = []
        for _, dirs, _ in os.walk(model_path):
            for dirname in dirs:
                if dirname.isdigit():
                    versions.append(dirname)
        return self.validate_model_versions(versions)

    def validate_model_versions(self, versions):
        log.info(versions)
        if not versions:
            return False
        for v in versions:
            if v.isdigit():
                # TensorFlow model server will succeed with any versions found
                # even if there are directories that's not a valid model version,
                # the loading will succeed.
                return True
        return False


class PingResource:
    def on_get(self, req, res):  # pylint: disable=W0613
        res.status = falcon.HTTP_200


class ServiceResources:
    def __init__(self):
        self._enable_model_manager = SAGEMAKER_MULTI_MODEL_ENABLED
        self._python_service_resource = PythonServiceResource()
        self._ping_resource = PingResource()

    def add_routes(self, application):
        application.add_route('/ping', self._ping_resource)
        application.add_route('/invocations', self._python_service_resource)

        if self._enable_model_manager:
            application.add_route('/models', self._python_service_resource)
            application.add_route('/models/{model_name}', self._python_service_resource)
            application.add_route('/models/{model_name}/invoke', self._python_service_resource)


app = falcon.API()
resources = ServiceResources()
resources.add_routes(app)
