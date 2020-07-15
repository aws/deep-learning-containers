# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import logging
import os
import re
import signal
import subprocess
import tfs_utils

from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

JS_PING = 'js_content ping'
JS_INVOCATIONS = 'js_content invocations'
GUNICORN_PING = 'proxy_pass http://gunicorn_upstream/ping'
GUNICORN_INVOCATIONS = 'proxy_pass http://gunicorn_upstream/invocations'

PYTHON_LIB_PATH = '/opt/ml/model/code/lib'
REQUIREMENTS_PATH = '/opt/ml/model/code/requirements.txt'
INFERENCE_PATH = '/opt/ml/model/code/inference.py'


class ServiceManager(object):
    def __init__(self):
        self._state = 'initializing'
        self._nginx = None
        self._tfs = None
        self._gunicorn = None
        self._gunicorn_command = None
        self._enable_python_service = os.path.exists(INFERENCE_PATH)
        self._tfs_version = os.environ.get('SAGEMAKER_TFS_VERSION', '1.13')
        self._nginx_http_port = os.environ.get('SAGEMAKER_BIND_TO_PORT', '8080')
        self._nginx_loglevel = os.environ.get('SAGEMAKER_TFS_NGINX_LOGLEVEL', 'error')
        self._tfs_default_model_name = os.environ.get('SAGEMAKER_TFS_DEFAULT_MODEL_NAME', 'None')
        self._sagemaker_port_range = os.environ.get('SAGEMAKER_SAFE_PORT_RANGE', None)
        self._tfs_config_path = '/sagemaker/model-config.cfg'
        self._tfs_batching_config_path = '/sagemaker/batching-config.cfg'

        _enable_batching = os.environ.get('SAGEMAKER_TFS_ENABLE_BATCHING', 'false').lower()
        _enable_multi_model_endpoint = os.environ.get('SAGEMAKER_MULTI_MODEL',
                                                      'false').lower()

        if _enable_batching not in ['true', 'false']:
            raise ValueError('SAGEMAKER_TFS_ENABLE_BATCHING must be "true" or "false"')
        self._tfs_enable_batching = _enable_batching == 'true'

        if _enable_multi_model_endpoint not in ['true', 'false']:
            raise ValueError('SAGEMAKER_MULTI_MODEL must be "true" or "false"')
        self._tfs_enable_multi_model_endpoint = _enable_multi_model_endpoint == 'true'

        self._use_gunicorn = self._enable_python_service or self._tfs_enable_multi_model_endpoint

        if self._sagemaker_port_range is not None:
            parts = self._sagemaker_port_range.split('-')
            low = int(parts[0])
            hi = int(parts[1])
            if low + 2 > hi:
                raise ValueError('not enough ports available in SAGEMAKER_SAFE_PORT_RANGE ({})'
                                 .format(self._sagemaker_port_range))
            self._tfs_grpc_port = str(low)
            self._tfs_rest_port = str(low + 1)
        else:
            # just use the standard default ports
            self._tfs_grpc_port = '9000'
            self._tfs_rest_port = '8501'

        # set environment variable for python service
        os.environ['TFS_GRPC_PORT'] = self._tfs_grpc_port
        os.environ['TFS_REST_PORT'] = self._tfs_rest_port

    def _create_tfs_config(self):
        models = tfs_utils.find_models()
        if not models:
            raise ValueError('no SavedModel bundles found!')

        if self._tfs_default_model_name == 'None':
            default_model = os.path.basename(models[0])
            if default_model:
                self._tfs_default_model_name = default_model
                log.info('using default model name: {}'.format(self._tfs_default_model_name))
            else:
                log.info('no default model detected')

        # config (may) include duplicate 'config' keys, so we can't just dump a dict
        config = 'model_config_list: {\n'
        for m in models:
            config += '  config: {\n'
            config += '    name: "{}",\n'.format(os.path.basename(m))
            config += '    base_path: "{}",\n'.format(m)
            config += '    model_platform: "tensorflow"\n'
            config += '  }\n'
        config += '}\n'

        log.info('tensorflow serving model config: \n%s\n', config)

        with open('/sagemaker/model-config.cfg', 'w') as f:
            f.write(config)

    def _setup_gunicorn(self):
        python_path_content = []
        python_path_option = ''

        if self._enable_python_service:
            lib_path_exists = os.path.exists(PYTHON_LIB_PATH)
            requirements_exists = os.path.exists(REQUIREMENTS_PATH)
            python_path_content = ['/opt/ml/model/code']
            python_path_option = '--pythonpath '

            if lib_path_exists:
                python_path_content.append(PYTHON_LIB_PATH)

            if requirements_exists:
                if lib_path_exists:
                    log.warning('loading modules in "{}", ignoring requirements.txt'
                                .format(PYTHON_LIB_PATH))
                else:
                    log.info('installing packages from requirements.txt...')
                    pip_install_cmd = 'pip3 install -r {}'.format(REQUIREMENTS_PATH)
                    try:
                        subprocess.check_call(pip_install_cmd.split())
                    except subprocess.CalledProcessError:
                        log.error('failed to install required packages, exiting.')
                        self._stop()
                        raise ChildProcessError('failed to install required packages.')

        gunicorn_command = (
            'gunicorn -b unix:/tmp/gunicorn.sock -k gevent --chdir /sagemaker '
            '{}{} -e TFS_GRPC_PORT={} -e SAGEMAKER_MULTI_MODEL={} -e SAGEMAKER_SAFE_PORT_RANGE={} '
            'python_service:app').format(python_path_option, ','.join(python_path_content),
                                         self._tfs_grpc_port, self._tfs_enable_multi_model_endpoint,
                                         self._sagemaker_port_range)

        log.info('gunicorn command: {}'.format(gunicorn_command))
        self._gunicorn_command = gunicorn_command

    def _create_nginx_config(self):
        template = self._read_nginx_template()
        pattern = re.compile(r'%(\w+)%')

        template_values = {
            'TFS_VERSION': self._tfs_version,
            'TFS_REST_PORT': self._tfs_rest_port,
            'TFS_DEFAULT_MODEL_NAME': self._tfs_default_model_name,
            'NGINX_HTTP_PORT': self._nginx_http_port,
            'NGINX_LOG_LEVEL': self._nginx_loglevel,
            'FORWARD_PING_REQUESTS': GUNICORN_PING if self._use_gunicorn else JS_PING,
            'FORWARD_INVOCATION_REQUESTS': GUNICORN_INVOCATIONS if self._use_gunicorn
            else JS_INVOCATIONS,
        }

        config = pattern.sub(lambda x: template_values[x.group(1)], template)
        log.info('nginx config: \n%s\n', config)

        with open('/sagemaker/nginx.conf', 'w') as f:
            f.write(config)

    def _read_nginx_template(self):
        with open('/sagemaker/nginx.conf.template', 'r') as f:
            template = f.read()
            if not template:
                raise ValueError('failed to read nginx.conf.template')

            return template

    def _start_tfs(self):
        self._log_version('tensorflow_model_server --version', 'tensorflow version info:')
        cmd = tfs_utils.tfs_command(
            self._tfs_grpc_port,
            self._tfs_rest_port,
            self._tfs_config_path,
            self._tfs_enable_batching,
            self._tfs_batching_config_path,
        )
        log.info('tensorflow serving command: {}'.format(cmd))
        p = subprocess.Popen(cmd.split())
        log.info('started tensorflow serving (pid: %d)', p.pid)
        self._tfs = p

    def _start_gunicorn(self):
        self._log_version('gunicorn --version', 'gunicorn version info:')
        env = os.environ.copy()
        env['TFS_DEFAULT_MODEL_NAME'] = self._tfs_default_model_name
        p = subprocess.Popen(self._gunicorn_command.split(), env=env)
        log.info('started gunicorn (pid: %d)', p.pid)
        self._gunicorn = p

    def _start_nginx(self):
        self._log_version('/usr/sbin/nginx -V', 'nginx version info:')
        p = subprocess.Popen('/usr/sbin/nginx -c /sagemaker/nginx.conf'.split())
        log.info('started nginx (pid: %d)', p.pid)
        self._nginx = p

    def _log_version(self, command, message):
        try:
            output = subprocess.check_output(
                command.split(),
                stderr=subprocess.STDOUT).decode('utf-8', 'backslashreplace').strip()
            log.info('{}\n{}'.format(message, output))
        except subprocess.CalledProcessError:
            log.warning('failed to run command: %s', command)

    def _stop(self, *args):  # pylint: disable=W0613
        self._state = 'stopping'
        log.info('stopping services')
        try:
            os.kill(self._nginx.pid, signal.SIGQUIT)
        except OSError:
            pass
        try:
            if self._gunicorn:
                os.kill(self._gunicorn.pid, signal.SIGTERM)
        except OSError:
            pass
        try:
            os.kill(self._tfs.pid, signal.SIGTERM)
        except OSError:
            pass

        self._state = 'stopped'
        log.info('stopped')

    def _wait_for_gunicorn(self):
        while True:
            if os.path.exists('/tmp/gunicorn.sock'):
                log.info('gunicorn server is ready!')
                return

    @contextmanager
    def _timeout(self, seconds):
        def _raise_timeout_error(signum, frame):
            raise TimeoutError('time out after {} seconds'.format(seconds))

        try:
            signal.signal(signal.SIGALRM, _raise_timeout_error)
            signal.alarm(seconds)
            yield
        finally:
            signal.alarm(0)

    def start(self):
        log.info('starting services')
        self._state = 'starting'
        signal.signal(signal.SIGTERM, self._stop)

        self._create_nginx_config()

        if self._tfs_enable_batching:
            log.info('batching is enabled')
            tfs_utils.create_batching_config(self._tfs_batching_config_path)

        if self._tfs_enable_multi_model_endpoint:
            log.info('multi-model endpoint is enabled, TFS model servers will be started later')
        else:
            tfs_utils.create_tfs_config(
                self._tfs_default_model_name,
                self._tfs_config_path
            )
            self._create_tfs_config()
            self._start_tfs()

        if self._use_gunicorn:
            self._setup_gunicorn()
            self._start_gunicorn()
            # make sure gunicorn is up
            with self._timeout(seconds=30):
                self._wait_for_gunicorn()

        self._start_nginx()
        self._state = 'started'

        while True:
            pid, status = os.wait()

            if self._state != 'started':
                break

            if pid == self._nginx.pid:
                log.warning('unexpected nginx exit (status: {}). restarting.'.format(status))
                self._start_nginx()

            elif pid == self._tfs.pid:
                log.warning(
                    'unexpected tensorflow serving exit (status: {}). restarting.'.format(status))
                self._start_tfs()

            elif self._gunicorn and pid == self._gunicorn.pid:
                log.warning('unexpected gunicorn exit (status: {}). restarting.'
                            .format(status))
                self._start_gunicorn()

        self._stop()


if __name__ == '__main__':
    ServiceManager().start()
