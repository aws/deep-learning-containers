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
import multiprocessing
import os
import re
import signal
import subprocess

from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

JS_PING = 'js_content ping'
JS_PING_WITHOUT_MODEL = 'js_content ping_without_model'
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

        _enable_batching = os.environ.get('SAGEMAKER_TFS_ENABLE_BATCHING', 'false').lower()
        _enable_dynamic_endpoint = os.environ.get('SAGEMAKER_MULTI_MODEL',
                                                  'false').lower()

        if _enable_batching not in ['true', 'false']:
            raise ValueError('SAGEMAKER_TFS_ENABLE_BATCHING must be "true" or "false"')
        self._tfs_enable_batching = _enable_batching == 'true'

        if _enable_dynamic_endpoint not in ['true', 'false']:
            raise ValueError('SAGEMAKER_MULTI_MODEL must be "true" or "false"')
        self._tfs_enable_dynamic_endpoint = _enable_dynamic_endpoint == 'true'

        self._use_gunicorn = self._enable_python_service or self._tfs_enable_dynamic_endpoint

        if 'SAGEMAKER_SAFE_PORT_RANGE' in os.environ:
            port_range = os.environ['SAGEMAKER_SAFE_PORT_RANGE']
            parts = port_range.split('-')
            low = int(parts[0])
            hi = int(parts[1])
            if low + 2 > hi:
                raise ValueError('not enough ports available in SAGEMAKER_SAFE_PORT_RANGE ({})'
                                 .format(port_range))
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
        # config (may) include duplicate 'config' keys, so we can't just dump a dict
        if self._tfs_enable_dynamic_endpoint:
            config = 'model_config_list: {\n}\n'
            with open('/sagemaker/model-config.cfg', 'w') as f:
                f.write(config)
        else:
            models = self._find_models()
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

    def _create_batching_config(self):

        class _BatchingParameter:
            def __init__(self, key, env_var, value, defaulted_message):
                self.key = key
                self.env_var = env_var
                self.value = value
                self.defaulted_message = defaulted_message

        cpu_count = multiprocessing.cpu_count()
        batching_parameters = [
            _BatchingParameter('max_batch_size', 'SAGEMAKER_TFS_MAX_BATCH_SIZE', 8,
                               "max_batch_size defaulted to {}. Set {} to override default. "
                               "Tuning this parameter may yield better performance."),
            _BatchingParameter('batch_timeout_micros', 'SAGEMAKER_TFS_BATCH_TIMEOUT_MICROS', 1000,
                               "batch_timeout_micros defaulted to {}. Set {} to override "
                               "default. Tuning this parameter may yield better performance."),
            _BatchingParameter('num_batch_threads', 'SAGEMAKER_TFS_NUM_BATCH_THREADS',
                               cpu_count, "num_batch_threads defaulted to {},"
                               "the number of CPUs. Set {} to override default."),
            _BatchingParameter('max_enqueued_batches', 'SAGEMAKER_TFS_MAX_ENQUEUED_BATCHES',
                               # Batch limits number of concurrent requests, which limits number
                               # of enqueued batches, so this can be set high for Batch
                               100000000 if 'SAGEMAKER_BATCH' in os.environ else cpu_count,
                               "max_enqueued_batches defaulted to {}. Set {} to override default. "
                               "Tuning this parameter may be necessary to tune out-of-memory "
                               "errors occur."),
        ]

        warning_message = ''
        for batching_parameter in batching_parameters:
            if batching_parameter.env_var in os.environ:
                batching_parameter.value = os.environ[batching_parameter.env_var]
            else:
                warning_message += batching_parameter.defaulted_message.format(
                    batching_parameter.value, batching_parameter.env_var)
                warning_message += '\n'
        if warning_message:
            log.warning(warning_message)

        config = ''
        for batching_parameter in batching_parameters:
            config += '%s { value: %s }\n' % (batching_parameter.key, batching_parameter.value)

        log.info('batching config: \n%s\n', config)
        with open('/sagemaker/batching-config.cfg', 'w') as f:
            f.write(config)

    def _get_tfs_batching_args(self):
        if self._tfs_enable_batching:
            return "--enable_batching=true " \
                   "--batching_parameters_file=/sagemaker/batching-config.cfg"
        else:
            return ""

    def _find_models(self):
        base_path = '/opt/ml/model'
        models = []
        for f in self._find_saved_model_files(base_path):
            parts = f.split('/')
            if len(parts) >= 6 and re.match(r'^\d+$', parts[-2]):
                model_path = '/'.join(parts[0:-2])
                if model_path not in models:
                    models.append(model_path)
        return models

    def _find_saved_model_files(self, path):
        for e in os.scandir(path):
            if e.is_dir():
                yield from self._find_saved_model_files(os.path.join(path, e.name))
            else:
                if e.name == 'saved_model.pb':
                    yield os.path.join(path, e.name)

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
            '{}{} -e TFS_GRPC_PORT={} -e SAGEMAKER_MULTI_MODEL={} '
            'python_service:app').format(python_path_option, ','.join(python_path_content),
                                         self._tfs_grpc_port, self._tfs_enable_dynamic_endpoint)

        log.info('gunicorn command: {}'.format(gunicorn_command))
        self._gunicorn_command = gunicorn_command

    def _create_nginx_config(self):
        template = self._read_nginx_template()
        pattern = re.compile(r'%(\w+)%')
        ping_request = JS_PING
        if self._enable_python_service:
            ping_request = GUNICORN_PING
        if self._tfs_enable_dynamic_endpoint or self._tfs_version:
            ping_request = JS_PING_WITHOUT_MODEL

        template_values = {
            'TFS_VERSION': self._tfs_version,
            'TFS_REST_PORT': self._tfs_rest_port,
            'TFS_DEFAULT_MODEL_NAME': self._tfs_default_model_name,
            'NGINX_HTTP_PORT': self._nginx_http_port,
            'NGINX_LOG_LEVEL': self._nginx_loglevel,
            'FORWARD_PING_REQUESTS': ping_request,
            'FORWARD_INVOCATION_REQUESTS': GUNICORN_INVOCATIONS if self._enable_python_service
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
        tfs_config_path = '/sagemaker/model-config.cfg'
        cmd = "tensorflow_model_server " \
              "--port={} " \
              "--rest_api_port={} " \
              "--model_config_file={} " \
              "--max_num_load_retries=0 {}"\
            .format(self._tfs_grpc_port, self._tfs_rest_port, tfs_config_path,
                    self._get_tfs_batching_args())
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

        self._create_tfs_config()
        self._create_nginx_config()

        if self._tfs_enable_batching:
            log.info('batching is enabled')
            self._create_batching_config()

        if self._tfs_enable_dynamic_endpoint:
            log.info('dynamic endpooint is enabled')

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
