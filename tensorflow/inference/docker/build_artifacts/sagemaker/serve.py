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

import boto3
import logging
import os
import re
import signal
import subprocess
import tfs_utils

from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

JS_PING = "js_content tensorflowServing.ping"
JS_INVOCATIONS = "js_content tensorflowServing.invocations"
GUNICORN_PING = "proxy_pass http://gunicorn_upstream/ping"
GUNICORN_INVOCATIONS = "proxy_pass http://gunicorn_upstream/invocations"
MULTI_MODEL = "s" if os.environ.get("SAGEMAKER_MULTI_MODEL", "False").lower() == "true" else ""
MODEL_DIR = f"model{MULTI_MODEL}"
CODE_DIR = "/opt/ml/{}/code".format(MODEL_DIR)
PYTHON_LIB_PATH = os.path.join(CODE_DIR, "lib")
REQUIREMENTS_PATH = os.path.join(CODE_DIR, "requirements.txt")
INFERENCE_PATH = os.path.join(CODE_DIR, "inference.py")


class ServiceManager(object):
    def __init__(self):
        self._state = "initializing"
        self._nginx = None
        self._tfs = []
        self._gunicorn = None
        self._gunicorn_command = None
        self._enable_python_service = False
        self._tfs_version = os.environ.get("SAGEMAKER_TFS_VERSION", "1.13")
        self._nginx_http_port = os.environ.get("SAGEMAKER_BIND_TO_PORT", "8080")
        self._nginx_loglevel = os.environ.get("SAGEMAKER_TFS_NGINX_LOGLEVEL", "error")
        self._tfs_default_model_name = os.environ.get("SAGEMAKER_TFS_DEFAULT_MODEL_NAME", "None")
        self._sagemaker_port_range = os.environ.get("SAGEMAKER_SAFE_PORT_RANGE", None)
        self._gunicorn_workers = os.environ.get("SAGEMAKER_GUNICORN_WORKERS", 1)
        self._gunicorn_threads = os.environ.get("SAGEMAKER_GUNICORN_THREADS", 1)
        self._gunicorn_loglevel = os.environ.get("SAGEMAKER_GUNICORN_LOGLEVEL", "info")
        self._tfs_config_path = "/sagemaker/model-config.cfg"
        self._tfs_batching_config_path = "/sagemaker/batching-config.cfg"

        _enable_batching = os.environ.get("SAGEMAKER_TFS_ENABLE_BATCHING", "false").lower()
        _enable_multi_model_endpoint = os.environ.get("SAGEMAKER_MULTI_MODEL", "false").lower()
        # Use this to specify memory that is needed to initialize CUDA/cuDNN and other GPU libraries
        self._tfs_gpu_margin = float(os.environ.get("SAGEMAKER_TFS_FRACTIONAL_GPU_MEM_MARGIN", 0.2))
        self._tfs_instance_count = int(os.environ.get("SAGEMAKER_TFS_INSTANCE_COUNT", 1))
        self._tfs_wait_time_seconds = int(os.environ.get("SAGEMAKER_TFS_WAIT_TIME_SECONDS", 300))
        self._tfs_inter_op_parallelism = os.environ.get("SAGEMAKER_TFS_INTER_OP_PARALLELISM", 0)
        self._tfs_intra_op_parallelism = os.environ.get("SAGEMAKER_TFS_INTRA_OP_PARALLELISM", 0)
        self._gunicorn_worker_class = os.environ.get("SAGEMAKER_GUNICORN_WORKER_CLASS", "gevent")
        self._gunicorn_timeout_seconds = int(
            os.environ.get("SAGEMAKER_GUNICORN_TIMEOUT_SECONDS", 30)
        )

        if os.environ.get("OMP_NUM_THREADS") is None:
            os.environ["OMP_NUM_THREADS"] = "1"

        if _enable_multi_model_endpoint not in ["true", "false"]:
            raise ValueError("SAGEMAKER_MULTI_MODEL must be 'true' or 'false'")
        self._tfs_enable_multi_model_endpoint = _enable_multi_model_endpoint == "true"

        self._need_python_service()
        log.info("PYTHON SERVICE: {}".format(str(self._enable_python_service)))

        if _enable_batching not in ["true", "false"]:
            raise ValueError("SAGEMAKER_TFS_ENABLE_BATCHING must be 'true' or 'false'")
        self._tfs_enable_batching = _enable_batching == "true"

        if _enable_multi_model_endpoint not in ["true", "false"]:
            raise ValueError("SAGEMAKER_MULTI_MODEL must be 'true' or 'false'")
        self._tfs_enable_multi_model_endpoint = _enable_multi_model_endpoint == "true"

        self._use_gunicorn = self._enable_python_service or self._tfs_enable_multi_model_endpoint

        if self._sagemaker_port_range is not None:
            parts = self._sagemaker_port_range.split("-")
            low = int(parts[0])
            hi = int(parts[1])
            self._tfs_grpc_ports = []
            self._tfs_rest_ports = []
            if low + 2 * self._tfs_instance_count > hi:
                raise ValueError(
                    "not enough ports available in SAGEMAKER_SAFE_PORT_RANGE ({})".format(
                        self._sagemaker_port_range
                    )
                )
            # select non-overlapping grpc and rest ports based on tfs instance count
            for i in range(self._tfs_instance_count):
                self._tfs_grpc_ports.append(str(low + 2 * i))
                self._tfs_rest_ports.append(str(low + 2 * i + 1))
            # concat selected ports respectively in order to pass them to python service
            self._tfs_grpc_concat_ports = self._concat_ports(self._tfs_grpc_ports)
            self._tfs_rest_concat_ports = self._concat_ports(self._tfs_rest_ports)
        else:
            # just use the standard default ports
            self._tfs_grpc_ports = ["9000"]
            self._tfs_rest_ports = ["8501"]
            # provide single concat port here for default case
            self._tfs_grpc_concat_ports = "9000"
            self._tfs_rest_concat_ports = "8501"

        # set environment variable for python service
        os.environ["TFS_GRPC_PORTS"] = self._tfs_grpc_concat_ports
        os.environ["TFS_REST_PORTS"] = self._tfs_rest_concat_ports

    def _need_python_service(self):
        if os.path.exists(INFERENCE_PATH):
            self._enable_python_service = True
        if os.environ.get("SAGEMAKER_MULTI_MODEL_UNIVERSAL_BUCKET") and os.environ.get(
            "SAGEMAKER_MULTI_MODEL_UNIVERSAL_PREFIX"
        ):
            self._enable_python_service = True

    def _concat_ports(self, ports):
        str_ports = [str(port) for port in ports]
        concat_str_ports = ",".join(str_ports)
        return concat_str_ports

    def _create_tfs_config(self):
        models = tfs_utils.find_models()

        if not models:
            raise ValueError("no SavedModel bundles found!")

        if self._tfs_default_model_name == "None":
            default_model = os.path.basename(models[0])
            if default_model:
                self._tfs_default_model_name = default_model
                log.info("using default model name: {}".format(self._tfs_default_model_name))
            else:
                log.info("no default model detected")

        # config (may) include duplicate 'config' keys, so we can't just dump a dict
        config = "model_config_list: {\n"
        for m in models:
            config += "  config: {\n"
            config += "    name: '{}'\n".format(os.path.basename(m))
            config += "    base_path: '{}'\n".format(m)
            config += "    model_platform: 'tensorflow'\n"

            config += "    model_version_policy: {\n"
            config += "      specific: {\n"
            for version in tfs_utils.find_model_versions(m):
                config += "        versions: {}\n".format(version)
            config += "      }\n"
            config += "    }\n"

            config += "  }\n"
        config += "}\n"

        log.info("tensorflow serving model config: \n%s\n", config)

        with open(self._tfs_config_path, "w", encoding="utf8") as f:
            f.write(config)

    def _setup_gunicorn(self):
        python_path_content = []
        python_path_option = ""

        bucket = os.environ.get("SAGEMAKER_MULTI_MODEL_UNIVERSAL_BUCKET", None)
        prefix = os.environ.get("SAGEMAKER_MULTI_MODEL_UNIVERSAL_PREFIX", None)

        if not os.path.exists(CODE_DIR) and bucket and prefix:
            self._download_scripts(bucket, prefix)

        if self._enable_python_service:
            lib_path_exists = os.path.exists(PYTHON_LIB_PATH)
            requirements_exists = os.path.exists(REQUIREMENTS_PATH)
            python_path_content = ["/opt/ml/model/code"]
            python_path_option = "--pythonpath "

            if lib_path_exists:
                python_path_content.append(PYTHON_LIB_PATH)

            if requirements_exists:
                if lib_path_exists:
                    log.warning(
                        "loading modules in '{}', ignoring requirements.txt".format(PYTHON_LIB_PATH)
                    )
                else:
                    log.info("installing packages from requirements.txt...")
                    pip_install_cmd = "pip3 install -r {}".format(REQUIREMENTS_PATH)
                    try:
                        subprocess.check_call(pip_install_cmd.split())
                    except subprocess.CalledProcessError:
                        log.error("failed to install required packages, exiting.")
                        self._stop()
                        raise ChildProcessError("failed to install required packages.")

        gunicorn_command = (
            "gunicorn -b unix:/tmp/gunicorn.sock -k {} --chdir /sagemaker "
            "--workers {} --threads {} --log-level {} --timeout {} "
            "{}{} -e TFS_GRPC_PORTS={} -e TFS_REST_PORTS={} "
            "-e SAGEMAKER_MULTI_MODEL={} -e SAGEMAKER_SAFE_PORT_RANGE={} "
            "-e SAGEMAKER_TFS_WAIT_TIME_SECONDS={} "
            "python_service:app"
        ).format(
            self._gunicorn_worker_class,
            self._gunicorn_workers,
            self._gunicorn_threads,
            self._gunicorn_loglevel,
            self._gunicorn_timeout_seconds,
            python_path_option,
            ",".join(python_path_content),
            self._tfs_grpc_concat_ports,
            self._tfs_rest_concat_ports,
            self._tfs_enable_multi_model_endpoint,
            self._sagemaker_port_range,
            self._tfs_wait_time_seconds,
        )

        log.info("gunicorn command: {}".format(gunicorn_command))
        self._gunicorn_command = gunicorn_command

    def _download_scripts(self, bucket, prefix):
        log.info("checking boto session region ...")
        boto_session = boto3.session.Session()
        boto_region = boto_session.region_name
        if boto_region in ("us-iso-east-1", "us-gov-west-1"):
            raise ValueError("Universal scripts is not supported in us-iso-east-1 or us-gov-west-1")

        log.info("downloading universal scripts ...")
        client = boto3.client("s3")
        resource = boto3.resource("s3")
        # download files
        paginator = client.get_paginator("list_objects")
        for result in paginator.paginate(Bucket=bucket, Delimiter="/", Prefix=prefix):
            for file in result.get("Contents", []):
                destination = os.path.join(CODE_DIR, file.get("Key"))
                if not os.path.exists(os.path.dirname(destination)):
                    os.makedirs(os.path.dirname(destination))
                resource.meta.client.download_file(bucket, file.get("Key"), destination)

    def _create_nginx_tfs_upstream(self):
        indentation = "    "
        tfs_upstream = ""
        for port in self._tfs_rest_ports:
            tfs_upstream += "{}server localhost:{};\n".format(indentation, port)
        tfs_upstream = tfs_upstream[len(indentation) : -2]

        return tfs_upstream

    def _create_nginx_config(self):
        template = self._read_nginx_template()
        pattern = re.compile(r"%(\w+)%")

        template_values = {
            "TFS_VERSION": self._tfs_version,
            "TFS_UPSTREAM": self._create_nginx_tfs_upstream(),
            "TFS_DEFAULT_MODEL_NAME": self._tfs_default_model_name,
            "NGINX_HTTP_PORT": self._nginx_http_port,
            "NGINX_LOG_LEVEL": self._nginx_loglevel,
            "FORWARD_PING_REQUESTS": GUNICORN_PING if self._use_gunicorn else JS_PING,
            "FORWARD_INVOCATION_REQUESTS": GUNICORN_INVOCATIONS
            if self._use_gunicorn
            else JS_INVOCATIONS,
        }

        config = pattern.sub(lambda x: template_values[x.group(1)], template)
        log.info("nginx config: \n%s\n", config)

        with open("/sagemaker/nginx.conf", "w", encoding="utf8") as f:
            f.write(config)

    def _read_nginx_template(self):
        with open("/sagemaker/nginx.conf.template", "r", encoding="utf8") as f:
            template = f.read()
            if not template:
                raise ValueError("failed to read nginx.conf.template")

            return template

    def _enable_per_process_gpu_memory_fraction(self):
        nvidia_smi_exist = os.path.exists("/usr/bin/nvidia-smi")
        if self._tfs_instance_count > 1 and nvidia_smi_exist:
            return True

        return False

    def _calculate_per_process_gpu_memory_fraction(self):
        return round((1 - self._tfs_gpu_margin) / float(self._tfs_instance_count), 4)

    def _start_tfs(self):
        self._log_version("tensorflow_model_server --version", "tensorflow version info:")

        for i in range(self._tfs_instance_count):
            p = self._start_single_tfs(i)
            self._tfs.append(p)

    def _start_gunicorn(self):
        self._log_version("gunicorn --version", "gunicorn version info:")
        env = os.environ.copy()
        env["TFS_DEFAULT_MODEL_NAME"] = self._tfs_default_model_name
        p = subprocess.Popen(self._gunicorn_command.split(), env=env)
        log.info("started gunicorn (pid: %d)", p.pid)
        self._gunicorn = p

    def _start_nginx(self):
        self._log_version("/usr/sbin/nginx -V", "nginx version info:")
        p = subprocess.Popen("/usr/sbin/nginx -c /sagemaker/nginx.conf".split())
        log.info("started nginx (pid: %d)", p.pid)
        self._nginx = p

    def _log_version(self, command, message):
        try:
            output = (
                subprocess.check_output(command.split(), stderr=subprocess.STDOUT)
                .decode("utf-8", "backslashreplace")
                .strip()
            )
            log.info("{}\n{}".format(message, output))
        except subprocess.CalledProcessError:
            log.warning("failed to run command: %s", command)

    def _stop(self, *args):  # pylint: disable=W0613
        self._state = "stopping"
        log.info("stopping services")
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
            for tfs in self._tfs:
                os.kill(tfs.pid, signal.SIGTERM)
        except OSError:
            pass

        self._state = "stopped"
        log.info("stopped")

    def _wait_for_gunicorn(self):
        while True:
            if os.path.exists("/tmp/gunicorn.sock"):
                log.info("gunicorn server is ready!")
                return

    def _wait_for_tfs(self):
        for i in range(self._tfs_instance_count):
            tfs_utils.wait_for_model(
                self._tfs_rest_ports[i], self._tfs_default_model_name, self._tfs_wait_time_seconds
            )

    @contextmanager
    def _timeout(self, seconds):
        def _raise_timeout_error(signum, frame):
            raise TimeoutError("time out after {} seconds".format(seconds))

        try:
            signal.signal(signal.SIGALRM, _raise_timeout_error)
            signal.alarm(seconds)
            yield
        finally:
            signal.alarm(0)

    def _is_tfs_process(self, pid):
        for p in self._tfs:
            if p.pid == pid:
                return True
        return False

    def _find_tfs_process(self, pid):
        for index, p in enumerate(self._tfs):
            if p.pid == pid:
                return index
        return None

    def _restart_single_tfs(self, pid):
        instance_id = self._find_tfs_process(pid)
        if instance_id is None:
            raise ValueError("Cannot find tfs with pid: {};".format(pid))
        p = self._start_single_tfs(instance_id)
        self._tfs[instance_id] = p

    def _start_single_tfs(self, instance_id):
        cmd = tfs_utils.tfs_command(
            self._tfs_grpc_ports[instance_id],
            self._tfs_rest_ports[instance_id],
            self._tfs_config_path,
            self._tfs_enable_batching,
            self._tfs_batching_config_path,
            tfs_intra_op_parallelism=self._tfs_intra_op_parallelism,
            tfs_inter_op_parallelism=self._tfs_inter_op_parallelism,
            tfs_enable_gpu_memory_fraction=self._enable_per_process_gpu_memory_fraction(),
            tfs_gpu_memory_fraction=self._calculate_per_process_gpu_memory_fraction(),
        )
        log.info("tensorflow serving command: {}".format(cmd))
        p = subprocess.Popen(cmd.split())
        log.info("started tensorflow serving (pid: %d)", p.pid)
        return p

    def _monitor(self):
        while True:
            pid, status = os.wait()

            if self._state != "started":
                break

            if pid == self._nginx.pid:
                log.warning("unexpected nginx exit (status: {}). restarting.".format(status))
                self._start_nginx()

            elif self._is_tfs_process(pid):
                log.warning(
                    "unexpected tensorflow serving exit (status: {}). restarting.".format(status)
                )
                try:
                    self._restart_single_tfs(pid)
                except (ValueError, OSError) as error:
                    log.error("Failed to restart tensorflow serving. {}".format(error))

            elif self._gunicorn and pid == self._gunicorn.pid:
                log.warning("unexpected gunicorn exit (status: {}). restarting.".format(status))
                self._start_gunicorn()

    def start(self):
        log.info("starting services")
        self._state = "starting"
        signal.signal(signal.SIGTERM, self._stop)

        if self._tfs_enable_batching:
            log.info("batching is enabled")
            tfs_utils.create_batching_config(self._tfs_batching_config_path)

        if self._tfs_enable_multi_model_endpoint:
            log.info("multi-model endpoint is enabled, TFS model servers will be started later")
        else:
            self._create_tfs_config()
            self._start_tfs()
            self._wait_for_tfs()

        self._create_nginx_config()

        if self._use_gunicorn:
            self._setup_gunicorn()
            self._start_gunicorn()
            # make sure gunicorn is up
            with self._timeout(seconds=self._gunicorn_timeout_seconds):
                self._wait_for_gunicorn()

        self._start_nginx()
        self._state = "started"
        self._monitor()
        self._stop()


if __name__ == "__main__":
    ServiceManager().start()
