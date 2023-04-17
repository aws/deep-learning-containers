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
import argparse
import importlib.util
import json
import logging
import os
import signal
import subprocess
import grpc
import sys
import shutil
import copy
import pickle

import falcon
import requests
import random

from multi_model_utils import MultiModelException, lock
import tfs_utils

SAGEMAKER_MULTI_MODEL_ENABLED = os.environ.get("SAGEMAKER_MULTI_MODEL", "false").lower() == "true"
INFERENCE_SCRIPT_PATH = (
    "/opt/ml/code/inference.py"
    if SAGEMAKER_MULTI_MODEL_ENABLED
    else "/opt/ml/model/code/inference.py"
)

SAGEMAKER_BATCHING_ENABLED = os.environ.get("SAGEMAKER_TFS_ENABLE_BATCHING", "false").lower()
MODEL_CONFIG_FILE_PATH = "/sagemaker/model-config.cfg"
TFS_GRPC_PORTS = os.environ.get("TFS_GRPC_PORTS")
TFS_REST_PORTS = os.environ.get("TFS_REST_PORTS")
SAGEMAKER_TFS_PORT_RANGE = os.environ.get("SAGEMAKER_SAFE_PORT_RANGE")
TFS_INSTANCE_COUNT = int(os.environ.get("SAGEMAKER_TFS_INSTANCE_COUNT", "1"))

logging.basicConfig(
    format="%(process)d %(asctime)s %(levelname)-8s %(message)s", force=True, level=logging.INFO
)
log = logging.getLogger(__name__)

CUSTOM_ATTRIBUTES_HEADER = "X-Amzn-SageMaker-Custom-Attributes"
MME_TFS_INSTANCE_STATUS_FILE = "/sagemaker/tfs_instance.pickle"


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


class TfsInstanceStatus:
    def __init__(self, rest_port: str, grpc_port: str, pid: int):
        self.rest_port = rest_port
        self.grpc_port = grpc_port
        self.pid = pid

    def __repr__(self):
        return f"TFS Instance Status (rest_port : {self.rest_port}, grpc_port: {self.grpc_port}, pid: {self.pid}))"


class PythonServiceResource:
    def __init__(self):
        if SAGEMAKER_MULTI_MODEL_ENABLED:
            self._mme_tfs_instances_status: dict[str, [TfsInstanceStatus]] = {}
            self._tfs_ports = self._parse_sagemaker_port_range_mme(SAGEMAKER_TFS_PORT_RANGE)
            self._tfs_available_ports = self._parse_sagemaker_port_range_mme(
                SAGEMAKER_TFS_PORT_RANGE
            )
            # If Multi-Model mode is enabled, dependencies/handlers will be imported
            # during the _handle_load_model_post()
            self.model_handlers = {}
        else:
            self._tfs_grpc_ports = self._parse_concat_ports(TFS_GRPC_PORTS)
            self._tfs_rest_ports = self._parse_concat_ports(TFS_REST_PORTS)

            self._channels = {}
            for grpc_port in self._tfs_grpc_ports:
                # Initialize grpc channel here so gunicorn worker could have mapping
                # between each grpc port and channel
                self._setup_channel(grpc_port)

        self._default_handlers_enabled = False
        if os.path.exists(INFERENCE_SCRIPT_PATH):
            # Single-Model Mode & Multi-Model Mode both use one inference.py
            self._handler, self._input_handler, self._output_handler = self._import_handlers()
            self._handlers = self._make_handler(
                self._handler, self._input_handler, self._output_handler
            )
        else:
            self._handlers = default_handler
            self._default_handlers_enabled = True

        self._tfs_enable_batching = SAGEMAKER_BATCHING_ENABLED == "true"
        self._tfs_default_model_name = os.environ.get("TFS_DEFAULT_MODEL_NAME", "None")
        self._tfs_inter_op_parallelism = os.environ.get("SAGEMAKER_TFS_INTER_OP_PARALLELISM", 0)
        self._tfs_intra_op_parallelism = os.environ.get("SAGEMAKER_TFS_INTRA_OP_PARALLELISM", 0)
        self._tfs_instance_count = int(os.environ.get("SAGEMAKER_TFS_INSTANCE_COUNT", 1))
        self._gunicorn_workers = int(os.environ.get("SAGEMAKER_GUNICORN_WORKERS", 1))
        self._tfs_wait_time_seconds = int(
            os.environ.get("SAGEMAKER_TFS_WAIT_TIME_SECONDS", 55 // self._tfs_instance_count)
        )

    def on_post(self, req, res, model_name=None):
        if model_name or "invocations" in req.uri:
            self._handle_invocation_post(req, res, model_name)
        else:
            data = json.loads(req.stream.read().decode("utf-8"))
            self._handle_load_model_post(res, data)

    def _parse_concat_ports(self, concat_ports):
        return concat_ports.split(",")

    def _pick_port(self, ports):
        return random.choice(ports)

    def _parse_sagemaker_port_range_mme(self, port_range):
        lower, upper = port_range.split("-")
        lower = int(lower)
        upper = lower + int((int(upper) - lower) * 0.9)  # only utilizing 90% of the ports
        rest_port = lower
        grpc_port = (lower + upper) // 2
        tfs_ports = {
            "rest_port": [port for port in range(rest_port, grpc_port)],
            "grpc_port": [port for port in range(grpc_port, upper)],
        }
        return tfs_ports

    def _ports_available(self):
        rest_ports = self._tfs_available_ports["rest_port"]
        grpc_ports = self._tfs_available_ports["grpc_port"]
        return len(rest_ports) > 0 and len(grpc_ports) > 0

    def _update_ports_available(self):
        self._tfs_available_ports = copy.deepcopy(self._tfs_ports)
        for _, tf_status_list in self._mme_tfs_instances_status.items():
            for tf_status in tf_status_list:
                if tf_status.rest_port in self._tfs_available_ports["rest_port"]:
                    self._tfs_available_ports["rest_port"].remove(tf_status.rest_port)
                if tf_status.grpc_port in self._tfs_available_ports["grpc_port"]:
                    self._tfs_available_ports["grpc_port"].remove(tf_status.grpc_port)
        log.info(f"available ports : {self._tfs_available_ports}")

    def _load_model(self, model_name, base_path, rest_port, grpc_port, model_index):
        if self.validate_model_dir(base_path):
            try:
                self._import_custom_modules(model_name)
                tfs_config = tfs_utils.create_tfs_config_individual_model(model_name, base_path)
                tfs_config_file = "/sagemaker/tfs-config/{}/{}/model-config.cfg".format(
                    model_name, model_index
                )
                log.info("tensorflow serving model config: \n%s\n", tfs_config)
                os.makedirs(os.path.dirname(tfs_config_file))
                with open(tfs_config_file, "w", encoding="utf8") as f:
                    f.write(tfs_config)

                batching_config_file = "/sagemaker/batching/{}/{}/batching-config.cfg".format(
                    model_name, model_index
                )
                if self._tfs_enable_batching:
                    tfs_utils.create_batching_config(batching_config_file)

                cmd = tfs_utils.tfs_command(
                    grpc_port,
                    rest_port,
                    tfs_config_file,
                    self._tfs_enable_batching,
                    batching_config_file,
                    tfs_intra_op_parallelism=self._tfs_intra_op_parallelism,
                    tfs_inter_op_parallelism=self._tfs_inter_op_parallelism,
                )
                log.info("MME starts tensorflow serving with command: {}".format(cmd))
                p = subprocess.Popen(cmd.split())

                tfs_utils.wait_for_model(rest_port, model_name, self._tfs_wait_time_seconds, p.pid)

                log.info("started tensorflow serving (pid: %d)", p.pid)

                return {
                    "status": falcon.HTTP_200,
                    "body": json.dumps(
                        {
                            "success": "Successfully loaded model {}, "
                            "listening on rest port {} "
                            "and grpc port {}.".format(model_name, rest_port, grpc_port)
                        },
                    ),
                    "pid": p.pid,
                }
            except MultiModelException as multi_model_exception:
                if multi_model_exception.code == 409:
                    return {
                        "status": falcon.HTTP_409,
                        "body": multi_model_exception.msg,
                        "pid": multi_model_exception.pid,
                    }
                elif multi_model_exception.code == 408:
                    cpu_memory_usage = tfs_utils.get_cpu_memory_util()
                    log.info(f"cpu memory usage {cpu_memory_usage}")
                    if cpu_memory_usage > 70:
                        return {
                            "status": falcon.HTTP_507,
                            "body": "Memory exhausted: not enough memory to start TFS instance",
                            "pid": multi_model_exception.pid,
                        }
                    return {
                        "status": falcon.HTTP_408,
                        "body": multi_model_exception.msg,
                        "pid": multi_model_exception.pid,
                    }
                else:
                    return {
                        "status": falcon.HTTP_500,
                        "body": multi_model_exception.msg,
                        "pid": multi_model_exception.pid,
                    }
            except FileExistsError as e:
                return {
                    "status": falcon.HTTP_409,
                    "body": json.dumps(
                        {"error": "Model {} is already loaded. {}".format(model_name, str(e))}
                    ),
                }
            except OSError as os_error:
                log.error(f"failed to load model with exception {os_error}")
                if os_error.errno == 12:
                    return {
                        "status": falcon.HTTP_507,
                        "body": "Memory exhausted: not enough memory to start TFS instance",
                    }
                else:
                    return {
                        "status": falcon.HTTP_500,
                        "body": os_error.strerror,
                    }
        else:
            return {
                "status": falcon.HTTP_404,
                "body": json.dumps(
                    {
                        "error": "Could not find valid base path {} for servable {}".format(
                            base_path, model_name
                        )
                    }
                ),
            }

    def _handle_load_model_post(self, res, data):  # noqa: C901
        with lock():
            model_name = data["model_name"]
            base_path = data["url"]

            # sync sync_local_mme_instance_status & update available ports
            self._sync_local_mme_instance_status()
            self._update_ports_available()

            # model is already loaded
            if model_name in self._mme_tfs_instances_status:
                res.status = falcon.HTTP_409
                res.body = json.dumps({"error": "Model {} is already loaded.".format(model_name)})
                return

            is_load_successful = True
            response = {}
            for i in range(self._tfs_instance_count):
                # check if there are available ports
                if not self._ports_available():
                    is_load_successful = False
                    response["status"] = falcon.HTTP_507
                    response["body"] = json.dumps(
                        {"error": "Memory exhausted: no available ports to load the model."}
                    )
                    break
                tfs_rest_port = self._tfs_available_ports["rest_port"].pop()
                tfs_grpc_port = self._tfs_available_ports["grpc_port"].pop()

                response = self._load_model(model_name, base_path, tfs_rest_port, tfs_grpc_port, i)

                if "pid" in response:
                    self._mme_tfs_instances_status.setdefault(model_name, []).append(
                        TfsInstanceStatus(tfs_rest_port, tfs_grpc_port, response["pid"])
                    )

                if response["status"] != falcon.HTTP_200:
                    log.info(f"Failed to load model : {model_name}")
                    is_load_successful = False
                    break

            if not is_load_successful:
                log.info(f"Failed to load model : {model_name}, Starting to cleanup...")
                self._delete_model(model_name)
                self._remove_model_config(model_name)
            else:
                self._upload_mme_instance_status()

            res.status = response["status"]
            res.body = response["body"]

    def _import_custom_modules(self, model_name):
        inference_script_path = "/opt/ml/models/{}/model/code/inference.py".format(model_name)
        python_lib_path = "/opt/ml/models/{}/model/code/lib".format(model_name)
        if os.path.exists(python_lib_path):
            log.info(
                "Add Python code library for the model {} found at path {}.".format(
                    model_name, python_lib_path
                )
            )
            sys.path.append(python_lib_path)
        else:
            log.info(
                "Python code library for the model {} not found at path {}.".format(
                    model_name, python_lib_path
                )
            )
        if os.path.exists(inference_script_path):
            log.info(
                "Importing handlers from model-specific inference script for the model {} found at path {}.".format(
                    model_name, inference_script_path
                )
            )
            handler, input_handler, output_handler = self._import_handlers(inference_script_path)
            model_handlers = self._make_handler(handler, input_handler, output_handler)
            self.model_handlers[model_name] = model_handlers
        else:
            log.info(
                "Model-specific inference script for the model {} not found at path {}.".format(
                    model_name, inference_script_path
                )
            )

    def _handle_invocation_post(self, req, res, model_name=None):
        if SAGEMAKER_MULTI_MODEL_ENABLED:
            if model_name:
                if self._gunicorn_workers > 1:
                    if model_name not in self._mme_tfs_instances_status or not self._check_pid(
                        self._mme_tfs_instances_status[model_name][0].pid
                    ):
                        with lock():
                            self._sync_local_mme_instance_status()
                        self._import_custom_modules(model_name)

                if model_name not in self._mme_tfs_instances_status:
                    res.status = falcon.HTTP_404
                    res.body = json.dumps(
                        {"error": "Model {} is not loaded yet.".format(model_name)}
                    )
                    return
                else:
                    log.info("model name: {}".format(model_name))
                    rest_ports = [
                        status.rest_port for status in self._mme_tfs_instances_status[model_name]
                    ]
                    rest_port = self._pick_port(rest_ports)
                    log.info("rest port: {}".format(str(rest_port)))
                    grpc_ports = [
                        status.grpc_port for status in self._mme_tfs_instances_status[model_name]
                    ]
                    grpc_port = grpc_ports[rest_ports.index(rest_port)]
                    log.info("grpc port: {}".format(str(grpc_port)))
                    data, context = tfs_utils.parse_request(
                        req,
                        rest_port,
                        grpc_port,
                        self._tfs_default_model_name,
                        model_name=model_name,
                    )
            else:
                res.status = falcon.HTTP_400
                res.body = json.dumps({"error": "Invocation request does not contain model name."})
                return
        else:
            # Randomly pick port used for routing incoming request.
            grpc_port = self._pick_port(self._tfs_grpc_ports)
            rest_port = self._pick_port(self._tfs_rest_ports)
            data, context = tfs_utils.parse_request(
                req,
                rest_port,
                grpc_port,
                self._tfs_default_model_name,
                channel=self._channels[grpc_port],
            )

        try:
            res.status = falcon.HTTP_200
            handlers = self._handlers
            if SAGEMAKER_MULTI_MODEL_ENABLED and model_name in self.model_handlers:
                log.info(
                    "Model-specific inference script for the model {} exists, importing handlers.".format(
                        model_name
                    )
                )
                handlers = self.model_handlers[model_name]
            elif not self._default_handlers_enabled:
                log.info(
                    "Universal inference script exists at path {}, importing handlers.".format(
                        INFERENCE_SCRIPT_PATH
                    )
                )
            else:
                log.info(
                    "Model-specific inference script and universal inference script both do not exist, using default handlers."
                )
            res.body, res.content_type = handlers(data, context)
        except Exception as e:  # pylint: disable=broad-except
            log.exception("exception handling request: {}".format(e))
            res.status = falcon.HTTP_500
            res.body = json.dumps({"error": str(e)}).encode("utf-8")  # pylint: disable=E1101

    def _setup_channel(self, grpc_port):
        if grpc_port not in self._channels:
            log.info("Creating grpc channel for port: %s", grpc_port)
            self._channels[grpc_port] = grpc.insecure_channel("localhost:{}".format(grpc_port))

    def _import_handlers(self, inference_script=INFERENCE_SCRIPT_PATH):
        spec = importlib.util.spec_from_file_location("inference", inference_script)
        inference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inference)

        _custom_handler, _custom_input_handler, _custom_output_handler = None, None, None
        if hasattr(inference, "handler"):
            _custom_handler = inference.handler
        elif hasattr(inference, "input_handler") and hasattr(inference, "output_handler"):
            _custom_input_handler = inference.input_handler
            _custom_output_handler = inference.output_handler
        else:
            raise NotImplementedError("Handlers are not implemented correctly in user script.")

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
        with lock():
            self._sync_local_mme_instance_status()
            if model_name is None:
                models_info = {}
                uri = "http://localhost:{}/v1/models/{}"
                for model, tfs_instance_status in self._mme_tfs_instances_status.items():
                    try:
                        info = json.loads(
                            requests.get(
                                uri.format(tfs_instance_status[0].rest_port, model)
                            ).content
                        )
                        models_info[model] = info
                    except ValueError as e:
                        log.exception("exception handling request: {}".format(e))
                        res.status = falcon.HTTP_500
                        res.body = json.dumps({"error": str(e)}).encode("utf-8")
                res.status = falcon.HTTP_200
                res.body = json.dumps(models_info)
            else:
                if model_name not in self._mme_tfs_instances_status:
                    res.status = falcon.HTTP_404
                    res.body = json.dumps(
                        {"error": "Model {} is loaded yet.".format(model_name)}
                    ).encode("utf-8")
                else:
                    port = self._mme_tfs_instances_status[model_name].rest_port
                    uri = "http://localhost:{}/v1/models/{}".format(port, model_name)
                    try:
                        info = requests.get(uri)
                        res.status = falcon.HTTP_200
                        res.body = json.dumps({"model": info}).encode("utf-8")
                    except ValueError as e:
                        log.exception("exception handling GET models request.")
                        res.status = falcon.HTTP_500
                        res.body = json.dumps({"error": str(e)}).encode("utf-8")

    def on_delete(self, req, res, model_name):  # pylint: disable=W0613
        with lock():
            self._sync_local_mme_instance_status()
            if model_name not in self._mme_tfs_instances_status:
                res.status = falcon.HTTP_404
                res.body = json.dumps({"error": "Model {} is not loaded yet".format(model_name)})
            else:
                try:
                    self._delete_model(model_name)
                    self._remove_model_config(model_name)
                    del self._mme_tfs_instances_status[model_name]
                    self._upload_mme_instance_status()
                    res.status = falcon.HTTP_200
                    res.body = json.dumps(
                        {"success": "Successfully unloaded model {}.".format(model_name)}
                    )
                except OSError as error:
                    res.status = falcon.HTTP_500
                    res.body = json.dumps({"error": str(error)}).encode("utf-8")

    def _delete_model(self, model_name):
        if model_name not in self._mme_tfs_instances_status:
            return
        for tfs_status in self._mme_tfs_instances_status[model_name]:
            os.kill(tfs_status.pid, signal.SIGKILL)

    def _remove_model_config(self, model_name):
        shutil.rmtree("/sagemaker/tfs-config/{}".format(model_name), ignore_errors=True)
        shutil.rmtree("/sagemaker/batching/{}".format(model_name), ignore_errors=True)

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
        if not versions:
            return False
        for v in versions:
            if v.isdigit():
                # TensorFlow model server will succeed with any versions found
                # even if there are directories that's not a valid model version,
                # the loading will succeed.
                return True
        return False

    def _upload_mme_instance_status(self):
        log.info(
            "uploaded mme instance status file with content: {}".format(
                self._mme_tfs_instances_status
            )
        )
        with open(MME_TFS_INSTANCE_STATUS_FILE, "wb") as handle:
            pickle.dump(self._mme_tfs_instances_status, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _sync_local_mme_instance_status(self):
        if not os.path.exists(MME_TFS_INSTANCE_STATUS_FILE):
            log.info("mme instance status file does not found.")
            return
        with open(MME_TFS_INSTANCE_STATUS_FILE, "rb") as handle:
            self._mme_tfs_instances_status = pickle.load(handle)
        log.info(
            "updated local mme instance status with content: {}".format(
                self._mme_tfs_instances_status
            )
        )

    def _check_pid(self, pid):
        """Check For the existence of a unix pid."""
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        else:
            return True


class PingResource:
    def on_get(self, req, res):  # pylint: disable=W0613
        res.status = falcon.HTTP_200


class ServiceResources:
    def __init__(self):
        self._enable_model_manager = SAGEMAKER_MULTI_MODEL_ENABLED
        self._python_service_resource = PythonServiceResource()
        self._ping_resource = PingResource()

    def add_routes(self, application):
        application.add_route("/ping", self._ping_resource)
        application.add_route("/invocations", self._python_service_resource)

        if self._enable_model_manager:
            application.add_route("/models", self._python_service_resource)
            application.add_route("/models/{model_name}", self._python_service_resource)
            application.add_route("/models/{model_name}/invoke", self._python_service_resource)


app = falcon.API()
resources = ServiceResources()
resources.add_routes(app)

if __name__ == "__main__":
    # Define the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--bind", type=str, required=True, help="Specify a server socket to bind."
    )
    parser.add_argument(
        "-k",
        "--worker-class",
        type=str,
        required=True,
        choices=["sync", "eventlet", "gevent", "tornado", "gthread", "sync"],
        help="The type of worker process to run",
    )
    parser.add_argument("-c", "--chdir", type=str, required=True, help="Change root dir")
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        required=True,
        help="The number of worker processes. This number should generally be between 2-4 workers per core in the server.",
    )
    parser.add_argument("-t", "--threads", type=int, required=True, help="The number of threads")
    parser.add_argument("-l", "--log-level", type=str, required=True)
    parser.add_argument("-o", "--timeout", type=int, required=True, help="Gunicorn timeout")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Create gunicorn options
    options = {
        "bind": args.bind,
        "worker_class": args.worker_class,
        "chdir": args.chdir,
        "workers": args.workers,
        "threads": args.threads,
        "loglevel": args.log_level,
        "timeout": args.timeout,
        "raw_env": [
            f"TFS_GRPC_PORTS={TFS_GRPC_PORTS}",
            f"TFS_REST_PORTS={TFS_REST_PORTS}",
            f'SAGEMAKER_MULTI_MODEL={os.environ.get("SAGEMAKER_MULTI_MODEL")}',
            f"SAGEMAKER_SAFE_PORT_RANGE={SAGEMAKER_TFS_PORT_RANGE}",
            f'SAGEMAKER_TFS_WAIT_TIME_SECONDS={os.environ.get("SAGEMAKER_TFS_WAIT_TIME_SECONDS")}',
            f'SAGEMAKER_TFS_INTER_OP_PARALLELISM={os.environ.get("SAGEMAKER_TFS_INTER_OP_PARALLELISM", 0)}',
            f'SAGEMAKER_TFS_INTRA_OP_PARALLELISM={os.environ.get("SAGEMAKER_TFS_INTRA_OP_PARALLELISM", 0)}',
            f'SAGEMAKER_TFS_INSTANCE_COUNT={os.environ.get("SAGEMAKER_TFS_INSTANCE_COUNT", "1")}',
            f'SAGEMAKER_GUNICORN_WORKERS={os.environ.get("SAGEMAKER_GUNICORN_WORKERS", "1")}',
        ],
    }

    from gunicorn.app.base import BaseApplication

    class StandaloneApplication(BaseApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super().__init__()

        def load_config(self):
            config = {
                key: value
                for key, value in self.options.items()
                if key in self.cfg.settings and value is not None
            }
            for key, value in config.items():
                self.cfg.set(key.lower(), value)

        def load(self):
            return self.application

    StandaloneApplication(app, options).run()
