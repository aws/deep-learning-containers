"""Container helper — replaces ai_algorithms_container_tests.

Creates /opt/ml/ directory structure in temp dirs, writes config JSON files,
mounts volumes, and runs the container via docker-py.

Training mode: run container to completion, return exit code + logs + model files.
Serving mode:  start container, poll health check, send HTTP requests.
"""

import json
import logging
import os
import shutil
import tempfile
import time

import requests

import docker.types

LOGGER = logging.getLogger(__name__)

TRAIN_TIMEOUT = 300
SERVE_STARTUP_TIMEOUT = 120
HEALTH_CHECK_INTERVAL = 2
SERVE_PORT = 8080


# ---------------------------------------------------------------------------
# /opt/ml layout helpers
# ---------------------------------------------------------------------------

def _create_opt_ml(tmpdir):
    """Create the /opt/ml directory tree inside *tmpdir* and return paths dict."""
    paths = {
        "input_config": os.path.join(tmpdir, "input", "config"),
        "input_train": os.path.join(tmpdir, "input", "data", "train"),
        "input_validation": os.path.join(tmpdir, "input", "data", "validation"),
        "model": os.path.join(tmpdir, "model"),
        "output": os.path.join(tmpdir, "output"),
        "checkpoints": os.path.join(tmpdir, "checkpoints"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def _write_configs(config_dir, hyperparameters, inputdataconfig, resourceconfig,
                   checkpointconfig=None):
    with open(os.path.join(config_dir, "hyperparameters.json"), "w") as f:
        json.dump(hyperparameters, f)
    with open(os.path.join(config_dir, "inputdataconfig.json"), "w") as f:
        json.dump(inputdataconfig, f)
    with open(os.path.join(config_dir, "resourceconfig.json"), "w") as f:
        json.dump(resourceconfig, f)
    if checkpointconfig is not None:
        with open(os.path.join(config_dir, "checkpointconfig.json"), "w") as f:
            json.dump(checkpointconfig, f)


def _copy_files(src_files, dest_dir):
    """Copy a list of files (or all files in a directory) into *dest_dir*."""
    for src in src_files:
        if os.path.isdir(src):
            for fname in os.listdir(src):
                shutil.copy2(os.path.join(src, fname), dest_dir)
        else:
            shutil.copy2(src, dest_dir)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(docker_client, image_uri, hyperparameters, inputdataconfig,
                 resourceconfig, training_files, validation_files=None,
                 checkpointconfig=None, environment=None, timeout=TRAIN_TIMEOUT):
    """Run a training container and return (exit_code, logs, model_files, paths).

    *paths* is the dict returned by ``_create_opt_ml`` so callers can inspect
    checkpoints, model dir, etc.
    """
    tmpdir = tempfile.mkdtemp(prefix="xgb-train-")
    paths = _create_opt_ml(tmpdir)

    _write_configs(paths["input_config"], hyperparameters, inputdataconfig,
                   resourceconfig, checkpointconfig)
    _copy_files(training_files, paths["input_train"])
    if validation_files:
        _copy_files(validation_files, paths["input_validation"])

    volumes = {tmpdir: {"bind": "/opt/ml", "mode": "rw"}}
    env = environment.copy() if environment else {}

    container = docker_client.containers.run(
        image_uri,
        command="train",
        volumes=volumes,
        environment=env,
        detach=True,
    )

    try:
        result = container.wait(timeout=timeout)
        exit_code = result.get("StatusCode", -1)
    except Exception:
        LOGGER.warning("Training did not finish within %ss", timeout)
        exit_code = -1
    finally:
        logs = container.logs().decode("utf-8", errors="replace")
        LOGGER.info("Container logs:\n%s", logs)
        container.remove(force=True)

    model_files = [f for f in os.listdir(paths["model"]) if "model" in f]
    return exit_code, logs, model_files, paths


def run_distributed_training(docker_client, image_uri, hyperparameters, inputdataconfig,
                             resourceconfigs, training_files, timeout=TRAIN_TIMEOUT):
    """Run multi-container distributed training. Returns list of (exit_code, logs, paths)."""
    hosts = [rc["current_host"] for rc in resourceconfigs]
    network_name = "xgb-test-network"
    subnet = "10.5.5.0/24"
    base_ip = 2

    # Create docker network
    try:
        network = docker_client.networks.get(network_name)
        network.remove()
    except Exception:
        pass
    ipam_pool = docker.types.IPAMPool(subnet=subnet)
    ipam_config = docker.types.IPAMConfig(pool_configs=[ipam_pool])
    network = docker_client.networks.create(network_name, driver="bridge", ipam=ipam_config)

    containers = []
    all_paths = []
    try:
        host_ips = {h: f"10.5.5.{base_ip + i}" for i, h in enumerate(hosts)}
        extra_hosts = {h: ip for h, ip in host_ips.items()}

        for i, rc in enumerate(resourceconfigs):
            tmpdir = tempfile.mkdtemp(prefix=f"xgb-dist-{i}-")
            paths = _create_opt_ml(tmpdir)
            _write_configs(paths["input_config"], hyperparameters, inputdataconfig, rc)
            _copy_files(training_files, paths["input_train"])
            all_paths.append(paths)

            volumes = {tmpdir: {"bind": "/opt/ml", "mode": "rw"}}
            networking_config = docker_client.api.create_networking_config({
                network_name: docker_client.api.create_endpoint_config(
                    ipv4_address=host_ips[rc["current_host"]]
                )
            })
            container = docker_client.containers.run(
                image_uri, command="train", volumes=volumes,
                hostname=rc["current_host"],
                extra_hosts=extra_hosts,
                network=network_name,
                detach=True,
            )
            containers.append(container)

        # Wait for all containers
        results = []
        for container in containers:
            try:
                result = container.wait(timeout=timeout)
                exit_code = result.get("StatusCode", -1)
            except Exception:
                exit_code = -1
            logs = container.logs().decode("utf-8", errors="replace")
            results.append((exit_code, logs))
    finally:
        for c in containers:
            try:
                c.remove(force=True)
            except Exception:
                pass
        try:
            network.remove()
        except Exception:
            pass

    return [(r[0], r[1], all_paths[i]) for i, r in enumerate(results)]


# ---------------------------------------------------------------------------
# Serving (inference / batch transform)
# ---------------------------------------------------------------------------

class ServingContainer:
    """Context manager that starts a serving container and exposes HTTP helpers."""

    def __init__(self, docker_client, image_uri, model_dir, environment=None):
        self._client = docker_client
        self._image = image_uri
        self._model_dir = model_dir
        self._env = environment or {}
        self._container = None
        self._host_port = None

    # -- lifecycle -----------------------------------------------------------

    def __enter__(self):
        tmpdir = tempfile.mkdtemp(prefix="xgb-serve-")
        self._opt_ml = tmpdir
        paths = _create_opt_ml(tmpdir)
        # Copy model files
        _copy_files([self._model_dir], paths["model"])
        _write_configs(paths["input_config"], {}, {}, {"current_host": "algo-1", "hosts": ["algo-1"]})

        volumes = {tmpdir: {"bind": "/opt/ml", "mode": "rw"}}
        env = dict(self._env)

        self._container = self._client.containers.run(
            self._image,
            command="serve",
            volumes=volumes,
            environment=env,
            ports={f"{SERVE_PORT}/tcp": None},
            detach=True,
        )
        self._wait_healthy()
        return self

    def __exit__(self, *exc):
        if self._container:
            logs = self._container.logs().decode("utf-8", errors="replace")
            LOGGER.info("Serving container logs:\n%s", logs)
            self._container.remove(force=True)
        if self._opt_ml:
            shutil.rmtree(self._opt_ml, ignore_errors=True)

    # -- health check --------------------------------------------------------

    def _wait_healthy(self):
        deadline = time.time() + SERVE_STARTUP_TIMEOUT
        while time.time() < deadline:
            self._container.reload()
            if self._container.status != "running":
                raise RuntimeError(
                    f"Container exited: {self._container.logs().decode()}"
                )
            try:
                resp = requests.get(self._url("/ping"), timeout=2)
                if resp.status_code == 200:
                    LOGGER.info("Serving container healthy")
                    return
            except (requests.ConnectionError, RuntimeError):
                pass
            time.sleep(HEALTH_CHECK_INTERVAL)
        raise TimeoutError("Serving container did not become healthy")

    # -- HTTP helpers --------------------------------------------------------

    def _url(self, path):
        self._container.reload()
        port_map = self._container.ports.get(f"{SERVE_PORT}/tcp")
        if not port_map:
            raise RuntimeError("No port mapping found")
        self._host_port = int(port_map[0]["HostPort"])
        return f"http://localhost:{self._host_port}{path}"

    def ping(self):
        return requests.get(self._url("/ping"), timeout=5)

    def invocations(self, data, content_type, accept=None):
        headers = {"Content-Type": content_type}
        if accept:
            headers["Accept"] = accept
        return requests.post(
            self._url("/invocations"), data=data, headers=headers, timeout=60
        )

    def execution_parameters(self):
        return requests.get(self._url("/execution-parameters"), timeout=5)

    def get_logs(self):
        if self._container:
            return self._container.logs().decode("utf-8", errors="replace")
        return ""
