"""TF training image GPU tests on an EC2 G7 (RTX PRO 4500 Blackwell, sm_120).

Runs the image's single-GPU and unit suites on a g7.2xlarge (see conftest.py)
with CUDA_DISABLE_PTX_JIT=1, so any kernel without native sm_120 code fails
instead of being JIT-compiled from PTX. The CI runner drives the instance over
SSH; the tests themselves run inside the image on the instance.
"""

import os
import shlex
import subprocess
import tempfile

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
WORKDIR = "/home/ec2-user/workdir"
CONTAINER = "tf-blackwell"
NATIVE_ONLY = "CUDA_DISABLE_PTX_JIT=1"

# Inside the image: TensorFlow must see the GPU at compute capability 12.0 and
# run conv, matmul and generated elementwise kernels (AddV2, Rsqrt) on it.
GPU_CHECK = """
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
assert gpus, "TensorFlow sees no GPU"
details = tf.config.experimental.get_device_details(gpus[0])
print("GPU:", details)
assert details.get("compute_capability") == (12, 0), details
with tf.device("/GPU:0"):
    x = tf.random.normal([8, 32, 32, 16])
    k = tf.random.normal([3, 3, 16, 32])
    y = tf.nn.conv2d(x, k, strides=1, padding="SAME")
    z = tf.math.rsqrt(tf.abs(y) + 1.0) + tf.matmul(tf.ones([4, 4]), tf.ones([4, 4]))[0, 0]
    print("conv/rsqrt/matmul OK on", z.device, float(tf.reduce_sum(z)))
"""


def _run(conn, command, **kwargs):
    result = conn.run(command, warn=True, **kwargs)
    assert result.ok, (
        f"exit {result.exited}: {command}\n{result.stdout[-4000:]}{result.stderr[-4000:]}"
    )
    return result


def _exec(conn, command):
    env = " ".join(
        f"-e {name}={shlex.quote(os.environ.get(name, ''))}"
        for name in ("DLC_TENSORFLOW_VERSION", "DLC_TENSORFLOW_FULL_VERSION")
    )
    return _run(
        conn,
        f"docker exec -e {NATIVE_ONLY} -e PYTHONPATH=/workdir/test -e DLC_WORKDIR=/workdir "
        f"{env} {CONTAINER} {command}",
    )


@pytest.fixture(scope="session")
def container(conn, image_uri, region):
    """Copy test/ to the instance, pull the image and start it with all GPUs."""
    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, "test.tgz")
        subprocess.run(
            ["tar", "czf", archive, "--exclude=__pycache__", "test"], cwd=REPO_ROOT, check=True
        )
        conn.put(archive, "/tmp/test.tgz")
    _run(conn, f"mkdir -p {WORKDIR} && tar xzf /tmp/test.tgz -C {WORKDIR}")
    registry = image_uri.split("/")[0]
    _run(
        conn,
        f"aws ecr get-login-password --region {region} | "
        f"docker login --username AWS --password-stdin {registry}",
    )
    _run(conn, f"docker pull {image_uri}", hide=True)
    _run(
        conn,
        f"docker run -d --rm --name {CONTAINER} --gpus all --shm-size=2g --entrypoint /bin/bash "
        f"-v {WORKDIR}:/workdir --workdir /workdir {image_uri} -c 'sleep infinity'",
    )
    _exec(conn, "pip install -q -r /workdir/test/requirements.txt")
    yield CONTAINER
    conn.run(f"docker rm -f {CONTAINER}", warn=True)


def test_host_has_blackwell_gpu(conn):
    out = _run(
        conn, "nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader"
    ).stdout
    print(out)
    assert "Blackwell" in out and "12.0" in out, out


def test_tensorflow_runs_on_blackwell_without_ptx_jit(conn, container):
    script = shlex.quote(GPU_CHECK)
    print(_exec(conn, f"python -c {script}").stdout)


def test_single_gpu_suite(conn, container):
    print(_exec(conn, "pytest -v -rA /workdir/test/tensorflow/single_gpu/").stdout[-6000:])


def test_unit_suite(conn, container):
    print(_exec(conn, "pytest -v -rA /workdir/test/tensorflow/unit/").stdout[-6000:])
