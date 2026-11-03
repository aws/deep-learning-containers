"""Unit tests for deep_learning_container.py, the telemetry script every image runs.

The image test runs the script on one mock image, as one framework on one device.
These tests cover the branches that run elsewhere: every registered framework, the
framework-specific version handling, each device, and the exact URL and tag the
script reports.
"""

import io
import json
import os
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import deep_learning_container as dlc
import pytest
import select_telemetry_tests as stt

REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTRY = stt.parse_registry((REPO_ROOT / stt.DLC_SCRIPT).read_text())
INSTANCE_ID = "i-0123456789abcdef0"
PY_VERSION = sys.version.split(" ")[0]
DEVICE_PATHS = {"/usr/local/cuda", "/opt/ei_tools", "/usr/local/bin/tensorflow_model_server_neuron"}


@pytest.fixture(autouse=True)
def no_test_mode(monkeypatch):
    # TEST_MODE makes the script write its requests to /tmp.
    monkeypatch.delenv("TEST_MODE", raising=False)


@pytest.fixture
def argv(monkeypatch):
    def set_args(framework, framework_version="", container_type="general", *extra):
        args = ["--framework", framework, "--container-type", container_type]
        if framework_version:
            args += ["--framework-version", framework_version]
        monkeypatch.setattr(sys, "argv", ["deep_learning_container.py", *args, *extra])

    return set_args


@pytest.fixture
def device_paths(monkeypatch):
    """Make exactly the given device marker paths exist."""
    real_isdir, real_exists = os.path.isdir, os.path.exists

    def present(*paths):
        monkeypatch.setattr(
            os.path, "isdir", lambda p: p in paths if p in DEVICE_PATHS else real_isdir(p)
        )
        monkeypatch.setattr(
            os.path, "exists", lambda p: p in paths if p in DEVICE_PATHS else real_exists(p)
        )

    return present


# --------------------------------------------------------------------------- #
#                                 parse_args                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("framework", REGISTRY.frameworks)
def test_every_registered_framework_is_reported_as_is(argv, framework):
    argv(framework, "1.2.3")
    args = dlc.parse_args()
    assert (args.framework, args.framework_version) == (framework, "1.2.3")


@pytest.mark.parametrize("container_type", ["training", "inference", "general"])
def test_every_container_type_is_accepted(argv, container_type):
    argv("base", "1.0.0", container_type)
    assert dlc.parse_args().container_type == container_type


@pytest.mark.parametrize(
    "version, reported",
    [
        pytest.param("2.11.0+cu130", "2.11.0", id="cuda-build"),
        pytest.param("2.11.0+cpu", "2.11.0", id="cpu-build"),
        pytest.param("2.6.0a0+git1234567", "2.6.0", id="source-build"),
        pytest.param("2.11.0", "2.11.0", id="release"),
    ],
)
def test_pytorch_version_drops_the_build_suffix(argv, version, reported):
    argv("pytorch", version)
    assert dlc.parse_args().framework_version == reported


@pytest.mark.parametrize(
    "container_type, reported",
    [("inference", "2.12.1"), ("training", "2.12.0")],
)
def test_tensorflow_serving_2_12_reports_its_own_version(argv, container_type, reported):
    # TFS 2.12.1 ships TF 2.12.0, and the image reports the TF version.
    argv("tensorflow", "2.12.0", container_type)
    assert dlc.parse_args().framework_version == reported


def test_framework_version_may_be_empty(argv):
    argv("base")
    assert dlc.parse_args().framework_version == ""


def test_invalid_framework_version_is_rejected(argv):
    argv("base", "not-a-version")
    with pytest.raises(AssertionError, match="PEP 440"):
        dlc.parse_args()


def test_unregistered_framework_is_rejected(argv):
    argv("not-a-dlc", "1.0.0")
    with pytest.raises(SystemExit):
        dlc.parse_args()


def test_unknown_arguments_are_ignored(argv):
    argv("base", "1.0.0", "general", "--future-flag", "x")
    assert dlc.parse_args().framework == "base"


# --------------------------------------------------------------------------- #
#                         Device, CUDA, and OS probes                         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "paths, device",
    [
        pytest.param({"/usr/local/cuda", "/opt/ei_tools"}, "gpu", id="gpu"),
        pytest.param({"/opt/ei_tools"}, "eia", id="eia"),
        pytest.param({"/usr/local/bin/tensorflow_model_server_neuron"}, "neuron", id="neuron"),
        pytest.param(set(), "cpu", id="cpu"),
    ],
)
def test_device_is_detected_from_marker_paths(device_paths, paths, device):
    device_paths(*paths)
    assert dlc._retrieve_device() == device


@pytest.mark.parametrize(
    "target, version",
    [("/usr/local/cuda-13.0", "13.0"), ("cuda-12.9", "12.9"), ("/opt/cuda", "")],
)
def test_cuda_version_comes_from_the_cuda_link(monkeypatch, target, version):
    monkeypatch.setattr(os, "readlink", lambda p: target)
    assert dlc._retrieve_cuda() == version


def test_cuda_version_is_empty_without_the_cuda_link(monkeypatch):
    def missing(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(os, "readlink", missing)
    assert dlc._retrieve_cuda() == ""


def test_os_comes_from_os_release(monkeypatch):
    os_release = 'NAME="Ubuntu"\nID=ubuntu\nVERSION_ID="22.04"\n'
    real_open = open
    monkeypatch.setattr(
        dlc,
        "open",
        lambda path, *a: io.StringIO(os_release)
        if path == "/etc/os-release"
        else real_open(path, *a),
        raising=False,
    )
    assert dlc._retrieve_os() == "ubuntu22.04"


# --------------------------------------------------------------------------- #
#                         What the script reports                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("region", ["us-west-2", "me-west-1"])
def test_instance_region_is_kept_only_when_mapped(monkeypatch, region):
    document = json.dumps({"region": region})
    monkeypatch.setattr(dlc, "requests_helper_imds", lambda url, token=None: document)
    assert dlc._retrieve_instance_region("token") == region


def test_unmapped_instance_region_disables_telemetry(monkeypatch):
    document = json.dumps({"region": "xx-nowhere-1"})
    monkeypatch.setattr(dlc, "requests_helper_imds", lambda url, token=None: document)
    assert dlc._retrieve_instance_region("token") is None


@pytest.mark.parametrize(
    "instance_id, valid",
    [(INSTANCE_ID, INSTANCE_ID), ("i-short", None), ("not-an-instance", None)],
)
def test_instance_id_is_validated(instance_id, valid):
    assert dlc._validate_instance_id(instance_id) == valid


def test_query_bucket_url(monkeypatch, argv):
    urls = []
    monkeypatch.setattr(dlc, "requests_helper", lambda url, timeout: urls.append(url))
    argv("huggingface-pytorch-training", "2.11.0", "training")
    dlc.query_bucket(INSTANCE_ID, "us-west-2")

    (url,) = urls
    parsed, params = urlparse(url), parse_qs(urlparse(url).query)
    bucket = dlc.REGION_MAPPING["us-west-2"]
    assert parsed.hostname == f"aws-deep-learning-containers-{bucket}.s3.us-west-2.amazonaws.com"
    assert parsed.path == f"/dlc-containers-{INSTANCE_ID}.txt"
    assert params == {
        "x-instance-id": [INSTANCE_ID],
        "x-framework": ["huggingface-pytorch-training"],
        "x-framework_version": ["2.11.0"],
        "x-py_version": [PY_VERSION],
        "x-container_type": ["training"],
    }


def test_query_bucket_skips_without_instance_or_region(monkeypatch, argv):
    urls = []
    monkeypatch.setattr(dlc, "requests_helper", lambda url, timeout: urls.append(url))
    argv("base", "1.0.0")
    dlc.query_bucket(None, "us-west-2")
    dlc.query_bucket(INSTANCE_ID, None)
    assert not urls


class FakeEC2:
    def __init__(self):
        self.calls = []

    def create_tags(self, Resources, Tags):
        self.calls.append((Resources, Tags))
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}


@pytest.fixture
def ec2(monkeypatch):
    client = FakeEC2()

    class Session:
        def create_client(self, service, region_name):
            assert (service, region_name) == ("ec2", "us-west-2")
            return client

    monkeypatch.setattr(dlc.botocore.session, "get_session", Session)
    monkeypatch.setattr(dlc, "_retrieve_os", lambda: "ubuntu22.04")
    monkeypatch.setattr(dlc, "_retrieve_cuda", lambda: "13.0")
    return client


@pytest.mark.parametrize(
    "device, suffix",
    [
        ("gpu", "gpu_cuda13.0_ubuntu22.04"),
        ("cpu", "cpu_ubuntu22.04"),
        ("neuron", "neuron_ubuntu22.04"),
    ],
)
def test_instance_tag(monkeypatch, argv, ec2, device, suffix):
    monkeypatch.setattr(dlc, "_retrieve_device", lambda: device)
    argv("vllm", "0.31.1", "general")
    assert dlc.tag_instance(INSTANCE_ID, "us-west-2") == 200

    (call,) = ec2.calls
    assert call == (
        [INSTANCE_ID],
        [
            {
                "Key": "aws-dlc-autogenerated-tag-do-not-delete",
                "Value": f"vllm_general_0.31.1_python{PY_VERSION}_{suffix}",
            }
        ],
    )


def test_instance_tag_skips_without_instance_or_region(monkeypatch, argv, ec2):
    monkeypatch.setattr(dlc, "_retrieve_device", lambda: "cpu")
    argv("base", "1.0.0")
    assert dlc.tag_instance(None, "us-west-2") is None
    assert dlc.tag_instance(INSTANCE_ID, None) is None
    assert not ec2.calls
