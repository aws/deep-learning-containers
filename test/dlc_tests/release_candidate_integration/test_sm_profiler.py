import os
import json
import time

from packaging.version import Version

import pytest

from invoke.context import Context
from invoke.exceptions import UnexpectedExit

from test.test_utils import (
    is_mainline_context,
    is_rc_test_context,
    get_framework_and_version_from_tag,
    get_container_name,
    get_processor_from_image_uri,
    is_tf_version,
    LOGGER,
)


@pytest.mark.usefixtures("sagemaker_only", "huggingface")
@pytest.mark.integration("smprofiler")
@pytest.mark.model("N/A")
@pytest.mark.skipif(not is_mainline_context() and not is_rc_test_context(), reason="Mainline only test")
def test_sm_profiler_pt(pytorch_training):
    processor = get_processor_from_image_uri(pytorch_training)
    if processor not in ("cpu", "gpu"):
        pytest.skip(f"Processor {processor} not supported. Skipping test.")

    ctx = Context()

    profiler_tests_dir = os.path.join(
        os.getenv("CODEBUILD_SRC_DIR"), get_container_name("smprof", pytorch_training), "smprofiler_tests"
    )
    ctx.run(f"mkdir -p {profiler_tests_dir}", hide=True)

    # Download sagemaker-tests zip
    sm_tests_zip = "sagemaker-tests.zip"
    ctx.run(
        f"aws s3 cp {os.getenv('SMPROFILER_TESTS_BUCKET')}/{sm_tests_zip} {profiler_tests_dir}/{sm_tests_zip}",
        hide=True,
    )

    # PT test setup requirements
    with ctx.prefix(f"cd {profiler_tests_dir}"):
        ctx.run(f"unzip {sm_tests_zip}", hide=True)
        with ctx.prefix("cd sagemaker-tests/tests/scripts/pytorch_scripts"):
            ctx.run("mkdir -p data", hide=True)
            ctx.run(
                "aws s3 cp s3://smdebug-testing/datasets/cifar-10-python.tar.gz data/cifar-10-batches-py.tar.gz",
                hide=True,
            )
            ctx.run("aws s3 cp s3://smdebug-testing/datasets/MNIST_pytorch.tar.gz data/MNIST_pytorch.tar.gz", hide=True)
            with ctx.prefix("cd data"):
                ctx.run("tar -zxf MNIST_pytorch.tar.gz", hide=True)
                ctx.run("tar -zxf cifar-10-batches-py.tar.gz", hide=True)

    run_sm_profiler_tests(pytorch_training, profiler_tests_dir, "test_profiler_pytorch.py", processor)


@pytest.mark.usefixtures("sagemaker_only", "huggingface")
@pytest.mark.integration("smprofiler")
@pytest.mark.model("N/A")
@pytest.mark.skipif(not is_mainline_context() and not is_rc_test_context(), reason="Mainline only test")
def test_sm_profiler_tf(tensorflow_training):
    if is_tf_version("1", tensorflow_training):
        pytest.skip("Skipping test on TF1, since there are no smprofiler config files for TF1")
    processor = get_processor_from_image_uri(tensorflow_training)
    if processor not in ("cpu", "gpu"):
        pytest.skip(f"Processor {processor} not supported. Skipping test.")

    ctx = Context()

    profiler_tests_dir = os.path.join(
        os.getenv("CODEBUILD_SRC_DIR"), get_container_name("smprof", tensorflow_training), "smprofiler_tests"
    )
    ctx.run(f"mkdir -p {profiler_tests_dir}", hide=True)

    # Download sagemaker-tests zip
    sm_tests_zip = "sagemaker-tests.zip"
    ctx.run(
        f"aws s3 cp {os.getenv('SMPROFILER_TESTS_BUCKET')}/{sm_tests_zip} {profiler_tests_dir}/{sm_tests_zip}",
        hide=True
    )
    ctx.run(f"cd {profiler_tests_dir} && unzip {sm_tests_zip}", hide=True)

    # Install tf datasets
    ctx.run(
        f"echo 'tensorflow-datasets==4.0.1' >> "
        f"{profiler_tests_dir}/sagemaker-tests/tests/scripts/tf_scripts/requirements.txt",
        hide=True,
    )

    run_sm_profiler_tests(tensorflow_training, profiler_tests_dir, "test_profiler_tensorflow.py", processor)


class SMProfilerRCTestFailure(Exception):
    pass


def run_sm_profiler_tests(image, profiler_tests_dir, test_file, processor):
    """
    Testrunner to execute SM profiler tests from DLC repo
    """
    ctx = Context()

    # Install profiler requirements only once - pytest-rerunfailures has a known issue
    # with the latest pytest https://github.com/pytest-dev/pytest-rerunfailures/issues/128
    try:
        ctx.run(
            "pip install -r "
            "https://raw.githubusercontent.com/awslabs/sagemaker-debugger/master/config/profiler/requirements.txt && "
            "pip install smdebug && "
            "pip uninstall -y pytest-rerunfailures",
            hide=True,
        )
    except UnexpectedExit:
        # Wait a minute and a half if we get an invoke failure - since smprofiler test requirements can be flaky
        time.sleep(90)

    framework, version = get_framework_and_version_from_tag(image)

    # Conditionally set sm data parallel tests, based on config file rules from link below:
    # https://github.com/awslabs/sagemaker-debugger/tree/master/config/profiler
    enable_sm_data_parallel_tests = "true"
    if framework == "pytorch" and Version(version) < Version("1.6"):
        enable_sm_data_parallel_tests = "false"
    if framework == "tensorflow" and Version(version) < Version("2.3"):
        enable_sm_data_parallel_tests = "false"

    # Set SMProfiler specific environment variables
    smprof_configs = {
        "use_current_branch": "false",
        "enable_smdataparallel_tests": enable_sm_data_parallel_tests,
        "force_run_tests": "false",
        "framework": framework,
        "build_type": "release"
    }

    # Command to set all necessary environment variables
    export_cmd = " && ".join(f"export {key}={val}" for key, val in smprof_configs.items())
    export_cmd = f"{export_cmd} && export ENV_CPU_TRAIN_IMAGE=test && export ENV_GPU_TRAIN_IMAGE=test && " \
                 f"export ENV_{processor.upper()}_TRAIN_IMAGE={image}"

    test_results_outfile = os.path.join(os.getcwd(), f"{get_container_name('smprof', image)}.txt")
    with ctx.prefix(f"cd {profiler_tests_dir}"):
        with ctx.prefix(f"cd sagemaker-tests && {export_cmd}"):
            try:
                ctx.run(
                    f"pytest --json-report --json-report-file={test_results_outfile} -n=auto "
                    f"-v -s -W=ignore tests/{test_file}::test_{processor}_jobs",
                    hide=True,
                )
                with open(test_results_outfile) as outfile:
                    result_data = json.load(outfile)
                    LOGGER.info(f"Tests passed on {image}; Results:\n{json.dumps(result_data, indent=4)}")
            except Exception as e:
                if os.path.exists(test_results_outfile):
                    with open(test_results_outfile) as outfile:
                        result_data = json.load(outfile)
                    raise SMProfilerRCTestFailure(
                        f"Failed SM Profiler tests. Results:\n{json.dumps(result_data, indent=4)}"
                    ) from e
                raise
