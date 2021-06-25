import os
import json

from packaging.version import Version

import pytest

from invoke.context import Context

from test.test_utils import is_mainline_context, get_container_name


@pytest.mark.integration("smprofiler")
@pytest.mark.model("N/A")
@pytest.mark.skipif(not is_mainline_context(), reason="Mainline only test")
def test_sm_profiler_pt(pytorch_training):

    ctx = Context()
    profiler_tests_dir = os.path.join(os.getenv('CODEBUILD_SRC_DIR'), get_container_name('smprof', pytorch_training), 'smprofiler_tests')
    with ctx.prefix(f"cd {profiler_tests_dir}/scripts/"):
        ctx.run("mkdir -p data")

    run_sm_profiler_tests(pytorch_training, profiler_tests_dir, "test_profiler_pytorch.py")


@pytest.mark.integration("smprofiler")
@pytest.mark.model("N/A")
@pytest.mark.skipif(not is_mainline_context(), reason="Mainline only test")
def test_sm_profiler_tf(tensorflow_training):
    run_sm_profiler_tests(tensorflow_training, "test_profiler_tensorflow.py")


def run_sm_profiler_tests(image, profiler_tests_dir, test_file):
    ctx = Context()
    smdebug_version = ctx.run(f"docker run {image} python -c 'import smdebug; print(smdebug.__version__)'").stdout.strip()

    if Version(smdebug_version) < Version("1"):
        pytest.skip(f"smdebug version {smdebug_version} is less than 1, so smprofiler not expected to be present")
    sm_tests_zip = "sagemaker-tests.zip"
    ctx.run(f'aws s3 cp s3://smprofiler-test-artifacts/{sm_tests_zip} {profiler_tests_dir}/{sm_tests_zip}')

    test_results_outfile = f"{get_container_name('smprof', image)}.txt"
    with ctx.prefix(f'cd {profiler_tests_dir}'):
        ctx.run(f'unzip {sm_tests_zip}')
        with ctx.prefix(f"cd {sm_tests_zip.strip('.zip')}/tests"):
            test_output = ctx.run(f"pytest --json-report --json-report-file={test_results_outfile} -n=auto -v -s -W=ignore {test_file}", warn=True)
        with open(test_results_outfile) as outfile:
            result_data = json.load(outfile)
        if test_output.return_code != 0:
            pytest.fail(f"Failed SM Profiler tests. Results: {result_data}")

