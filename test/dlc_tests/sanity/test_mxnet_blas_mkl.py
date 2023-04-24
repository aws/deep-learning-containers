import pytest

from invoke.context import Context
from packaging.version import Version

from test.test_utils import (
    get_container_name,
    get_framework_and_version_from_tag,
    get_processor_from_image_uri,
    run_cmd_on_container,
    start_container,
    stop_and_remove_container,
)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("N/A")
@pytest.mark.processor("cpu")
@pytest.mark.integration("mxnet_blas_mkl_sanity")
def test_mxnet_blas_mkl_sanity(mxnet_inference, cpu_only):
    """
    Check that the container's version of MXNet includes BLAS MKL.
    :param mxnet_inference: framework fixture for mxnet inference
    """
    _, framework_version = get_framework_and_version_from_tag(mxnet_inference)
    if Version(framework_version) < Version("1.9.0"):
        pytest.skip(
            "Skipping this test MXNet versions less than 1.9.0 which do not use BLAS MKL."
        )

    ctx = Context()
    container_name = get_container_name("mxnet-blas-mkl", mxnet_inference)
    start_container(container_name, mxnet_inference, ctx)

    output = run_cmd_on_container(
        container_name,
        ctx,
        'import mxnet; assert mxnet.runtime.Features().is_enabled("BLAS_MKL") == True',
        executable="python",
    )

    try:
        # If BLAS MKL is enabled, output should be blank. Otherwise, assertion errors will be raised.
        assert f"{output.stdout}" == ""
    finally:
        stop_and_remove_container(container_name, ctx)
