import pytest
from invoke.context import Context


@pytest.mark.integration("Generating this coverage doc")
def test_generate_coverage_doc():
    """
    Test generating the test coverage doc
    """
    ctx = Context()
    ctx.run("export DLC_TESTS='test' && pytest --collect-only  --generate-coverage-doc --ignore=container_tests/")
