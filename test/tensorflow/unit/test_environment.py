"""TF-specific: Intel MKL env vars for oneDNN/Eigen backend."""

import os


class TestContainerEnv:
    """Verify TF-specific KMP env vars set in the Dockerfile."""

    def test_kmp_affinity(self):
        assert os.environ.get("KMP_AFFINITY") == "granularity=fine,compact,1,0"

    def test_kmp_blocktime(self):
        assert os.environ.get("KMP_BLOCKTIME") == "1"
