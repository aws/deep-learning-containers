#!/usr/bin/env python3
"""
Sanity tests for the Hugging Face-contributed vLLM DLC images.

Scoped to the huggingface-vllm family and invoked only by
.github/workflows/huggingface-vllm.pipeline.yml — these are not part of the
shared vLLM/SGLang sanity suite.

Run inside the container:
    docker run --rm -v $(pwd):/workdir --workdir /workdir \
        --entrypoint python3 <image> \
        test/sanity/scripts/test_sanity_huggingface_vllm.py

The image is identified by its HF entrypoint; every test skips cleanly when the
container is not a Hugging Face vLLM SageMaker image.
"""

import os
import signal
import subprocess
import unittest

ENTRYPOINT = "/usr/local/bin/sagemaker_entrypoint.sh"
OPTIMIZATIONS = "/usr/local/bin/hf_optimizations.sh"


def _is_hf_vllm_image():
    """True when this container is a Hugging Face vLLM SageMaker image."""
    if not os.path.isfile(ENTRYPOINT) or not os.path.isfile(OPTIMIZATIONS):
        return False
    with open(ENTRYPOINT) as f:
        return "SM_VLLM_" in f.read()


class TestServerStartup(unittest.TestCase):
    """The server must survive its own startup path.

    Guards the failure class where something in the server's module graph is
    imported eagerly and cannot load in this image — a package that is absent,
    or a native extension whose shared libraries this image broke. The server
    then crash-loops for *every* workload, not only the feature that needs the
    dependency, and a SageMaker endpoint never reaches InService.

    The motivating regression: vLLM 0.25.0 imported torchcodec at server-module
    import time; torchcodec dlopen's libtorchcodec_core*.so, which links
    libavutil.so.*; this image built FFmpeg from source without --enable-shared,
    so those libs did not exist. Text-only serving died at startup even though
    nothing in the request path touched video.

    Failures are asserted by the *absence* of import/link signatures rather than
    the presence of an expected error, so the checks do not rot when upstream
    error messages change. That is also what makes this runnable on the GPU-less
    sanity runner: the module graph loads before any device is needed, and the
    process is then expected to die in engine init (vLLM raises "Failed to infer
    device type" once it builds a VllmConfig).
    """

    FATAL_PATTERNS = (
        "ModuleNotFoundError",
        "ImportError",
        "cannot open shared object file",
        "undefined symbol",
    )

    STARTUP_TIMEOUT = 600

    def setUp(self):
        if not _is_hf_vllm_image():
            self.skipTest("not a Hugging Face vLLM SageMaker image")

    def _assert_loadable(self, output, context):
        hits = [p for p in self.FATAL_PATTERNS if p in output]
        self.assertEqual(
            hits,
            [],
            f"{context} hit {hits} — the server cannot load its own code, so it will "
            f"crash-loop for every model:\n{output[-4000:]}",
        )

    def test_entrypoint_startup_loads_all_code(self):
        """The real SageMaker entrypoint must not die on unloadable code.

        Invokes the entrypoint the way SageMaker does. This exercises what a
        synthetic import cannot: hf_optimizations.sh, the SM_VLLM_* to CLI
        translation, and the injected --kv-transfer-config and --middleware.

        ``PROCESS_AUTO_RECOVERY=false`` keeps the run bounded. standard-supervisor
        always launches the server under supervisord — the flag only sets
        ``autorestart=false`` — so without it a startup failure would be retried
        PROCESS_MAX_START_RETRIES times before the process gave up.

        Without an accelerator the process is expected to fail in engine/platform
        init, so the assertion is on *how* it fails, not whether it survives.
        """
        model_dir = "/tmp/hf_sanity_startup_model"
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            f.write("{}")

        env = {k: v for k, v in os.environ.items() if not k.startswith("SM_VLLM_")}
        env["PROCESS_AUTO_RECOVERY"] = "false"
        env["SM_VLLM_MODEL"] = model_dir

        proc = subprocess.Popen(
            [ENTRYPOINT],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            output = proc.communicate(timeout=self.STARTUP_TIMEOUT)[0]
        except subprocess.TimeoutExpired:
            # A server that reaches its listening state never exits, and its
            # children would hold the stdout pipe open — kill the whole group.
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            output = proc.communicate()[0]

        self.assertRegex(
            output.lower(),
            r"vllm",
            f"{ENTRYPOINT} produced no server output, so nothing was verified:\n{output[-4000:]}",
        )
        self._assert_loadable(output, f"{ENTRYPOINT} startup")


if __name__ == "__main__":
    unittest.main(verbosity=2)
