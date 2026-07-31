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

import json
import os
import signal
import subprocess
import tempfile
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

    A second class the same launch produces: an argument mangled in transit, so
    the server starts, parses its own flags, and dies on malformed input. Those
    are asserted on the argv the server actually receives.

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

    # Importing torch plus the server graph is slow; capturing argv is not.
    STARTUP_TIMEOUT = 600
    ARGV_TIMEOUT = 120

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

    def _run_entrypoint(self, env_overrides, stub_interpreter=False):
        """Run the real entrypoint; return (server argv, combined output).

        With stub_interpreter, a fake ``python3`` shadows the real one on PATH and
        records the argv it was handed, so the launch chain runs end to end without
        paying for a vLLM import. argv is empty when no stub was installed.
        """
        workdir = tempfile.mkdtemp(prefix="hf_sanity_")
        model_dir = os.path.join(workdir, "model")
        os.makedirs(model_dir)
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            f.write("{}")

        env = {k: v for k, v in os.environ.items() if not k.startswith("SM_VLLM_")}
        env["SM_VLLM_MODEL"] = model_dir
        env.update(env_overrides)

        argv_file = os.path.join(workdir, "argv")
        if stub_interpreter:
            stub = os.path.join(workdir, "python3")
            with open(stub, "w") as f:
                f.write(f'#!/bin/sh\nprintf "%s\\0" "$@" >{argv_file}\n')
            os.chmod(stub, 0o755)
            env["PATH"] = workdir + os.pathsep + env.get("PATH", "")

        proc = subprocess.Popen(
            [ENTRYPOINT],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        timeout = self.ARGV_TIMEOUT if stub_interpreter else self.STARTUP_TIMEOUT
        try:
            output = proc.communicate(timeout=timeout)[0]
        except subprocess.TimeoutExpired:
            # A server that reaches its listening state never exits, and supervisord
            # respawns it — kill the whole group so no child holds the stdout pipe.
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            output = proc.communicate()[0]

        argv = []
        if os.path.isfile(argv_file):
            with open(argv_file) as f:
                argv = f.read().split("\0")[:-1]
        return argv, output

    def test_entrypoint_startup_loads_all_code(self):
        """The real SageMaker entrypoint must not die on unloadable code.

        Invokes the entrypoint the way SageMaker does. This exercises what a
        synthetic import cannot: hf_optimizations.sh, the SM_VLLM_* to CLI
        translation, and the injected --kv-transfer-config and --middleware.

        ``PROCESS_AUTO_RECOVERY=false`` makes standard-supervisor os.execvp() the
        server instead of running it under supervisord, so this is one bounded
        attempt rather than PROCESS_MAX_START_RETRIES of them.

        Without an accelerator the process is expected to fail in engine/platform
        init, so the assertion is on *how* it fails, not whether it survives.
        """
        output = self._run_entrypoint({"PROCESS_AUTO_RECOVERY": "false"})[1]
        self.assertRegex(
            output.lower(),
            r"vllm",
            f"{ENTRYPOINT} produced no server output, so nothing was verified:\n{output[-4000:]}",
        )
        self._assert_loadable(output, f"{ENTRYPOINT} startup")

    def test_server_argv_survives_launch_chain(self):
        """Arguments must reach the server intact on both launch paths.

        A stub ``python3`` earlier on PATH captures the argv the server would have
        been exec'd with, so the whole real chain runs — entrypoint,
        hf_optimizations.sh, standard-supervisor, and in supervisor mode
        supervisord's command= round-trip — without importing vLLM.

        Guards the failure class where a value is mangled in transit rather than
        rejected outright: the server starts, parses its own flags, and dies on
        malformed input. standard-supervisor 0.1.16 switched from " ".join to
        shlex.join when writing command=, which flipped whether the LMCache
        --kv-transfer-config JSON had to be pre-quoted; getting it wrong for the
        installed version crash-loops every model.
        """
        for auto_recovery in ("false", "true"):
            with self.subTest(process_auto_recovery=auto_recovery):
                argv = self._run_entrypoint(
                    {"PROCESS_AUTO_RECOVERY": auto_recovery, "PROCESS_MAX_START_RETRIES": "1"},
                    stub_interpreter=True,
                )[0]
                self.assertIn("--port", argv, f"entrypoint never reached the server: argv={argv}")
                for flag, value in zip(argv, argv[1:]):
                    if "{" not in value:
                        continue
                    try:
                        json.loads(value)
                    except ValueError as exc:
                        self.fail(
                            f"{flag} reached the server malformed ({exc}); the launch chain "
                            f"mangled it: {value!r}"
                        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
