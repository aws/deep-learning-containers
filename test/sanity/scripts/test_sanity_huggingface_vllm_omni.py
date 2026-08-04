#!/usr/bin/env python3
"""
Sanity tests for the Hugging Face-contributed vLLM-Omni DLC image.

Scoped to the huggingface-vllm-omni family and invoked only by
.github/workflows/huggingface-vllm-omni.pipeline.yml — these are not part of the
shared vLLM/SGLang sanity suite.

Run inside the container:
    docker run --rm -v $(pwd):/workdir --workdir /workdir \
        --entrypoint python3 <image> \
        test/sanity/scripts/test_sanity_huggingface_vllm_omni.py

The image is identified by its HF omni entrypoint; every test skips cleanly when
the container is not a Hugging Face vLLM-Omni SageMaker image. No accelerator is
required.
"""

import json
import os
import signal
import subprocess
import tempfile
import unittest
from importlib.metadata import entry_points, version

ENTRYPOINT = "/usr/local/bin/sagemaker_entrypoint.sh"
OPTIMIZATIONS = "/usr/local/bin/hf_optimizations.sh"
MIDDLEWARE = "/usr/local/bin/omni_sagemaker_serve.py"
MIDDLEWARE_ARG = "omni_sagemaker_serve.SageMakerRouteMiddleware"


def _is_hf_omni_image():
    """True when this container is a Hugging Face vLLM-Omni SageMaker image.

    Keyed on structural markers only — the HF entrypoint, the shared HF
    optimization layer, and the omni /invocations router, which together exist in
    no other image. Deliberately NOT keyed on anything this file asserts (omni
    mode, the middleware flag, LMCache defaults): a gate that inspects the
    behaviour under test turns a regression into a silent skip, and these tests
    run in one pipeline only, so a skip reads as a green build.
    """
    for path in (ENTRYPOINT, OPTIMIZATIONS, MIDDLEWARE):
        if not os.path.isfile(path):
            return False
    with open(ENTRYPOINT) as f:
        return "SM_VLLM_" in f.read()


def _minor(raw):
    """major.minor of a version string, ignoring any local/pre-release suffix."""
    return ".".join(raw.split("+")[0].split(".")[:2])


class TestOmniStack(unittest.TestCase):
    """The plugin contract this image is built on must hold.

    vllm-omni ships no vLLM of its own: it is a pure-Python plugin installed onto
    the vLLM DLC's environment, registering itself through the
    ``vllm.general_plugins`` entry point and adding ``--omni`` to vLLM's parser.
    Its wheel metadata declares no vllm requirement, so pip/uv will never object
    to a mismatched pair, and upstream only guarantees vllm-omni X.Y against vLLM
    X.Y. These checks are the run-time counterpart of the Dockerfile's build-time
    assertions: they fail on an image where the base moved, a dependency replaced
    the base's source-built vLLM, or the plugin hook stopped being registered —
    each of which yields a container that starts and then serves nothing.
    """

    def setUp(self):
        if not _is_hf_omni_image():
            self.skipTest("not a Hugging Face vLLM-Omni SageMaker image")

    def test_vllm_and_omni_minors_pair(self):
        vllm_version = version("vllm")
        omni_version = version("vllm-omni")
        self.assertEqual(
            _minor(vllm_version),
            _minor(omni_version),
            f"vllm {vllm_version} is not the pair of vllm-omni {omni_version}; "
            "upstream only supports matching minors, and nothing in the "
            "dependency metadata enforces it",
        )

    def test_plugin_hook_is_registered(self):
        hooks = {ep.name: ep.value for ep in entry_points(group="vllm.general_plugins")}
        self.assertIn(
            "vllm_omni_register_models",
            hooks,
            f"vllm-omni did not register its vLLM plugin hook (found {hooks}); "
            "omni models would be unknown to the server",
        )

    def test_middleware_imports(self):
        """The /invocations router must import with only what the image ships."""
        proc = subprocess.run(
            ["python3", "-c", "import omni_sagemaker_serve as m; m.SageMakerRouteMiddleware"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"{MIDDLEWARE} is not importable, so every non-default /invocations "
            f"route would 404:\n{proc.stderr[-2000:]}",
        )


class TestServerStartup(unittest.TestCase):
    """The server must survive its own startup path.

    Guards the failure class where something in the server's module graph is
    imported eagerly and cannot load in this image — a package that is absent, or
    a native extension whose shared libraries are missing. The server then
    crash-loops for *every* workload, and a SageMaker endpoint never reaches
    InService. omni widens that surface considerably: av, imageio-ffmpeg,
    openai-whisper, onnxruntime and the fa3-fwd native extension all sit in the
    import path, on top of an image whose ffmpeg comes from SPAL.

    A second class the same launch produces: an argument mangled in transit, so
    the server starts, parses its own flags, and dies on malformed input. Those
    are asserted on the argv the server actually receives.

    Failures are asserted by the *absence* of import/link signatures rather than
    the presence of an expected error, so the checks do not rot when upstream
    error messages change. That is also what makes this runnable on the GPU-less
    sanity runner: the module graph loads before any device is needed, and the
    process is then expected to die in engine init.
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
        if not _is_hf_omni_image():
            self.skipTest("not a Hugging Face vLLM-Omni SageMaker image")

    def _run_entrypoint(self, env_overrides, stub_server=False, model_source="env"):
        """Run the real entrypoint; return (server argv, combined output).

        With stub_server, a fake ``vllm`` shadows the real one on PATH and records
        the argv it was handed, so the whole launch chain runs — entrypoint,
        hf_optimizations.sh, SM_VLLM_* translation — without importing vLLM.
        argv is empty when no stub was installed.

        model_source selects how the model reaches the server:
          "env"    - SM_VLLM_MODEL points at a scratch model directory.
          "mount"  - nothing is set; a copy of the entrypoint with /opt/ml/model
                     redirected at that directory exercises SageMaker's mounted
                     model artifact path without writing to the real /opt/ml.
          "none"   - nothing is set, so the caller drives the HF_MODEL_ID or
                     no-model-specified branch.
        """
        workdir = tempfile.mkdtemp(prefix="hf_omni_sanity_")
        model_dir = os.path.join(workdir, "model")
        os.makedirs(model_dir)
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            f.write("{}")

        env = {k: v for k, v in os.environ.items() if not k.startswith("SM_VLLM_")}
        env.pop("HF_MODEL_ID", None)
        if model_source == "env":
            env["SM_VLLM_MODEL"] = model_dir
        env.update(env_overrides)

        entrypoint = ENTRYPOINT
        if model_source == "mount":
            with open(ENTRYPOINT) as f:
                script = f.read().replace("/opt/ml/model", model_dir)
            entrypoint = os.path.join(workdir, "sagemaker_entrypoint.sh")
            with open(entrypoint, "w") as f:
                f.write(script)
            os.chmod(entrypoint, 0o755)

        argv_file = os.path.join(workdir, "argv")
        if stub_server:
            stub = os.path.join(workdir, "vllm")
            with open(stub, "w") as f:
                f.write(f'#!/bin/sh\nprintf "%s\\0" "$@" >{argv_file}\n')
            os.chmod(stub, 0o755)
            env["PATH"] = workdir + os.pathsep + env.get("PATH", "")

        proc = subprocess.Popen(
            [entrypoint],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        timeout = self.ARGV_TIMEOUT if stub_server else self.STARTUP_TIMEOUT
        try:
            output = proc.communicate(timeout=timeout)[0]
        except subprocess.TimeoutExpired:
            # A server that reaches its listening state never exits — kill the
            # whole group so no child holds the stdout pipe.
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            output = proc.communicate()[0]

        argv = []
        if os.path.isfile(argv_file):
            with open(argv_file) as f:
                argv = f.read().split("\0")[:-1]
        return argv, output, model_dir

    def test_entrypoint_startup_loads_all_code(self):
        """The real SageMaker entrypoint must not die on unloadable code.

        Invokes the entrypoint the way SageMaker does, which exercises what a
        synthetic import cannot: hf_optimizations.sh, the SM_VLLM_* to CLI
        translation, the injected --middleware, and vLLM's own plugin loading of
        vllm-omni. Without an accelerator the process is expected to fail in
        engine/platform init, so the assertion is on *how* it fails.
        """
        output = self._run_entrypoint({})[1]
        self.assertRegex(
            output.lower(),
            r"vllm",
            f"{ENTRYPOINT} produced no server output, so nothing was verified:\n{output[-4000:]}",
        )
        hits = [p for p in self.FATAL_PATTERNS if p in output]
        self.assertEqual(
            hits,
            [],
            f"{ENTRYPOINT} startup hit {hits} — the server cannot load its own "
            f"code, so it will crash-loop for every model:\n{output[-4000:]}",
        )

    def test_server_argv_survives_launch_chain(self):
        """Arguments must reach the server intact, with omni mode requested."""
        argv = self._run_entrypoint(
            {"SM_VLLM_MAX_MODEL_LEN": "4096", "SM_VLLM_TRUST_REMOTE_CODE": "true"},
            stub_server=True,
        )[0]

        self.assertEqual(
            argv[:2],
            ["serve", "--omni"],
            f"entrypoint must launch vLLM in omni mode: argv={argv}",
        )
        self.assertIn("--port", argv, f"entrypoint never reached the server: argv={argv}")
        self.assertIn(
            MIDDLEWARE_ARG,
            argv,
            "the omni /invocations router was not installed; SageMaker would 404 "
            f"every omni route: argv={argv}",
        )
        # SM_VLLM_* translation: value flags carry their value, booleans do not.
        self.assertEqual(
            argv[argv.index("--max-model-len") + 1],
            "4096",
            f"SM_VLLM_MAX_MODEL_LEN did not reach the server: argv={argv}",
        )
        self.assertIn("--trust-remote-code", argv)
        self.assertNotIn("true", argv, f"a boolean leaked its value: argv={argv}")

        for flag, value in zip(argv, argv[1:]):
            if "{" not in value:
                continue
            try:
                json.loads(value)
            except ValueError as exc:
                self.fail(
                    f"{flag} reached the server malformed ({exc}); the launch "
                    f"chain mangled it: {value!r}"
                )

    def test_model_resolution_paths(self):
        """Every documented way to name a model must reach the server.

        The shared vLLM sanity suite cannot cover this image: its dry-run harness
        neutralises the launch by rewriting `exec [standard-supervisor] python3
        ...`, and an omni entrypoint execs `vllm serve --omni`, so the whole
        framework is excluded from that step. These SageMaker contracts are
        therefore asserted here — a model mounted at /opt/ml/model (the S3 model
        artifact path), HF_MODEL_ID (hub id, the HF-idiomatic path), and a
        boolean SM_VLLM_* set false, which must vanish rather than arrive as a
        flag or leak the string "false".
        """
        mounted, _, model_dir = self._run_entrypoint({}, stub_server=True, model_source="mount")
        self.assertEqual(
            mounted[mounted.index("--model") + 1],
            model_dir,
            f"a mounted model artifact was not auto-detected: argv={mounted}",
        )

        model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        hub = self._run_entrypoint(
            {"HF_MODEL_ID": model_id}, stub_server=True, model_source="none"
        )[0]
        self.assertEqual(
            hub[hub.index("--model") + 1],
            model_id,
            f"HF_MODEL_ID did not reach the server: argv={hub}",
        )

        disabled = self._run_entrypoint(
            {"SM_VLLM_ENABLE_PREFIX_CACHING": "false"}, stub_server=True
        )[0]
        self.assertNotIn(
            "--enable-prefix-caching",
            disabled,
            f"a false boolean was still passed to the server: argv={disabled}",
        )
        self.assertNotIn("false", disabled, f"a boolean leaked its value: argv={disabled}")

    def test_lmcache_is_opt_in(self):
        """LMCache must stay off unless asked for, and be valid when asked.

        Diffusion and multi-stage omni models have no autoregressive KV cache for
        a KV connector to attach to, so a default-on LMCache would risk a startup
        failure for most of this image's workloads — hence the omni entrypoint
        flips the shared HF layer's default. The opt-in path still has to produce
        a --kv-transfer-config vLLM can json.loads().
        """
        default_argv = self._run_entrypoint({}, stub_server=True)[0]
        self.assertNotIn(
            "--kv-transfer-config",
            default_argv,
            f"LMCache must be opt-in for omni workloads: argv={default_argv}",
        )

        opted_in = self._run_entrypoint({"HF_ENABLE_LMCACHE": "1"}, stub_server=True)[0]
        self.assertIn(
            "--kv-transfer-config",
            opted_in,
            f"HF_ENABLE_LMCACHE=1 did not reach the server: argv={opted_in}",
        )
        config = json.loads(opted_in[opted_in.index("--kv-transfer-config") + 1])
        self.assertEqual(config.get("kv_connector"), "LMCacheConnectorV1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
