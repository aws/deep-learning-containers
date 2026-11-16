"""Serve a tiny text classifier on Neuron with the image entrypoint (hf-serve).

The model is laid out in /opt/ml/model as SageMaker extracts `model.tar.gz`, and
requests go through the SageMaker routes (`/ping`, `/invocations`).

    python hf_serve_neuron_inference.py --mode eager
        zero-code `text-classification` task, then the custom handler in eager mode
    python hf_serve_neuron_inference.py --mode compiled
        the custom handler with `torch.compile(backend="neuron")`
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transformers_neuron_inference import save_tiny_bert  # noqa: E402

ENTRYPOINT = "/usr/local/bin/hf-serve-neuron-entrypoint.sh"
MODEL_DIR = Path("/opt/ml/model")
HANDLER = Path(__file__).resolve().parent / "neuron_inference_handler.py"
PORT = 8080
STARTUP_TIMEOUT = 1200
LABELS = {0: "NEGATIVE", 1: "POSITIVE"}


def prepare_model_dir(custom_handler):
    shutil.rmtree(MODEL_DIR, ignore_errors=True)
    MODEL_DIR.mkdir(parents=True)
    save_tiny_bert(
        str(MODEL_DIR),
        "BertForSequenceClassification",
        num_labels=2,
        id2label=LABELS,
        label2id={label: i for i, label in LABELS.items()},
    )
    if custom_handler:
        # The SageMaker SDK packs `entry_point`/`source_dir` files under code/.
        (MODEL_DIR / "code").mkdir()
        shutil.copy(HANDLER, MODEL_DIR / "code" / "handler.py")


def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        return response.status, json.loads(response.read() or b"null")


def serve(name, env, check):
    log_path = Path(f"/tmp/hf-serve-{name.replace('/', '-')}.log")
    server_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("HF_MODEL", "MODEL_", "TASK", "HF_TASK"))
    }
    server_env.update(env, CLOUD="sagemaker", PORT=str(PORT))
    with open(log_path, "w") as log:
        server = subprocess.Popen(
            [ENTRYPOINT, "serve"], env=server_env, stdout=log, stderr=subprocess.STDOUT
        )
    try:
        deadline = time.time() + STARTUP_TIMEOUT
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"{name}: server exited with {server.returncode}")
            try:
                if request("/ping")[0] == 200:
                    break
            except (urllib.error.URLError, ConnectionError):
                pass
            if time.time() > deadline:
                raise TimeoutError(f"{name}: server did not start in {STARTUP_TIMEOUT}s")
            time.sleep(5)
        status, output = request("/invocations", {"inputs": "hamilton is the best musical"})
        assert status == 200, f"{name}: status {status}, {output}"
        check(output, log_path.read_text())
        print(f"PASS {name} output={output}", flush=True)
    except BaseException:
        print(log_path.read_text()[-5000:], flush=True)
        raise
    finally:
        server.terminate()
        server.wait(timeout=60)


def check_zero_code(output, log):
    assert "on `device=neuron`" in log, "hf-serve did not load the model on the neuron device"
    prediction = output[0] if isinstance(output, list) else output
    assert prediction["label"] in LABELS.values(), output
    assert 0.0 <= prediction["score"] <= 1.0, output


def check_custom(mode):
    def check(output, log):
        assert output["device"] == "neuron", output
        assert output["mode"] == mode, output
        assert output["label"] in LABELS.values(), output

    return check


def main(mode):
    if mode == "eager":
        prepare_model_dir(custom_handler=False)
        serve("eager/zero-code", {"HF_TASK": "text-classification"}, check_zero_code)
    prepare_model_dir(custom_handler=True)
    serve(
        f"{mode}/custom-handler",
        {"HF_TRUST_REMOTE_CODE": "true", "NEURON_INFERENCE_MODE": mode},
        check_custom(mode),
    )
    print(f"PASS all mode={mode}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("eager", "compiled"), required=True)
    main(parser.parse_args().mode)
