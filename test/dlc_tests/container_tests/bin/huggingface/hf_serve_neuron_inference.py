"""Serve tiny models on Neuron with the image entrypoint (hf-serve).

The model is laid out in /opt/ml/model as SageMaker extracts `model.tar.gz`, and
requests go through the SageMaker routes (`/ping`, `/invocations`).

    python hf_serve_neuron_inference.py --mode eager
        zero-code `text-classification` and `text-to-image` (tiny Stable Diffusion)
        tasks, then the custom handler in eager mode
    python hf_serve_neuron_inference.py --mode compiled
        the custom handler with `torch.compile(backend="neuron")`
"""

import argparse
import io
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
from transformers_neuron_inference import save_tiny_bert, save_tiny_stable_diffusion  # noqa: E402

ENTRYPOINT = "/usr/local/bin/hf-serve-neuron-entrypoint.sh"
MODEL_DIR = Path("/opt/ml/model")
HANDLER = Path(__file__).resolve().parent / "neuron_inference_handler.py"
PORT = 8080
STARTUP_TIMEOUT = 1200
LABELS = {0: "NEGATIVE", 1: "POSITIVE"}
TEXT_CLASSIFICATION_PAYLOAD = {"inputs": "hamilton is the best musical"}
TEXT_TO_IMAGE_PAYLOAD = {
    "inputs": "a photo of an astronaut riding a horse on mars",
    "parameters": {"width": 32, "height": 32, "num_inference_steps": 2, "seed": 42},
}


def reset_model_dir():
    shutil.rmtree(MODEL_DIR, ignore_errors=True)
    MODEL_DIR.mkdir(parents=True)


def prepare_model_dir(custom_handler):
    reset_model_dir()
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


def request(path, payload=None, accept="application/json"):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=data,
        headers={"Content-Type": "application/json", "Accept": accept},
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        body = response.read()
        if accept != "application/json":
            assert response.headers["Content-Type"] == accept, response.headers["Content-Type"]
            return response.status, body
        return response.status, json.loads(body or b"null")


def serve(name, env, check, payload=TEXT_CLASSIFICATION_PAYLOAD, accept="application/json"):
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
        status, output = request("/invocations", payload, accept)
        assert status == 200, f"{name}: status {status}, {output}"
        summary = check(output, log_path.read_text())
        print(f"PASS {name} output={summary or output}", flush=True)
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


def check_text_to_image(output, log):
    from PIL import Image

    assert "on `device=neuron`" in log, "hf-serve did not load the pipeline on the neuron device"
    image = Image.open(io.BytesIO(output))
    assert image.format == "PNG", image.format
    assert image.size == (32, 32), image.size
    # An all-black image would mean NaNs in the denoising loop or the VAE.
    assert image.convert("L").getextrema() != (0, 0), "text-to-image returned a black image"
    return f"PNG {image.size[0]}x{image.size[1]}"


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
        reset_model_dir()
        save_tiny_stable_diffusion(str(MODEL_DIR))
        serve(
            "eager/zero-code-text-to-image",
            {"HF_TASK": "text-to-image"},
            check_text_to_image,
            payload=TEXT_TO_IMAGE_PAYLOAD,
            accept="image/png",
        )
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
