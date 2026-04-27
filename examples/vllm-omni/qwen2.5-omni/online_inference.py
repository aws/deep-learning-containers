"""Online inference against a remote vLLM-Omni server running Qwen2.5-Omni-3B.

Set OMNI_ENDPOINT to the public URL of your EC2 instance, e.g.:
  export OMNI_ENDPOINT=http://ec2-xx-xx-xx-xx.us-west-2.compute.amazonaws.com:8080

See offline_inference.py for the local-server variant — the only difference is
the default endpoint.
"""

import os
import pathlib

from offline_inference import generate_audio

if __name__ == "__main__":
    endpoint = os.environ.get("OMNI_ENDPOINT")
    if not endpoint or endpoint.startswith("http://localhost"):
        raise SystemExit(
            "Set OMNI_ENDPOINT to the remote server URL, e.g. "
            "export OMNI_ENDPOINT=http://<ec2-host>:8080"
        )
    generate_audio(
        "Briefly describe the weather on Mars today.",
        pathlib.Path("out/mars.wav"),
    )
