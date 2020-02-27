import subprocess
import sys
import logging
import signal
import argparse

LOGGER = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def main():
    check_nvidia_smi()
    args = parse_args()
    framework = args.framework
    check_framework_configured_for_gpu(framework)
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework",
                        choices=["tensorflow", "mxnet", "pytorch"],
                        required=True)
    args = parser.parse_args()
    return args


def check_nvidia_smi():
    signal.signal(signal.SIGALRM, timeout_handler)
    # First start of nvidia-smi might take up to 100sec.
    signal.alarm(120)
    subprocess.check_output(["bash",
                             "-c",
                             "nvidia-smi"])
    signal.alarm(0)


def check_framework_configured_for_gpu(framework):
    # NOTE: this is an undocumented API.
    # https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    if framework == "tensorflow":
        subprocess.call([sys.executable, "-m", "pip", "install", "tensorflow-gpu==1.13.1"])
        subprocess.check_call([
            "bash", "-c",
            ("python -c '"
             "from tensorflow.python.client import device_lib;"
             "import sys;"
             "sys.exit(0 if any(d for d in device_lib.list_local_devices() if d.device_type == \"GPU\") else 1)"
             "'")])
    elif framework == "mxnet":
        subprocess.check_call([
            "bash", "-c",
            ("python -c '"
             "import mxnet as mx;"
             "import sys;"
             "sys.exit(0 if len(mx.test_utils.list_gpus()) !=0 else 1)"
             "'")
        ])
    elif framework == "pytorch":
        subprocess.check_call([
            "bash", "-c",
            ("python -c '"
             "import torch;"
             "import sys;"
             "sys.exit(0 if torch.cuda.is_available() else 1)"
             "'")
        ])
    else:
        print("framework is not supported currently")
        sys.exit(1)


def timeout_handler(signum, frame):
    raise Exception("The function timed out")

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
