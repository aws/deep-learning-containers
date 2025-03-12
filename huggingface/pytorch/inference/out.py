import os

try:
    if os.path.exists("/usr/local/bin/deep_learning_container.py") and (
        os.getenv("OPT_OUT_TRACKING") is None or os.getenv("OPT_OUT_TRACKING", "").lower() != "true"
    ):
        import threading

        cmd = "python /usr/local/bin/deep_learning_container.py --framework huggingface_pytorch --framework-version 2.3.0 --container-type inference &>/dev/null"
        x = threading.Thread(target=lambda: os.system(cmd))
        x.setDaemon(True)
        x.start()
except Exception:
    pass
