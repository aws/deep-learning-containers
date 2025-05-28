import os

try:
    if os.path.exists("/usr/local/bin/deep_learning_container.py") and (
        os.getenv("OPT_OUT_TRACKING") is None or os.getenv("OPT_OUT_TRACKING", "").lower() != "true"
    ):
        import threading

        cmd = "python /usr/local/bin/deep_learning_container.py --framework autogluon --framework-version 1.3.1 --container-type inference &>/dev/null"
        x = threading.Thread(target=lambda: os.system(cmd))
        x.setDaemon(True)
        x.start()
except Exception:
    pass
