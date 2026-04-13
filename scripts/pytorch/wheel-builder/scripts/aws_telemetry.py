################################################################################
# AWS Container Telemetry
################################################################################
try:
    if os.path.exists("/usr/local/bin/deep_learning_container.py") and (
        os.getenv("OPT_OUT_TRACKING") is None
        or os.getenv("OPT_OUT_TRACKING") not in ["True", "TRUE", "true"]
    ):
        import subprocess
        import threading

        dlc_container_type = os.getenv("DLC_CONTAINER_TYPE", "inference")
        cmd = (
            f"python /usr/local/bin/deep_learning_container.py --framework pytorch --container-type {dlc_container_type} "
            "--framework-version " + __version__ + " &>/dev/null"
        )
        # creating a daemon thread, so that main thread can complete without stalling.
        x = threading.Thread(target=lambda: os.system(cmd))
        x.setDaemon(True)
        x.start()
except Exception:
    pass
