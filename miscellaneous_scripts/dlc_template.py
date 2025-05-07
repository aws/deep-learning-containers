# import os

# try:
#     if os.path.exists("/usr/local/bin/deep_learning_container.py") and (
#         os.getenv("OPT_OUT_TRACKING") is None or os.getenv("OPT_OUT_TRACKING", "").lower() != "true"
#     ):
#         import threading

#         cmd = "python /usr/local/bin/deep_learning_container.py --framework {FRAMEWORK} --framework-version {FRAMEWORK_VERSION} --container-type {CONTAINER_TYPE} &>/dev/null"
#         x = threading.Thread(target=lambda: os.system(cmd))
#         x.setDaemon(True)
#         x.start()
# except Exception:
#     pass


import os

if not os.getenv("SITECUSTOMIZE_RUNNING"):
    try:
        os.environ["SITECUSTOMIZE_RUNNING"] = "1"
        if os.path.exists("/usr/local/bin/deep_learning_container.py") and (
            os.getenv("OPT_OUT_TRACKING") is None or os.getenv("OPT_OUT_TRACKING", "").lower() != "true"
        ):
            cmd = "python /usr/local/bin/deep_learning_container.py --framework {FRAMEWORK} --framework-version {FRAMEWORK_VERSION} --container-type {CONTAINER_TYPE} &>/dev/null"
            os.system(cmd)
            del os.environ["SITECUSTOMIZE_RUNNING"]
    except Exception:
        pass
