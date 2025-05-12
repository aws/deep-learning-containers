import os
import sys
import subprocess

try:
    if os.path.exists("/usr/local/bin/deep_learning_container.py") and (
        os.getenv("OPT_OUT_TRACKING") is None or os.getenv("OPT_OUT_TRACKING", "").lower() != "true"
    ):
        import threading

        python_executable = sys.executable

        def run_script():
            try:
                subprocess.run(
                    [
                        python_executable,
                        "/usr/local/bin/deep_learning_container.py",
                        "--framework",
                        "{FRAMEWORK}",
                        "--framework-version",
                        "{FRAMEWORK_VERSION}",
                        "--container-type",
                        "{CONTAINER_TYPE}",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                )
            except subprocess.TimeoutExpired:
                pass
            except Exception as e:
                pass

        x = threading.Thread(target=run_script)
        x.daemon = True
        x.start()
except Exception:
    pass
