import os
import sys
import subprocess
from pathlib import Path

TRACKING_SCRIPT = "/usr/local/bin/deep_learning_container.py"

try:
    if Path(TRACKING_SCRIPT).is_file() and not os.getenv("OPT_OUT_TRACKING", "").lower() == "true":

        subprocess.Popen(
            [
                sys.executable,
                TRACKING_SCRIPT,
                "--framework",
                "{FRAMEWORK}",
                "--framework-version",
                "{FRAMEWORK_VERSION}",
                "--container-type",
                "{CONTAINER_TYPE}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
except:
    pass
