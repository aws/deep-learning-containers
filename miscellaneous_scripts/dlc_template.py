def main():
    import os

    if os.getenv("OPT_OUT_TRACKING", "").lower() == "true":
        return

    try:
        if os.path.exists("/usr/local/bin/deep_learning_container.py"):
            import sys
            import subprocess

            subprocess.Popen(
                [
                    sys.executable,
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
                start_new_session=True,
            )
    except:
        pass


main()
