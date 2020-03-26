import os
import re

from invoke import run
import pytest

from test.test_utils import get_image_framework_version


@pytest.fixture(scope="module")
def mnist_serving_model():
    any_image = os.getenv("DLC_IMAGES").split(" ")[0]
    framework_version = get_image_framework_version(any_image)
    home_dir = run("pwd", hide="out").stdout.strip("\n")
    src_location = os.path.join(home_dir, "serving")
    if os.path.exists(src_location):
        run(f"rm -rf {src_location}", echo=True)
    if framework_version.startswith("1."):
        fw_short_version = re.search(r"\d+\.\d+", framework_version).group()
        run(f"git clone -b r{fw_short_version} https://github.com/tensorflow/serving.git {src_location}", echo=True)
        src_location = os.path.join(src_location, "tensorflow_serving", "example")
        tensorflow_package_name = "tensorflow"
    else:
        # tensorflow/serving is not yet updated with scripts for TF 2.1, so using locally modified scripts
        script_location = os.path.join("container_tests", "bin", "tensorflow_serving", "example")
        run(f"cp -r {script_location} {src_location}")
        tensorflow_package_name = "tensorflow-cpu"

    run(f"pip install --user -qq -U {tensorflow_package_name}=={framework_version}", echo=True)
    run(f"""pip install --user -qq "tensorflow-serving-api<={framework_version}" """, echo=True)

    script = os.path.join(src_location, "mnist_saved_model.py")
    model_path = os.path.join(src_location, "models", "mnist")
    run(f"python {script} {model_path}", hide="out")

    return src_location, model_path
