from os.path import exists, join
import re

from invoke.context import Context


def host_setup_for_tensorflow_inference(container_name, framework_version):
    context = Context()
    home_dir = context.run("pwd", hide="out").stdout.strip("\n")
    src_location = join(home_dir, f"{container_name}-serving")
    if exists(src_location):
        context.run(f"rm -rf {src_location}", echo=True)

    if framework_version.startswith('1.'):
        fw_short_version = re.search(r"\d+\.\d+", framework_version).group()
        context.run(
            f"git clone -b r{fw_short_version} https://github.com/tensorflow/serving.git {src_location}", echo=True
        )
    else:
        # tensorflow/serving is not yet updated with scripts for TF 2.1, so using locally modified scripts
        script_location = join("container_tests", "bin", "tensorflow_serving")
        context.run(f"cp -r {script_location} {src_location}")

    context.run(f"""pip install -qq -U tensorflow=={framework_version}"""
                f""" "tensorflow-serving-api<={framework_version}" """, echo=True)

    script = join(src_location, "tensorflow_serving", "example", "mnist_saved_model.py")
    model_path = join(src_location, "models", "mnist")
    context.run(f"python {script} {model_path}", hide="out")

    return src_location, model_path


def request_tensorflow_inference_grpc(src_location, ip_address="127.0.0.1", port="8500"):
    context = Context()
    script = join(src_location, "tensorflow_serving", "example", "mnist_client.py")
    context.run(f"python {script} --num_tests=1000 --server={ip_address}:{port}", echo=True)


def tensorflow_inference_test_cleanup(src_location):
    context = Context()
    context.run(f"rm -rf {src_location}", warn=True, echo=True)
