from invoke.context import Context
from os.path import exists, join


def host_setup_for_tensorflow_inference(container_name, framework_version):
    context = Context()
    home_dir = context.run("$pwd", hide="out").stdout.strip("\n")
    src_location = join(home_dir, f"{container_name}-serving")
    if exists("serving"):
        context.run("rm -rf serving", echo=True)
    context.run(f"git clone https://github.com/tensorflow/serving.git {src_location}", echo=True)

    venv_location = join(home_dir, container_name)
    context.run(f"virtualenv {venv_location}", echo=True)
    if exists(venv_location):
        print(f"{venv_location} exists")
        if exists(f"{venv_location}/bin"):
            print(f"{venv_location}/bin exists")
            if exists(f"{venv_location}/bin/activate"):
                print(f"{venv_location}/bin/activate exists")
    with context.prefix(f"source {venv_location}/bin/activate"):
        context.run(f"""pip install -qq -U tensorflow=={framework_version}"""
                    f""" "tensorflow-serving-api<={framework_version}" """, echo=True)
        with context.cd(src_location):
            script = join("tensorflow_serving", "example", "mnist_saved_model.py")
            model_path = join("models", "mnist")
            context.run(f"python {script} {model_path}", hide="out")
    abs_model_path = join(src_location, model_path)
    return src_location, venv_location, abs_model_path


def request_tensorflow_inference_grpc(src_location, venv_location, ip_address="127.0.0.1", port="8500"):
    context = Context()
    with context.prefix(f"source {venv_location}/bin/activate"):
        with context.cd(src_location):
            script = join("tensorflow_serving", "example", "mnist_client.py")
            context.run(f"python {script} --num_tests=1000 --server={ip_address}:{port}", echo=True)


def tensorflow_inference_test_cleanup(src_location, venv_location):
    context = Context()
    context.run(f"rm -rf {venv_location} {src_location}", warn=True, echo=True)
