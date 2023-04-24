import os
import re


def execute_single_node_benchmark(context, image_uri, framework, task, py_version, script_url):
    """
    Shared code to execute single-node benchmarks
    :param context: Context() from callers
    :param image_uri: identifies mage
    :param framework: Framework ex(TF,MXNET)
    :param task: specifies task name
    :param py_version: python version
    :param script: github url to download script for benchmark
    """
    benchmark_client_id = os.getenv("BENCHMARK_CLIENT_ID")
    benchmark_endpoint = os.getenv("BENCHMARK_ENDPOINT")
    benchmark_exec = os.getenv("BENCHMARK_EXEC")
    container_name = f"{framework}_single_node-{image_uri.split('/')[-1].replace('.', '-').replace(':', '-')}"
    bai_dir = os.path.join(os.getcwd(), "benchmark", "bai")
    path_to_toml = os.path.join(bai_dir, framework, "training", f"{framework}_single.toml")
    custom_toml_name = f"custom_{framework}_{py_version}.toml"
    write_image_to_toml(image_uri, path_to_toml, task, custom_toml_name)
    toml = os.path.join("test", framework, "training", custom_toml_name)
    script_directory = re.search(r"[^/]+[A-Za-z0-9]+.git\Z", script_url).group().split(".")[0]
    context.run(
        f"docker run --name {container_name} -v {bai_dir}:{os.path.join(os.sep, 'test')} -itd bai_env_container",
        hide=True,
    )
    execute_cmd_on_container(context, container_name, f"git clone {script_url}")

    execute_cmd_on_container(
        context, container_name, f"curl http://{benchmark_endpoint}/api/tools/{benchmark_exec} " f"-o {benchmark_exec}"
    )
    execute_cmd_on_container(context, container_name, f"chmod u+x {benchmark_exec}")
    execute_cmd_on_container(context, container_name, f"./{benchmark_exec} --env-setup")
    execute_cmd_on_container(context, container_name, f"./{benchmark_exec} --register {benchmark_endpoint}")
    execute_cmd_on_container(context, container_name, f"./{benchmark_exec} --set-client-id {benchmark_client_id}")
    output = execute_cmd_on_container(context, container_name, f"./{benchmark_exec} --get-client-id")
    assert benchmark_client_id in output.stdout, output.stdout
    execute_cmd_on_container(
        context, container_name, f"./{benchmark_exec} --submit {toml} --script {script_directory}", True
    )
    execute_cmd_on_container(context, container_name, f"./{benchmark_exec} --watch --status --terminate")


def execute_cmd_on_container(context, container_name, bash_command, warn=False):
    """
    Executes bash commands on docker container
    :param context: Context passed from each testing method
    :param container_name: Name of container cmd will be executed on
    :param bash_command: Bash command that will be executed on container
    :param warn: Default is False, if set to True will not cause codebuild to Fail when returned with a non-zero exit code
    :return: Returns result object of bash command, contains stdout, stderr....
    """
    return context.run(f"docker exec --user root {container_name} bash -c '{bash_command}'", hide=True, warn=warn)


def write_image_to_toml(image_uri, path_to_toml, task, custom_toml_name):
    """
    Creates toml file from a template.
    :param image_uri: identifies image
    :param path_to_toml: Path to toml file in Docker
    :param task: Specifies task name
    :param custom_toml_name: Specifies name of newly created toml
    """
    toml_dir = os.path.dirname(path_to_toml)
    custom_toml = os.path.join(toml_dir, custom_toml_name)
    with open(path_to_toml, "r") as template_toml:
        with open(custom_toml, "w") as edited_toml:
            for line in template_toml:
                if "docker_image" in line:
                    edited_toml.write(f'docker_image = "{image_uri}"')
                elif "task_name" in line:
                    edited_toml.write(f'task_name = "{task}"')
                elif "description =" in line:
                    edited_toml.write(f'description = """ {task} -{image_uri}')
                else:
                    edited_toml.write(line)


def get_framework_version(image_uri):
    """
    Gets framework version from image_uri
    :param image_uri: Where framework_version will be regexed from
    :return: Returns str representing framework_version
    """
    framework_version = re.search(r":\s*([\d][.][\d]+)", image_uri).group(1)
    return framework_version


def get_py_version(image_uri):
    """
    Gets python version from image_uri
    :param image_uri: Where python_version will be regexed from
    :return: Returns str representing python_version
    """
    python_version = re.search(r"py\s*([\d])", image_uri).group()
    return python_version
