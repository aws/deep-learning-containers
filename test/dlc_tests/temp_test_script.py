import os
from invoke import Context

def run_upgrade_on_image_and_push(image):
    repository_name = os.getenv('UPGRADE_REPO_NAME')
    ecr_account = f"{os.getenv('ACCOUNT_ID')}.dkr.ecr.{os.getenv('REGION')}.amazonaws.com"
    upgraded_image_tag = '-'.join(image.replace("/",":").split(":")[1:]) + "-up"
    new_image_uri = f"{ecr_account}/{repository_name}:{upgraded_image_tag}"
    ctx = Context()
    docker_run_cmd = f"docker run -id --entrypoint='/bin/bash' {image}"
    container_id = ctx.run(f"{docker_run_cmd}", hide=True, warn=True).stdout.strip()
    apt_command = "apt-get update && apt-get upgrade"
    docker_exec_cmd = f"docker exec -i {container_id}"
    run_output = ctx.run(f"{docker_exec_cmd} {apt_command}", hide=True, warn=True)
    if not run_output.ok:
        raise ValueError("Could Not Run Apt Update and Upgrade.")
    ctx.run(f"docker commit {container_id} {new_image_uri}", hide=True, warn=True)
    ctx.run(f"docker push {new_image_uri}")

def run_upgrade_on_image_and_push(image, new_image_uri):
    ctx = Context()
    docker_run_cmd = f"docker run -id --entrypoint='/bin/bash' {image}"
    container_id = ctx.run(f"{docker_run_cmd}", hide=True, warn=True).stdout.strip()
    apt_command = "apt-get update && apt-get upgrade"
    docker_exec_cmd = f"docker exec -i {container_id}"
    run_output = ctx.run(f"{docker_exec_cmd} {apt_command}", hide=True, warn=True)
    if not run_output.ok:
        raise ValueError("Could not run apt update and upgrade.")
    ctx.run(f"docker commit {container_id} {new_image_uri}", hide=True, warn=True)
    ctx.run(f"docker push {new_image_uri}")

    # ctx.run(f"docker commit {container_id} {new_image_uri}")

    # 669063966089.dkr.ecr.us-west-2.amazonaws.com/beta-mxnet-training:1.8.0-gpu-py37-cu110-ubuntu16.04-example-2021-09-13-20-17-11

run_upgrade_on_image_and_push("669063966089.dkr.ecr.us-west-2.amazonaws.com/beta-mxnet-training:1.8.0-gpu-py37-cu110-ubuntu16.04-example-2021-09-13-20-17-11")