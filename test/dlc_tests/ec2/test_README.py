import os

import pytest

from invoke.context import Context

from test.test_utils import get_source_version


@pytest.mark.parametrize("ec2_instance_type", ["c5.18xlarge"], indirect=True)
@pytest.mark.parametrize("ec2_key_name", [f"local-build-doc-{get_source_version()}"], indirect=True)
def test_local_build_documentation(ec2_connection, dlc_images):
    ctx = Context()
    commit_hash = get_source_version()
    region = os.getenv("AWS_REGION", os.getenv("REGION", "us-west-2"))
    account_id = os.getenv("ACCOUNT_ID", dlc_images[0].split(".")[0])
    framework = None
    for fw in ("tensorflow", "mnxet", "pytorch"):
        if fw in dlc_images[0]:
            framework = fw
            break
    github_repo_link = ctx.run("git remote get-url origin").stdout
    github_repo_name = ctx.run("basename `git rev-parse --show-toplevel`").stdout
    with ec2_connection.prefix(f"git clone {github_repo_link}", hide=True):
        with ec2_connection.cd(github_repo_name):
            ec2_connection.run(f"git checkout {commit_hash}")
            ec2_connection.run(f"export ACCOUNT_ID={account_id} && "
                               f"export REGION={region} && "
                               f"export REPOSITORY_NAME=beta-{framework}-training", hide=True)
            # Log into ECR
            ec2_connection.run("aws ecr get-login-password --region $REGION | docker login --username AWS "
                               "--password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com", hide=True)

            # Create and source virtual env
            ec2_connection.run("python3 -m venv dlc", hide=True)
            with ec2_connection.prefix("source dlc/bin/activate"):
                ec2_connection.run("pip install -r src/requirements.txt", hide=True)
                ec2_connection.run(f"bash src/setup.sh {framework}", hide=True)

                # Run build
                ec2_connection.run(f"python src/main.py --buildspec {framework}/buildspec.yml --framework {framework}")
