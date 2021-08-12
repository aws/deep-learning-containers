from docker import APIClient
from docker import DockerClient

client = APIClient(base_url="unix://var/run/docker.sock")

folder_path = "/home/ubuntu/deep-learning-containers/src/"
dock_path = "/home/ubuntu/deep-learning-containers/src/Dockerfile.multipart"
build_args = {'FIRST_STAGE_IMAGE': '669063966089.dkr.ecr.us-west-2.amazonaws.com/beta-mxnet-training:1.8.0-cpu-py37-ubuntu16.04-2021-08-12-03-20-03'}

with open(dock_path, "rb") as dockerfile_obj:
    for line in client.build(fileobj=dockerfile_obj, path=folder_path, buildargs=build_args):
        print(line)