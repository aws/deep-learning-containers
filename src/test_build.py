from docker import APIClient
from docker import DockerClient

client = APIClient(base_url="unix://var/run/docker.sock")

folder_path = "/home/ubuntu/deep-learning-containers/src/"
dock_path = "/home/ubuntu/deep-learning-containers/src/Dockerfile.multipart"
build_args = {'FIRST_STAGE_IMAGE': '669063966089.dkr.ecr.us-west-2.amazonaws.com/beta-mxnet-training:1.8.0-cpu-py37-ubuntu16.04-2021-08-12-03-20-03'}

with open(dock_path, "rb") as dockerfile_obj:
    for line in client.build(fileobj=dockerfile_obj, path=folder_path, buildargs=build_args):
        print(line)

docker_multipart_path = os.path.join(os.sep, os.getenv("PYTHONPATH"), "src", "Dockerfile.multipart")
safety_report_path = os.path.join(os.sep, os.getenv("PYTHONPATH"), "src", "safety_report.json")
os.path.join(os.sep, os.getenv("PYTHONPATH"), "src") + "/"

# import os
# from context import Context
# ARTIFACTS = {}
# ARTIFACTS.update({
#                     "safety_report": {
#                         "source": f"safety_report.json",
#                         "target": "safety_report.json"
#                     }
#                 })

# ARTIFACTS.update(
#             {
#                 "dockerfile": {
#                     "source": f"Dockerfile.multipart",
#                     "target": "Dockerfile",
#                 }
#             }
#         )
# artifact_root = os.path.join(os.sep, os.getenv("PYTHONPATH"), "src") + "/"
# Context(ARTIFACTS, context_path=f'build/safety-json-file.tar.gz',artifact_root=artifact_root)