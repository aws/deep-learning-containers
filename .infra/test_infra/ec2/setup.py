import os
from invoke.context import Context
from test_infra.test_infra_utils import create_logger
from codebuild_environment import get_cloned_folder_path

LOGGER = create_logger(__name__)


class EC2Platform:
    def __init__(self):
        self.resources = None
        self.region = os.getenv("REGION", "us-west-2")
        self.build_context = os.getenv("BUILD_CONTEXT")
        self.image_uri = None
        self.ctx = Context()

    def setup(self, params):
        """
        Setup EC2 infrastructure
        """
        LOGGER.info(f"Setting up EC2 platform with params: {params}")

        framework = params.get("framework")
        self.image_uri = params.get("image_uri")

        if framework == "vllm":
            # vllm requires vLLM-specific setup (FSx + multi-node)
            LOGGER.info(f"Would call vLLM setup for image: {self.image_uri}")
        else:
            # standard EC2 setup for other frameworks
            LOGGER.info(f"Would call standard EC2 setup for image: {self.image_uri}")
            self._standard_ec2_setup(params)

    def execute_command(self, cmd):
        """
        Execute a test command with proper environment setup
        """
        env = {
            "AWS_REGION": self.region,
            "BUILD_CONTEXT": self.build_context,
            "DLC_IMAGE": self.image_uri,
        }

        repo_root = get_cloned_folder_path()

        with self.ctx.cd(repo_root):
            LOGGER.info(f"Executing command from {repo_root} with EC2 environment: {cmd}")
            self.ctx.run(cmd, env=env)

    def _standard_ec2_setup(self, params):
        """
        Generic EC2 setup - to be implemented
        """
        instance_type = params.get("instance_type", "g4dn.xlarge")
        node_count = params.get("node_count", 1)
        region = params.get("region", "us-west-2")

        LOGGER.info(f"Standard EC2 setup: {instance_type}, {node_count} nodes, {region}")

        # TODO: Implement generic EC2 provisioning
        raise NotImplementedError("Standard EC2 setup not yet implemented")
