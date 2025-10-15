import os
from invoke.context import Context
from codebuild_environment import get_cloned_folder_path
from infra.test_infra.test_infra_utils import create_logger

LOGGER = create_logger(__name__)


class EKSPlatform:
    def __init__(self):
        self.resources = None
        self.region = os.getenv("REGION", "us-west-2")
        self.build_context = os.getenv("BUILD_CONTEXT")
        self.cluster_name = None
        self.namespace = None
        self.image_uri = None
        self.framework = None
        self.arch_type = None
        self.ctx = Context()

    def setup(self, params):
        """
        Setup EKS infrastructure and return any resources needed for tests
        """
        LOGGER.info(f"Setting up EKS platform with params: {params}")

        self.framework = params.get("framework")
        self.arch_type = params.get("arch_type", "x86_64")
        cluster_prefix = params.get("cluster")
        self.cluster_name = f"{cluster_prefix}-{self.build_context}"
        self.namespace = params.get("namespace")
        self.image_uri = params.get("image_uri")

        LOGGER.info(
            f"EKS Platform - Framework: {self.framework}, Cluster: {self.cluster_name}, Namespace: {self.namespace}"
        )

    def execute_command(self, cmd):
        """
        Execute a test command with proper environment setup.
        Raises exception immediately if command fails.
        """
        try:
            env = {
                "AWS_REGION": self.region,
                "CLUSTER_NAME": self.cluster_name,
                "NAMESPACE": self.namespace,
                "BUILD_CONTEXT": self.build_context,
                "DLC_IMAGE": self.image_uri,
                "ARCH_TYPE": self.arch_type,
                "FRAMEWORK": self.framework,
            }

            repo_root = get_cloned_folder_path()

            with self.ctx.cd(repo_root):
                LOGGER.info(f"Executing command from {repo_root} with EKS environment: {cmd}")
                self.ctx.run(cmd, env=env)
                LOGGER.info(f"Command completed successfully: {cmd}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to execute command: {cmd}\nError: {str(e)}") from e
