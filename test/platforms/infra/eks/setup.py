import os
from invoke import run
from test.test_utils import LOGGER
from codebuild_environment import get_cloned_folder_path

class EKSPlatform:
    def __init__(self):
        self.resources = None
        self.region = os.getenv("REGION", "us-west-2")
        self.build_context = os.getenv("BUILD_CONTEXT")
        self.cluster_name = None
        self.namespace = None

    def setup(self, params):
        """
        Setup EKS infrastructure and return any resources needed for tests
        """
        LOGGER.info(f"Setting up EKS platform with params: {params}")

        framework = params.get("framework")
        cluster_prefix = params.get("cluster") 
        self.cluster_name = f"{cluster_prefix}-{self.build_context}"
        self.namespace = params.get("namespace")
        
        LOGGER.info(f"EKS Platform - Framework: {framework}")
        LOGGER.info(f"EKS Platform - Cluster: {self.cluster_name}")
        LOGGER.info(f"EKS Platform - Namespace: {self.namespace}")

        if not os.getenv("DLC_IMAGE"):
            raise ValueError("DLC_IMAGE environment variable not set")

    def execute_command(self, cmd):
        """
        Execute a test command with proper environment setup
        """
        env = {
            "AWS_REGION": self.region,
            "CLUSTER_NAME": self.cluster_name,
            "NAMESPACE": self.namespace,
            "BUILD_CONTEXT": self.build_context,
            "DLC_IMAGE": os.getenv("DLC_IMAGE"),
        }
        
        repo_root = get_cloned_folder_path()

        LOGGER.info(f"Executing command from {repo_root} with EKS environment: {cmd}")
        run(cmd, env=env, cwd=repo_root)