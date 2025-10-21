import os
import threading
from invoke.context import Context
from codebuild_environment import get_cloned_folder_path
from infra.test_infra.test_infra_utils import create_logger

LOGGER = create_logger(__name__)

# Registry for framework-specific test execution
# Frameworks requiring direct Python function calls with resources should be registered here
TEST_REGISTRY = {
    "vllm": {
        "module": "test.v2.ec2.vllm.test_ec2",
        "function": "test_vllm_on_ec2",
        "requires_resources": True,
    },
    # Future frameworks to be added here
}


class EC2Platform:
    def __init__(self):
        self.resources = None
        self.region = os.getenv("REGION", "us-west-2")
        self.build_context = os.getenv("BUILD_CONTEXT")
        self.image_uri = None
        self.framework = None
        self.arch_type = None
        self.ctx = Context()

    def setup(self, params):
        """
        Setup EC2 infrastructure
        """
        LOGGER.info(f"Setting up EC2 platform with params: {params}")

        self.framework = params.get("framework")
        self.arch_type = params.get("arch_type", "x86_64")
        self.image_uri = params.get("image_uri")
        self.instance_type = params.get("instance_type")
        self.node_count = params.get("node_count")

        if self.framework == "vllm":
            # Validate buildspec params match hardcoded logic for vLLM
            is_arm64 = "arm64" in self.image_uri
            expected_instance_type = "g5g.16xlarge" if is_arm64 else "p4d.24xlarge"
            expected_node_count = 1 if is_arm64 else 2
            
            if self.instance_type and self.instance_type != expected_instance_type:
                LOGGER.warning(
                    f"Buildspec instance_type '{self.instance_type}' differs from "
                    f"hardcoded value '{expected_instance_type}'. Using hardcoded value."
                )
            
            # Note: The platform validator already enforces node_count == 2 for multi-node EFA tests,
            # so the node_count check below would only trigger if the validator is bypassed or modified
            if self.node_count and self.node_count != expected_node_count:
                LOGGER.warning(
                    f"Buildspec node_count '{self.node_count}' differs from "
                    f"hardcoded value '{expected_node_count}'. Using hardcoded value."
                )
            
            # vLLM requires vLLM-specific setup (FSx + multi-node)
            LOGGER.info(f"Setting up vLLM infrastructure for image: {self.image_uri}")
            from infra.test_infra.ec2.vllm.setup_ec2 import setup as vllm_setup

            self.resources = vllm_setup(self.image_uri)
            LOGGER.info("vLLM setup completed successfully")
        else:
            # standard EC2 setup for other frameworks
            LOGGER.info(f"Would call standard EC2 setup for image: {self.image_uri}")
            self._standard_ec2_setup(params)

    def execute_command(self, cmd):
        """
        Execute a test command with proper environment setup.
        Raises exception immediately if command fails.
        """
        try:
            # Set up environment variables for all commands
            env = {
                "AWS_REGION": self.region,
                "BUILD_CONTEXT": self.build_context,
                "DLC_IMAGE": self.image_uri,
                "ARCH_TYPE": self.arch_type,
                "FRAMEWORK": self.framework,
            }

            # Check registry for framework-specific handling
            test_config = TEST_REGISTRY.get(self.framework)
            
            if test_config and test_config.get("requires_resources"):
                # Direct Python function call for tests requiring resource access
                LOGGER.info(f"Executing {self.framework} test via direct call: {cmd}")
                
                module_path = test_config["module"]
                function_name = test_config["function"]
                
                # Dynamically import and call the test function
                module = __import__(module_path, fromlist=[function_name])
                test_function = getattr(module, function_name)
                
                # Pass resources and image_uri directly
                test_function(self.resources, self.image_uri)
                LOGGER.info(f"Command completed successfully: {cmd}")
            else:
                # Standard shell command execution
                repo_root = get_cloned_folder_path()

                with self.ctx.cd(repo_root):
                    LOGGER.info(f"Executing command from {repo_root} with EC2 environment: {cmd}")
                    self.ctx.run(cmd, env=env)
                    LOGGER.info(f"Command completed successfully: {cmd}")

        except Exception as e:
            raise RuntimeError(f"Failed to execute command: {cmd}\nError: {str(e)}") from e

    def cleanup(self):
        """
        Cleanup EC2 resources
        """
        if not self.resources:
            LOGGER.info("No resources to cleanup")
            return

        if self.framework == "vllm":
            LOGGER.info("Cleaning up vLLM resources")

            cleanup_timer = threading.Timer(
                1000,
                lambda: LOGGER.warning(
                    "Cleanup timed out, some resources might need manual cleanup"
                ),
            )
            cleanup_timer.start()

            try:
                from infra.test_infra.ec2.vllm.setup_ec2 import cleanup_resources
                from infra.test_infra.ec2.vllm.fsx_utils import FsxSetup
                from infra.test_infra.ec2.utils import get_ec2_client

                ec2_client = get_ec2_client(self.region)
                fsx = FsxSetup(self.region)
                cleanup_resources(ec2_client, self.resources, fsx)
                cleanup_timer.cancel()
                LOGGER.info("vLLM cleanup completed successfully")
            except Exception as e:
                LOGGER.error(f"Error during vLLM cleanup: {e}")
                raise
            finally:
                cleanup_timer.cancel()
        else:
            LOGGER.info("Standard EC2 cleanup not yet implemented")

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
