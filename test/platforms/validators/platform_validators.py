from typing import List, Dict
from test.platforms.validators.base_platform_validator import BasePlatformValidator
from test.platforms.validators.platform_configs import EC2Config, EKSConfig


class EC2MultiNodeValidator(BasePlatformValidator):
    config_class = EC2Config
    SUPPORTED_INSTANCE_TYPES = {"p4d.24xlarge", "p4de.24xlarge", "p5.48xlarge"}

    def validate_config_specific(self, config: EC2Config, test_config: Dict) -> List[str]:
        errors = []
        if config.instance_type not in self.SUPPORTED_INSTANCE_TYPES:
            errors.append(
                f"Instance type must be one of {sorted(self.SUPPORTED_INSTANCE_TYPES)}, got: {config.instance_type}"
            )
        if not config.node_count or config.node_count != 2:
            errors.append("Multi-node EFA tests require node_count == 2")
        return errors


class EKSValidator(BasePlatformValidator):
    config_class = EKSConfig

    def validate_config_specific(self, config: EKSConfig, test_config: Dict) -> List[str]:
        errors = []
        framework = test_config.get("globals", {}).get("framework", "unknown")
        if config.namespace != framework:
            errors.append(
                f"namespace must match framework: expected '{framework}', got '{config.namespace}'"
            )
        return errors
