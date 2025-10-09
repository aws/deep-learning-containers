from typing import List, Dict
from test.platforms.validators.base_platform_validator import BasePlatformValidator
from test.platforms.validators.platform_configs import EC2Config, EKSConfig


class EC2MultiNodeValidator(BasePlatformValidator):
    def validate(self, test_config: Dict) -> List[str]:
        errors = []
        params = test_config.get("params", {})

        try:
            config = EC2Config(
                instance_type=params.get("instance_type"), node_count=params.get("node_count")
            )
            # Additional validation for multi-node specific requirements
            if not config.node_count or config.node_count < 2:
                errors.append("Multi-node EFA tests require node_count >= 2")

        except TypeError as e:
            errors.append(f"Parameter type error: {str(e)}")
        except Exception as e:
            errors.append(f"Invalid parameters: {str(e)}")

        missing_files = self.validate_script_paths(test_config.get("run", []))
        if missing_files:
            errors.extend(missing_files)

        return errors


class EKSValidator(BasePlatformValidator):
    def validate(self, test_config: Dict) -> List[str]:
        errors = []
        params = test_config.get("params", {})

        try:
            EKSConfig(cluster=params.get("cluster"), namespace=params.get("namespace"))
        except TypeError as e:
            errors.append(f"Parameter type error: {str(e)}")
        except Exception as e:
            errors.append(f"Invalid parameters: {str(e)}")

        missing_files = self.validate_script_paths(test_config.get("run", []))
        if missing_files:
            errors.extend(missing_files)

        return errors
