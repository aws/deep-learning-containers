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
            error_msg = str(e)
            if "instance_type" in error_msg:
                param_name = "instance_type"
            elif "node_count" in error_msg:
                param_name = "node_count"
            else:
                errors.append(f"Parameter error: {error_msg}")
                return errors

            value = params.get(param_name)
            param_types = {"instance_type": "string", "node_count": "integer"}
            if value is None:
                errors.append(f"Missing required parameter: {param_name}")
            else:
                expected_type = param_types.get(param_name, "unknown type")
                errors.append(
                    f"{param_name} must be a {expected_type}, got: {type(value).__name__}"
                )
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
        globals_data = test_config.get("globals", {})
        framework = globals_data.get("framework", "unknown")

        try:
            config = EKSConfig(cluster=params.get("cluster"), namespace=params.get("namespace"))
            if config.namespace != framework:
                errors.append(
                    f"namespace must match framework: expected '{framework}', got '{config.namespace}'"
                )
        except TypeError as e:
            error_msg = str(e)
            if "cluster" in error_msg:
                param_name = "cluster"
            elif "namespace" in error_msg:
                param_name = "namespace"
            else:
                errors.append(f"Parameter error: {error_msg}")
                return errors

            value = params.get(param_name)
            param_types = {"cluster": "string", "namespace": "string"}
            if value is None:
                errors.append(f"Missing required parameter: {param_name}")
            else:
                expected_type = param_types.get(param_name, "string")
                errors.append(
                    f"{param_name} must be a {expected_type}, got: {type(value).__name__}"
                )
        except Exception as e:
            errors.append(f"Invalid parameters: {str(e)}")

        missing_files = self.validate_script_paths(test_config.get("run", []))
        if missing_files:
            errors.extend(missing_files)

        return errors
