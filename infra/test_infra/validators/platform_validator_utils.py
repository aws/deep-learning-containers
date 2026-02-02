from infra.test_infra.validators.base_platform_validator import BasePlatformValidator
from infra.test_infra.validators.platform_validators import EC2MultiNodeValidator, EKSValidator

_VALIDATORS = {"ec2-multi-node": EC2MultiNodeValidator, "eks": EKSValidator}


def get_platform_validator(platform: str, base_path: str) -> BasePlatformValidator:
    if platform.startswith("ec2") and "multi-node" in platform:
        platform_type = "ec2-multi-node"
    else:
        platform_type = platform.split("-")[0]

    validator_class = _VALIDATORS.get(platform_type)
    if not validator_class:
        raise ValueError(f"No validator found for platform: {platform}")

    return validator_class(base_path)
