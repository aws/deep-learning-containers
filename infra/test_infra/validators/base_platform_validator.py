from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict


class BasePlatformValidator(ABC):
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)

    def validate_script_paths(self, commands: List[str]) -> List[str]:
        if not isinstance(commands, list):
            return ["'run' section must be a list of commands"]
        if not all(isinstance(cmd, str) for cmd in commands):
            return ["All commands must be strings"]

        missing_files = []
        for cmd in commands:
            parts = cmd.split()
            if len(parts) > 1 and parts[1].startswith("test/"):
                if not (self.base_path / parts[1]).exists():
                    missing_files.append(parts[1])

        return (
            ["Missing test scripts:"] + [f"  - {file}" for file in missing_files]
            if missing_files
            else []
        )

    @abstractmethod
    def validate_config_specific(self, config, test_config: Dict) -> List[str]:
        """Implement platform-specific validation rules"""
        pass

    def validate(self, test_config: Dict) -> List[str]:
        errors = []
        params = test_config.get("params", {})

        if not params:
            return ["Missing or empty params section"]

        # Validate parameter types
        for param, expected_type in self.config_class.__annotations__.items():
            value = params.get(param)
            if value is not None:
                actual_type = (
                    expected_type.__args__[0]
                    if hasattr(expected_type, "__args__")
                    else expected_type
                )
                if not isinstance(value, actual_type):
                    errors.append(
                        f"{param} must be a {actual_type.__name__}, got: {type(value).__name__}"
                    )

        # Create and validate config
        try:
            config = self.config_class(
                **{k: params.get(k) for k in self.config_class.__annotations__}
            )
            errors.extend(self.validate_config_specific(config, test_config))
        except TypeError as e:
            error_msg = str(e)
            for param in self.config_class.__annotations__:
                if param in error_msg:
                    errors.append(f"Missing required parameter: {param}")
                    break
            else:
                errors.append(f"Parameter error: {error_msg}")

        errors.extend(self.validate_script_paths(test_config.get("run", [])))
        return errors
