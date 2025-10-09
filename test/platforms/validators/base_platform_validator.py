from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict


class BasePlatformValidator(ABC):
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)

    def validate_file_exists(self, script_path: str) -> bool:
        full_path = self.base_path / script_path
        return full_path.exists()

    def validate_script_paths(self, commands: List[str]) -> List[str]:
        errors = []

        if not isinstance(commands, list):
            return ["'run' section must be a list of commands"]

        if not all(isinstance(cmd, str) for cmd in commands):
            return ["All commands must be strings"]

        missing_files = []
        for cmd in commands:
            parts = cmd.split()
            if len(parts) > 1:
                script_path = parts[1]
                # Only validate paths under test/
                if script_path.startswith("test/"):
                    if not self.validate_file_exists(script_path):
                        missing_files.append(script_path)

        if missing_files:
            errors.append("Missing test scripts:")
            errors.extend(f"  - {file}" for file in missing_files)

        return errors

    @abstractmethod
    def validate(self, test_config: Dict) -> List[str]:
        pass
