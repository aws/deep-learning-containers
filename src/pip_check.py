"""
Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"). You
may not use this file except in compliance with the License. A copy of
the License is located at

    http://aws.amazon.com/apache2.0/

or in the "license" file accompanying this file. This file is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
"""

import re
import subprocess
from dataclasses import dataclass, asdict
from typing import List, Tuple


"""
Module to run pip-check and parse its output.

Since the API is internal, we rely on the output of the `pip check` command. It
has the additional benefit, that we can execute pip check in some other context
(e.g. docker).

The command checks whether each installed package has all its required packages
installed and in a second step if the version requirements match::

    $ python -m pip check
    pyramid 1.5.2 requires WebOb, which is not installed.

    $ python -m pip check
    pyramid 1.5.2 has requirement WebOb>=1.3.1, but you have WebOb 0.8.

See: https://pip.pypa.io/en/stable/cli/pip_check/

"""


def select(keys, dct):
    return {key: dct[key] for key in set(keys) & dct.keys()}


def parse_version_requirement(version: str) -> Tuple[str, str]:
    """Parse version requirement output from `pip check`.

    >>> parse_version_requirement("pydantic!=1.7,!=1.7.1,!=1.7.2")
    ('pydantic', '!=1.7,!=1.7.1,!=1.7.2'>)

    """

    return re.match(
        r"(?P<package>.+?)(?P<version>(?:~=|==|!=|<=|>=|<|>|===).+)",
        version,
    ).groups()


@dataclass
class MissingRequirement:
    package: str
    version: str
    missing_package: str

    def matches(self, spec: dict):
        return select(spec, asdict(self)) == spec

    def format(self) -> str:
        return (
            f"{self.package} {self.version} requires "
            f"{self.missing_package}, which is not installed."
        )


@dataclass
class ConflictRequirement:
    package: str
    version: str
    conflict_package: str
    required_version: str
    installed_version: str

    def matches(self, spec: dict):
        return select(spec, asdict(self)) == spec

    def format(self) -> str:
        return (
            f"{self.package} {self.version} has requirement "
            f"{self.conflict_package}{self.required_version}, but you have "
            f"{self.conflict_package} {self.installed_version}."
        )


def filter_conflicts(conflicts, specs):
    for conflict in conflicts:
        if not any(conflict.matches(spec) for spec in specs):
            yield conflict


def parse_pip_check_output(output: str):
    for line in output.splitlines():
        tokens = line.rstrip(".").lower().split()
        package, version = tokens[:2]

        if tokens[2] == "requires":
            yield MissingRequirement(package, version, tokens[3].rstrip(","))

        elif tokens[2:4] == ["has", "requirement"]:
            conflict_package, required_version = parse_version_requirement(tokens[4].rstrip(",;"))

            yield ConflictRequirement(
                package,
                version,
                conflict_package=conflict_package,
                required_version=required_version,
                installed_version=tokens[-1],
            )
            requires = tokens[4]

        else:
            raise ValueError(f"Can't parse tokens {line!r}")


def run_pip_check(ignore=None, command=("pip", "check")):
    process = subprocess.run(command, stdout=subprocess.PIPE)

    conflicts = []

    if process.returncode:
        conflicts.extend(parse_pip_check_output(process.stdout.decode()))

    if ignore:
        conflicts = list(filter_conflicts(conflicts, ignore))

    return conflicts


if __name__ == "__main__":
    for conflict in run_pip_check():
        print(conflict.format())
