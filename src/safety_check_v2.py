"""
Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import sys
import argparse
from datetime import datetime
import pkg_resources
import safety as Safety
from safety import safety

TO_BE_DECIDED="To Be Decided"

def main():
    """Return json.dumps() string of below list structure.
    [
        {
            "package": "package",
            "scan_status": "SUCCEEDED/FAILED/IGNORED",
            "installed": "version",
            "vulnerabilities": [
                {
                    "vid": "safety_vulnerability_id",
                    "advisory": "description of the issue",
                    "reason_to_ignore":"reason to ignore the vid",
                    "spec": "version_spec"
                },
                ...
            ]
        }
        ...
    ]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--key",
        required=False,
    )
    parser.add_argument(
        "--ignore_dict_str",
        required=False,
    )
    args = parser.parse_args()


    safety_key = args.key
    ignore_dict = {}
    if args.ignore_dict_str is not None:
        ignore_dict = json.loads(args.ignore_dict_str)

    if Safety.__version__ == "1.8.7":
        proxy_dictionary = {}
    else:
        from safety.util import get_proxy_dict
        proxy_dictionary = get_proxy_dict("http", None, 80)

    # Get safety result
    packages = [
        d for d in pkg_resources.working_set if d.key not in {"python", "wsgiref", "argparse"}
    ]
    # return: Vulnerability(namedtuple("Vulnerability", ["name", "spec", "version", "advisory", "vuln_id", "cvssv2", "cvssv3"]))
    vulns = safety.check(
        packages=packages,
        key=safety_key,
        db_mirror="",
        cached=True,
        ignore_ids=[],
        proxy=proxy_dictionary,
    )

    # Generate output
    vulns_dict = {}
    vulns_list = []
    ignored_vulnerability_count = {}
    # Report for unsafe packages
    # "FAILED", "SUCCEEDED" , "IGNORED"
    for v in vulns:
        package = v.name
        spec = v.spec
        installed = v.version
        advisory = v.advisory
        vid = v.vuln_id
        vulnerability_details = {"vid": vid, "advisory": advisory, "spec": spec, "reason_to_ignore":"N/A"}

        if package not in ignored_vulnerability_count:
            ignored_vulnerability_count[package] = 0

        if vid in ignore_dict:
            vulnerability_details["reason_to_ignore"] = ignore_dict[vid]
            ignored_vulnerability_count[package] += 1

        if package not in vulns_dict:
            vulns_dict[package] = {
                "package": package,
                "scan_status": TO_BE_DECIDED,
                "installed": installed,
                "vulnerabilities": [vulnerability_details],
            }
        else:
            vulns_dict[package]["vulnerabilities"].append(vulnerability_details)

    # Report for safe packages
    timestamp = datetime.now().strftime("%d%m%Y")
    for pkg in packages:
        if pkg.key not in vulns_dict:
            vulns_dict[pkg.key] = {
                "package": pkg.key,
                "scan_status": "SUCCEEDED",
                "installed": pkg.version,
                "vulnerabilities": [{"vid": "N/A", "advisory": "N/A", "reason_to_ignore":"N/A", "spec":"N/A"}],
            }
    
    for (k, v) in vulns_dict.items():
        if v["scan_status"] == TO_BE_DECIDED:
            if len(v["vulnerabilities"]) == ignored_vulnerability_count[k]:
                v["scan_status"] = "IGNORED"
            else:
                v["scan_status"] = "FAILED"
        vulns_list.append(v)

    print(json.dumps(vulns_list))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
