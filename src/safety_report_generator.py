from invoke.context import Context
from datetime import datetime

import json
import os

TO_BE_DECIDED="To Be Decided"

class SafetyReportGenerator:
    """
    The SafetyReportGenerator class deals with the functionality of generating safety reports for running containers.
    The safety report takes the following format:
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
            "date":
        }
        ...
    ]
    """

    def __init__(self, container_id, ignore_dict = {}):
        self.container_id = container_id
        self.vulns_dict = {}
        self.vulns_list = []
        self.ignore_dict = ignore_dict
        self.ignored_vulnerability_count = {}
        self.ctx = Context()
        self.docker_exec_cmd = f"docker exec -i {container_id}"
        self.safety_check_output = None
    
    def insert_vulnerabilites_into_report(self, vulns):
        for v in vulns:
            package = v[0]
            spec = v[1]
            installed = v[2]
            advisory = v[3]
            vid = v[4]
            vulnerability_details = {"vid": vid, "advisory": advisory, "spec": spec, "reason_to_ignore":"N/A"}

            if package not in self.ignored_vulnerability_count:
                self.ignored_vulnerability_count[package] = 0

            if vid in self.ignore_dict:
                vulnerability_details["reason_to_ignore"] = self.ignore_dict[vid]
                self.ignored_vulnerability_count[package] += 1

            if package not in self.vulns_dict:
                self.vulns_dict[package] = {
                    "package": package,
                    "scan_status": TO_BE_DECIDED,
                    "installed": installed,
                    "vulnerabilities": [vulnerability_details],
                    "date":self.timestamp,
                }
            else:
                self.vulns_dict[package]["vulnerabilities"].append(vulnerability_details)
    
    def get_package_set_from_container(self):
        python_cmd_to_extract_package_set = """ python -c "import pkg_resources; \
                import json; \
                print(json.dumps([{'key':d.key, 'version':d.version} for d in pkg_resources.working_set]))" """
        
        run_output = self.ctx.run(f"{self.docker_exec_cmd} {python_cmd_to_extract_package_set}", hide=True, warn=True)
        if run_output.exited != 0:
            raise Exception('Package set cannot be retrieved from the container.')
        
        return json.loads(run_output.stdout)

    
    def insert_safe_packages_into_report(self, packages):
        for pkg in packages:
            if pkg['key'] not in self.vulns_dict:
                self.vulns_dict[pkg['key']] = {
                    "package": pkg['key'],
                    "scan_status": "SUCCEEDED",
                    "installed": pkg['version'],
                    "vulnerabilities": [{"vid": "N/A", "advisory": "N/A", "reason_to_ignore":"N/A", "spec":"N/A"}],
                    "date":self.timestamp
                }
    
    def process_report(self):
        for (k, v) in self.vulns_dict.items():
            if v["scan_status"] == TO_BE_DECIDED:
                if len(v["vulnerabilities"]) == self.ignored_vulnerability_count[k]:
                    v["scan_status"] = "IGNORED"
                else:
                    v["scan_status"] = "FAILED"
            self.vulns_list.append(v)

    def run_safety_check_in_non_cb_context(self):
        print('Running in Non CodeBuild Context')
        safety_check_command = f"{self.docker_exec_cmd} safety check --json"
        run_out = self.ctx.run(safety_check_command, warn=True, hide=True)
        if run_out.return_code != 0:
            print(
                f"safety check command returned non-zero error code. stderr printed for logging: {run_out.stderr}"
            )
        return run_out.stdout
    
    def run_safety_check_in_cb_context(self):
        print('Running in CodeBuild Context')
        from dlc.safety_check import SafetyCheck
        return SafetyCheck().run_safety_check_on_container(self.docker_exec_cmd)
    
    def generate(self):
        self.timestamp = datetime.now().strftime("%d-%m-%Y")
        if os.getenv('IS_CODEBUILD_IMAGE') is None:
            self.safety_check_output = self.run_safety_check_in_non_cb_context()
        elif os.getenv('IS_CODEBUILD_IMAGE').upper() == 'TRUE':
            self.safety_check_output = self.run_safety_check_in_cb_context()
        # In case of errors, json.loads command will fail. We want the failure to occur to ensure that
        # build process fails in case the safety report cannot be generated properly.
        vulns = json.loads(self.safety_check_output)
        self.insert_vulnerabilites_into_report(vulns)
        packages = self.get_package_set_from_container()
        self.insert_safe_packages_into_report(packages)
        self.process_report()
        return self.vulns_list





    



