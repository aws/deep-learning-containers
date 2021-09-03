from invoke.context import Context
from datetime import datetime

import json
import os


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
                    "vulnerability_id": "safety_vulnerability_id",
                    "advisory": "description of the issue",
                    "reason_to_ignore":"reason to ignore the vulnerability_id",
                    "spec": "version_spec"
                },
                ...
            ]
            "date":
        }
        ...
    ]
    """

    def __init__(self, container_id, ignore_dict={}):
        self.container_id = container_id
        self.vulnerability_dict = {}
        self.vulnerability_list = []
        self.ignore_dict = ignore_dict
        self.ignored_vulnerability_count = {}
        self.ctx = Context()
        self.docker_exec_cmd = f"docker exec -i {container_id}"
        self.safety_check_output = None

    def insert_vulnerabilites_into_report(self, scanned_vulnerabilities):
        """
        Takes the list of vulnerabilites produced by safety scan as the input and iterates through the list to insert
        the vulnerabilites into the vulnerability_dict.

        Parameters:
            scanned_vulnerabilities: list[list]

        Returns:
            None
        """
        for vulnerability in scanned_vulnerabilities:
            package, spec, installed, advisory, vulnerability_id = vulnerability[:5]
            vulnerability_details = {
                "vulnerability_id": vulnerability_id,
                "advisory": advisory,
                "spec": spec,
                "reason_to_ignore": "N/A",
            }

            if package not in self.ignored_vulnerability_count:
                self.ignored_vulnerability_count[package] = 0

            if vulnerability_id in self.ignore_dict:
                vulnerability_details["reason_to_ignore"] = self.ignore_dict[vulnerability_id]
                self.ignored_vulnerability_count[package] += 1

            if package not in self.vulnerability_dict:
                self.vulnerability_dict[package] = {
                    "package": package,
                    "scan_status": "TBD",
                    "installed": installed,
                    "vulnerabilities": [vulnerability_details],
                    "date": self.timestamp,
                }
            else:
                self.vulnerability_dict[package]["vulnerabilities"].append(vulnerability_details)

    def get_package_set_from_container(self):
        """
        Extracts package set of a container.
        """
        python_cmd_to_extract_package_set = """ python -c "import pkg_resources; \
                import json; \
                print(json.dumps([{'key':d.key, 'version':d.version} for d in pkg_resources.working_set]))" """

        run_output = self.ctx.run(f"{self.docker_exec_cmd} {python_cmd_to_extract_package_set}", hide=True, warn=True)
        if run_output.exited != 0:
            raise Exception("Package set cannot be retrieved from the container.")

        return json.loads(run_output.stdout)

    def insert_safe_packages_into_report(self, packages):
        """
        Takes the list of all the packages existing in a container and inserts safe packages into the
        vulnerability_dict.
        :param packages: list[dict], each dict looks like {"key":package_name, "version":package_version}
        """
        for pkg in packages:
            if pkg["key"] not in self.vulnerability_dict:
                self.vulnerability_dict[pkg["key"]] = {
                    "package": pkg["key"],
                    "scan_status": "SUCCEEDED",
                    "installed": pkg["version"],
                    "vulnerabilities": [
                        {"vulnerability_id": "N/A", "advisory": "N/A", "reason_to_ignore": "N/A", "spec": "N/A"}
                    ],
                    "date": self.timestamp,
                }

    def process_report(self):
        """
        Once all the packages (safe and unsafe both) have been inserted in the vulnerability_dict, this method is called.
        On being called, it processes each package within the vulnerability_dict and appends it to the vulnerability_list. 
        Before appending it checks if the scan_status is "TBD". If yes, it assigns the correct scan_status to the package.
        """
        for (package, package_scan_results) in self.vulnerability_dict.items():
            if package_scan_results["scan_status"] == "TBD":
                if len(package_scan_results["vulnerabilities"]) == self.ignored_vulnerability_count[package]:
                    package_scan_results["scan_status"] = "IGNORED"
                else:
                    package_scan_results["scan_status"] = "FAILED"
            self.vulnerability_list.append(package_scan_results)

    def run_safety_check_in_non_cb_context(self):
        """
        Runs the safety check on the container in Non-CodeBuild Context

        :return: string, A JSON formatted string containing vulnerabilities found in the container
        """
        safety_check_command = f"{self.docker_exec_cmd} safety check --json"
        run_out = self.ctx.run(safety_check_command, warn=True, hide=True)
        if run_out.return_code != 0:
            print(f"safety check command returned non-zero error code. stderr printed for logging: {run_out.stderr}")
        return run_out.stdout

    def run_safety_check_in_cb_context(self):
        """
        Runs the safety check on the container in CodeBuild Context
        """
        from dlc.safety_check import SafetyCheck

        return SafetyCheck().run_safety_check_on_container(self.docker_exec_cmd)

    def generate(self):
        """
        Acts as a driver function for this class that initiates the entire process of running safety check and returing
        the vulnerability_list
        :return: list[dict], the output follows the same format as mentioned in the description of the class
        """
        self.timestamp = datetime.now().strftime("%d-%m-%Y")
        if os.getenv("IS_CODEBUILD_IMAGE") is None:
            self.safety_check_output = self.run_safety_check_in_non_cb_context()
        elif os.getenv("IS_CODEBUILD_IMAGE").upper() == "TRUE":
            self.safety_check_output = self.run_safety_check_in_cb_context()
        # In case of errors, json.loads command will fail. We want the failure to occur to ensure that
        # build process fails in case the safety report cannot be generated properly.
        scanned_vulnerabilities = json.loads(self.safety_check_output)
        self.insert_vulnerabilites_into_report(scanned_vulnerabilities)
        packages = self.get_package_set_from_container()
        self.insert_safe_packages_into_report(packages)
        self.process_report()
        return self.vulnerability_list
