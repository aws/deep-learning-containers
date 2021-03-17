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
import re
import os
import json
import datetime
from os import listdir
import argparse
import sys
import subprocess
from pull_missing_info import PullMissingInformation
from create_license_attribution_text_file import CreateLicenseAttribution
from manage_licenses_warnings import ManageLicensesWarnings
from dataclasses import dataclass

@dataclass
class LicenseEntry(object):
    Name: str
    Author: str
    Version: str
    URL: str
    License: str
    LicenseFile: str
    LicenseText: str
    Internal_LicenseFile: str
    Internal_LicenseText: str
    NoticeFile: str
    NoticeText: str
    Internal_NoticeFile: str
    Internal_NoticeText: str

license_txt_file_name = "PYTHON_PACKAGES_LICENSES"
license_json_file_name = "PACKAGES_LICENSES_JSON"

home_dir = "/Users/jspatil/workspace/deep-learning-containers/src/"
python_packages_dir = os.path.join(home_dir, "oss_compliance", "python_packages")

dir_paths = {
    "piplicenses": os.path.join(python_packages_dir, "piplicenses"),
    "generate_licenses": os.path.join(python_packages_dir, "generate_licenses"),
    "datafiles": os.path.join(python_packages_dir, "generate_licenses/datafiles"),
    "licenses": os.path.join(python_packages_dir, "generate_licenses/datafiles", "licenses"),
    "notices": os.path.join(python_packages_dir, "generate_licenses/datafiles", "notices"),
}

# files needed to process input json file generated from piplicenses
file_paths = {
    "license_categories": os.path.join(dir_paths["datafiles"], "missing_packages_info", "license_categories.json"),
    "blessed_packages": os.path.join(dir_paths["datafiles"], "missing_packages_info", "blessed_packages.json"),
    "missing_licenses_info": os.path.join(dir_paths["datafiles"], "missing_packages_info", "missing_licenses_info.json"),
    "license_json_file": os.path.join(dir_paths["piplicenses"], license_json_file_name + ".json")
}

assert all(os.path.exists(path) for path in file_paths.values())

license_categories = json.load(open(file_paths["license_categories"]))

files_open = {
    "license_categories":json.load(open(file_paths["license_categories"])),
    "blessed_packages": json.load(open(file_paths["blessed_packages"]))["blessed_packages"],
    "missing_licenses_info": json.load(open(file_paths["missing_licenses_info"])),
    "license_json_file": open(file_paths["license_json_file"])
}


class ProcessLicensesJsonFile(object):
    """Process license json file generated from pip licenses by pulling missing data from datafiles 
    and write data to final license attribution text file
    """

    def __init__(self):
        license_text_file_path = os.path.join(home_dir, license_txt_file_name)
        self.license_txt_file = open(license_text_file_path, "w+")

        # open json file for parsing
        self.package_license_data = json.load(files_open["license_json_file"])
        self.package_license_data = [LicenseEntry(**i) for i in self.package_license_data]

    def process_licenses_json_file(self):
        """Generate license file for docker image
        """
        manage_licenses_warnings = ManageLicensesWarnings(dir_paths)

        for entry in self.package_license_data:
            
            copyright_info = []
            entry.License = entry.License.split("+")
            # pull missing information in json file from datafiles
            entry, copyright_info = PullMissingInformation(entry, dir_paths, files_open).pull_missing_info()
            # write data to license text file
            CreateLicenseAttribution(entry, files_open, manage_licenses_warnings).create_license_attribution_file(self.license_txt_file, entry, copyright_info)

        manage_licenses_warnings.print_licenses_warnings()

def main():
    args = parse_args()
    ProcessLicensesJsonFile().process_licenses_json_file()


def parse_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
