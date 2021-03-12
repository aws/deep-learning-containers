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
from dataclasses import dataclass

@dataclass
class LicenseEntry(object):
    Name: str
    Version: str
    LicenseLocation: str
    LicenseText: str

license_txt_file_name = "LINUX_PACKAGES_LICENSES"

home_dir = "/"

class License_attribution_file(object):

    def __init__(self):
        license_text_file_path = os.path.join(home_dir, license_txt_file_name)
        self.license_txt_file = open(license_text_file_path, "w+")

    def create_license_attribution_file(self):
        output = subprocess.check_output("dpkg -l | grep '^.[iufhwt]'", shell=True, universal_newlines=True)
        output_list = output.split("\n")
        for packages in output_list:
            package_details = ' '.join(packages.split()).split(' ')
            if '' in package_details:
                package_details = ' '.join(packages.split()).split(' ').remove('')

            if package_details:
                package_name = package_details[1].split(':')[0]

                try:
                    license_text = subprocess.check_output("cat /usr/share/doc/{}/copyright".format(package_name), shell=True, universal_newlines=True)
                    license_text = license_text.encode("utf8").decode("utf-8")
                except subprocess.CalledProcessError:
                    license_text = "License is not present for this package"

                entry = LicenseEntry(package_name, package_details[2], "/usr/share/doc/{}/copyright".format(package_name), license_text)
                
                self.license_txt_file.write("Package Name: ")
                self.license_txt_file.write(entry.Name)
                self.license_txt_file.write("\n")
                self.license_txt_file.write("Package Version: ")
                self.license_txt_file.write(entry.Version)
                self.license_txt_file.write("\n")
    
                self.license_txt_file.write("Package License Location: ")
                self.license_txt_file.write("/usr/share/doc/{}/copyright".format(entry.LicenseLocation))
                self.license_txt_file.write("\n")
                self.license_txt_file.write("Package License Text: ")
                self.license_txt_file.write("\n")
                self.license_txt_file.write(entry.LicenseText)
                self.license_txt_file.write("\n")
                self.license_txt_file.write("\n")
                self.license_txt_file.write("-------------------************************************-------------------")
                self.license_txt_file.write("\n")
                self.license_txt_file.write("\n")

def main():
    args = parse_args()
    License_attribution_file().create_license_attribution_file()


def parse_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
