import os
import json
import subprocess

class ManageLicensesWarnings(object):

    licenses_warnings = {}

    def __init__(self, dir_paths):
        self.dir_paths = dir_paths
        self.warnings_json_file_path = os.path.join(self.dir_paths["generate_licenses"], "_generate_licenses_warnings.json")

    def generate_licenses_warnings(self, key, text):
        """create list of warnings received during input json file processing
        """
        if key in self.licenses_warnings:
            self.licenses_warnings[key].append(text)
        else:
            self.licenses_warnings[key] = [text]

    def print_licenses_warnings(self):
        """upload json file having all warnings to S3 bucket
        """
        if self.licenses_warnings:
            print(self.licenses_warnings)
