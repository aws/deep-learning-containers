import os
import re
import subprocess

S3_BUCKET_PATH = "dlc-package-licenses"
S3_THIRD_PARTY_OBJECT_PATH = "third_party_source_code"

class CreateLicenseAttribution(object):

    def __init__(self, entry, files_open, manage_licenses_warnings):
        self.entry = entry
        self.not_blessed_licenses = files_open["license_categories"]["not_blessed_licenses"]
        self.blessed_packages = files_open["blessed_packages"]
        self.public_licenses = files_open["license_categories"]["public_licenses"]
        self.third_party_source_code_needed_licenses = files_open["license_categories"]["third_party_source_code_needed_licenses"]
        self.manage_licenses_warnings = manage_licenses_warnings
        self.copyright_regex = r'(?:©|\(c\)|copyright\b)\s*(\d{4})'
        self.cloned_folder_name = f"{self.entry.Name}_v{self.entry.Version}_source_code"

    def create_license_attribution_file(self, license_txt_file, entry, copyright_info):
        """write each entry in json file to license attribution text file
        """

        if 'UNKNOWN' not in entry.LicenseText:
            self.check_if_license_is_blessed()

            license_txt_file.write("Package Name: ")
            license_txt_file.write(entry.Name)
            license_txt_file.write("\n")
            license_txt_file.write("Package Version: ")
            license_txt_file.write(entry.Version)
            license_txt_file.write("\n")

            if 'UNKNOWN' not in entry.License:
                for row in entry.License:
                    if 'UNKNOWN' not in entry.URL:
                        license_txt_file.write("Package URL: ")
                        license_txt_file.write(entry.URL)
                        license_txt_file.write("\n")
                        if any(license in row for license in self.third_party_source_code_needed_licenses):
                            self.upload_third_party_source_code_needed_licenses_to_s3()
                            license_txt_file.write("Package source code URL: ")
                            license_txt_file.write("https://{0}.s3.amazonaws.com/{1}/{2}.tar.gz".format(S3_BUCKET_PATH, S3_THIRD_PARTY_OBJECT_PATH, self.cloned_folder_name))
                            license_txt_file.write("\n")
                    else:
                        self.manage_licenses_warnings.generate_licenses_warnings("no_url", "package {} has no url info but it is not needed".format(entry.Name))

                    license_txt_file.write("Package License: ")
                    license_txt_file.write(row)
                    license_txt_file.write("\n")

            if 'UNKNOWN' not in entry.LicenseFile:
                for row in entry.LicenseFile:
                    license_txt_file.write("Package License File: ")
                    license_txt_file.write(row)
                    license_txt_file.write("\n")

            if 'UNKNOWN' not in entry.NoticeFile:
                for row in entry.NoticeFile:
                    license_txt_file.write("Package Notice File: ")
                    license_txt_file.write(row)
                    license_txt_file.write("\n")

            copyright_check = False
            if 'UNKNOWN' not in entry.NoticeText:
                for row in entry.NoticeText:
                    if re.search(self.copyright_regex, row.lower()):
                        copyright_check = True
                    license_txt_file.write("Package Notice Text: \n")
                    license_txt_file.write(row.encode("utf8").decode("utf-8"))
                    license_txt_file.write("\n")
            elif 'Apache' in entry.License:
                self.manage_licenses_warnings.generate_licenses_warnings("no_notice", "package {} has no notice info".format(entry.Name))

            for row in entry.LicenseText:                
                row = row.encode("utf8").decode("utf-8")
                # if "cmake" == entry.Name:
                #     print(row.lower())
                #     print(entry.Name)
                # check whether copyright is present in license text
                if re.search(self.copyright_regex, row.lower()) is None and not copyright_check:
                    if copyright_info:
                        license_txt_file.write("Package Copyright Text: \n")
                        for copyright_row in copyright_info:
                            license_txt_file.write(copyright_row)
                            license_txt_file.write("\n")
                        copyright_check = True
                else:
                    copyright_check = True
            if not copyright_check:
                self.manage_licenses_warnings.generate_licenses_warnings("no_copyright", "package {} has no copyright info".format(entry.Name))

            for row in entry.LicenseText:                
                row = row.encode("utf8").decode("utf-8") 
                license_txt_file.write("Package License Text: \n")
                license_txt_file.write(row.encode("utf8").decode("utf-8"))
                license_txt_file.write("\n\n")

        else:
            for row in entry.License:
                if row.lower() not in self.public_licenses:
                    self.manage_licenses_warnings.generate_licenses_warnings("no_license", "package {} has missing license text.".format(entry.Name))

        self.add_internal_license_attributions(license_txt_file)

    def add_internal_license_attributions(self, license_txt_file):
        """write internal license attributions in each entry of json file to license attribution text file
        """
        copyright_check = False
        if 'UNKNOWN' not in self.entry.Internal_LicenseText:
            license_txt_file.write("Below are Internal (third party) Package Licenses of package {}: \n".format(self.entry.Name))
            for i in range(len(self.entry.Internal_LicenseText)):
                license_txt_file.write("Internal (third party) Package License File of package {}: \n".format(self.entry.Name))
                license_txt_file.write(self.entry.Internal_LicenseFile[i])
                license_txt_file.write("\n\n")
                if re.search(self.copyright_regex, self.entry.Internal_LicenseText[i].lower()):
                    copyright_check = True
                license_txt_file.write("Internal (third party) Package License Text of package {}: \n".format(self.entry.Name))
                license_txt_file.write(self.entry.Internal_LicenseText[i])
                license_txt_file.write("\n\n")
        if 'UNKNOWN' not in self.entry.Internal_NoticeText:
            for i in range(len(self.entry.Internal_NoticeText)):
                license_txt_file.write("Internal (third party) Package Notice File: \n")
                license_txt_file.write(self.entry.Internal_NoticeFile[i])
                license_txt_file.write("\n\n")
                if re.search(self.copyright_regex, self.entry.Internal_NoticeText[i].lower()):
                    copyright_check = True
                license_txt_file.write("Internal (third party) Package Notice Text: \n")
                license_txt_file.write(self.entry.Internal_NoticeText[i])
                license_txt_file.write("\n\n")

        if 'UNKNOWN' not in self.entry.Internal_LicenseText:
            if not copyright_check:
                license_txt_file.write("No copyright statement was found in third party licenses for this project.\n")
                license_txt_file.write("\n\n")

    def check_if_license_is_blessed(self):
        for row in self.entry.License:
            if row in self.not_blessed_licenses and self.entry.Name not in self.blessed_packages:
                self.manage_licenses_warnings.generate_licenses_warnings("not_blessed_license", "package {} has not blessed License {}".format(self.entry.Name, row))

    def upload_third_party_source_code_needed_licenses_to_s3(self):
        if 'github' in self.entry.URL:
            if 'master' not in self.entry.URL:
                if not os.path.exists(self.cloned_folder_name):
                    subprocess.check_output(["bash", "-c", "git clone {0} {1}".format(self.entry.URL, self.cloned_folder_name)])
                    subprocess.check_output(["bash", "-c", "tar -czvf {0}.tar.gz {0}".format(self.cloned_folder_name)])
                    try:
                        subprocess.check_call(["bash", "-c", "aws s3api head-object --bucket {0} --key {1}/{2}.tar.gz".format(S3_BUCKET_PATH, S3_THIRD_PARTY_OBJECT_PATH, self.cloned_folder_name)])
                    except subprocess.CalledProcessError as e:
                        print("Uploading third party source code needed licenses: {}".format(self.cloned_folder_name))
                        subprocess.check_output(["bash", "-c", "aws s3 cp {0}.tar.gz s3://{1}/{2}/{0}.tar.gz"
                                                    .format(self.cloned_folder_name, S3_BUCKET_PATH, S3_THIRD_PARTY_OBJECT_PATH)])
            else:
                self.manage_licenses_warnings.generate_licenses_warnings("cant_clone_repo", "package {} repo cant be cloned as url is not supported for git clone {}".format(self.entry.Name, self.entry.URL))
        else:
            self.manage_licenses_warnings.generate_licenses_warnings("no_github_repo", "package {} has no github url to git clone {}".format(self.entry.Name, self.entry.URL))
