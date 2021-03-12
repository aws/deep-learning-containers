import os

class PullMissingInformation(object):

    def __init__(self, entry, dir_paths, files_open):
        self.entry = entry
        self.public_licenses = files_open["license_categories"]["public_licenses"]
        self.missing_licenses_info = files_open["missing_licenses_info"]
        self.dir_paths = dir_paths
        self.license_file_names = []
        for (dirpath, dirnames, filenames) in os.walk(dir_paths["licenses"]):
            self.license_file_names += [file for file in filenames]

    def pull_missing_info(self):
        """Update input json entry by pulling missing info from datafiles
        """
        copyright_info = []
        if self.entry.Name in self.missing_licenses_info.keys():
            copyright_info = self.pull_missing_copyright_info()
            self.pull_missing_notice()
            self.pull_missing_license_type()
            self.pull_missing_license()
            self.pull_missing_source_code_url()
        return self.entry, copyright_info


    def pull_missing_copyright_info(self):
        """Update input json entry by pulling missing copyright info from datafiles
        """
        copyright_info = []
        if 'copyright_info' in self.missing_licenses_info[self.entry.Name]:
            copyright_info = self.missing_licenses_info[self.entry.Name]["copyright_info"]
        return copyright_info


    def pull_missing_notice(self):
        """Update input json entry by pulling missing notice info from datafiles
        """
        if 'notice' in self.missing_licenses_info[self.entry.Name]:
            notice = self.missing_licenses_info[self.entry.Name]["notice"]
            # Each file in notices folder has file with suffix '_notice'. 
            # In missing_licenses_info.json, 'packagename_notice' is present under field 'notice'
            if "_notice" in notice:
                with open(os.path.join(self.dir_paths["notices"], notice),'r') as f:
                    self.entry.NoticeText = [f.read()]
            else:
                self.entry.NoticeText = [notice]


    def pull_missing_license_type(self):
        """Update input json entry by pulling missing license_type info from datafiles
        """
        if "license_type" in self.missing_licenses_info[self.entry.Name]:
            self.entry.License = self.missing_licenses_info[self.entry.Name]["license_type"].split("+")

    def pull_missing_license(self):
        """Update input json entry by pulling missing license texts info from datafiles
        """
        if 'UNKNOWN' in self.entry.LicenseText:
            for row in self.entry.License:
                # self.entry.License has list of license types which is used to pull license texts from 'licenses' folder
                if row.lower() not in self.public_licenses:
                    for file in self.license_file_names:
                        # if file name in licenses folder matches woth license type in self.entry.License, then read file
                        if file.lower() == row.lower():
                            if '_license' in row:
                                # Each file in 'custom_licenses' folder has file with suffix '_license'.
                                filepath = os.path.join(self.dir_paths["licenses"], "custom_licenses", row)
                            else:
                                filepath = os.path.join(self.dir_paths["licenses"], "generic_licenses", row)
                            with open(filepath,'r') as f:
                                if 'UNKNOWN' in self.entry.LicenseText:
                                    self.entry.LicenseText = [f.read()]
                                else:
                                    self.entry.LicenseText.append(f.read())

    def pull_missing_source_code_url(self):
        """Update input json entry by pulling missing source code url info from datafiles
        """
        if 'source_code_url' in self.missing_licenses_info[self.entry.Name]:
            self.entry.URL = self.missing_licenses_info[self.entry.Name]["source_code_url"]
