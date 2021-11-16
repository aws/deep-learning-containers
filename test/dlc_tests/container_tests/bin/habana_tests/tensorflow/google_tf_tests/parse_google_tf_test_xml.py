#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

# Utility script to parse all the google tf test report pytest xml files and generate a single csv file
# showing the statistics corresponding to each the xml files.
#
# Example generated csv file as below:
# TF Python Tests,Total Tests,Pass Tests,Error Tests,Failed Tests,Skipped Tests,Duration
# dlpack,76,45,0,30,1,3.455
# eager,1243,1016,8,88,131,2552.161
# framework,1498,1310,3,27,158, 1115.621
# autograph,843,774,4,12,53,50.101

import os
import sys
import xml.etree.ElementTree as ET

def get_xml_files(directory, extension):
    return (f for f in os.listdir(directory) if f.endswith('.' + extension))

def parse_xml_file(directory, xml_f):
    # xfail_expected_str to search in the xml files to calculate count of the xfailed tests
    xfail_expected_str = '<skipped message="expected test failure">'
    xfail_tests    = 0

    error_tests    = 0
    failed_tests   = 0
    # Tests marked as xfail are counted in skips in the pytest xml file
    # so we will subtract the xfail tests count from the skipped tests count.
    skipped_tests  = 0
    total_tests    = 0
    # pass_tests will be (total_tests - (skip_count + failures_count + errors_count))
    pass_tests     = 0
    duration       = 0

    # Get the xfail count by searching for xfail_expected_str
    with open(directory + xml_f, "r") as file:
        xfail_tests = file.read().count(xfail_expected_str)

    # https://docs.python.org/3/library/xml.etree.elementtree.html
    # Below execution will get below information (as an example) in a dict
    # <testsuite errors="8" failures="1" name="pytest" skips="218" tests="1243" time="2552.161">
    root = ET.parse(directory + xml_f).getroot()
    testsuite_list = []
    if (root.tag == 'testsuite'):
        testsuite_list.append(root.attrib)
    else:
        for child in root:
            if (child.tag == 'testsuite'):
                testsuite_list.append(child.attrib)

    for testsuite_dict in testsuite_list:
        if 'errors' in testsuite_dict:
            error_tests += int(testsuite_dict['errors'])

        if 'failures' in testsuite_dict:
            failed_tests += int(testsuite_dict['failures'])

        # Some pytest reports have 'skips' and some have 'skipped' depending upon pytest versions
        if 'skips' in testsuite_dict:
            skipped_tests += int(testsuite_dict['skips'])

        if 'skipped' in testsuite_dict:
            skipped_tests += int(testsuite_dict['skipped'])

        if 'tests' in testsuite_dict:
            total_tests += int(testsuite_dict['tests'])

        if 'time' in testsuite_dict:
            duration += float(testsuite_dict['time'])

    skipped_tests -= xfail_tests
    pass_tests     = (total_tests - (skipped_tests + xfail_tests + failed_tests + error_tests))

    return f"{total_tests},{pass_tests},{error_tests},{xfail_tests},{failed_tests},{skipped_tests},{duration}"

if __name__ == "__main__":

    # Path to the xml files is passed as an argument
    try:
        google_tf_xml_dir = sys.argv[1]
    except:
        google_tf_xml_dir = os.getcwd()

    google_tf_xml_dir += "/"

    # Get all the xml files in a list
    google_tf_xml_files = get_xml_files(google_tf_xml_dir, "xml")

    # Set the csv file name to be generated
    google_tf_csv_file  = os.getcwd() + "/google_tf_test_report.csv"

    # Create/truncate google_tf_csv_file and add the header information
    with open(google_tf_csv_file, "w") as file:
        file.write("TF Python Tests,Total Tests,Pass Tests,Error Tests,xFailed Tests,Failed Tests,Skipped Tests,Duration\n")

    # Parsing loop for the xml files
    for xml_file in sorted(google_tf_xml_files):

        # Below will give us autograph, kernel_tests, ops, keras etc
        # max split is set to 4 based upon google_tf_test_report_kernel_tests.xml so that we get kernel_tests too
        tf_python_test = xml_file.rpartition('.')[0].split('_', 4)[-1]

        stats = parse_xml_file(google_tf_xml_dir, xml_file)

        with open(google_tf_csv_file, "a") as file:
            # Appending statistics to the file
            file.write(f"{tf_python_test},{stats}\n")
