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

import os
import shutil
import glob
import filecmp
import argparse

help_txt = 'Script copies opensource tensorflow test files to google tf dir.'

parser = argparse.ArgumentParser()
parser.add_argument("--src_path", "-sp", required=True, help="absolute path to the source tensorflow directory")
parser.add_argument("--dst_path", "-dp", required=True, help="absolute path to the destination tensorflow directory")

args = parser.parse_args()
src_tf_repo = args.src_path
dst_tf_repo = args.dst_path

identical_files = []
modified_files = []
new_files = []
obsoleted_files = []

from enum import Enum
class FileCopy(Enum):
    ALL_PYTHON_TEST_FILES = 1
    ALL_FILES = 2
    SPECIFIC_FILES = 3

def copy_files_from_tf_repo_and_keep_intermediate_paths(dir, file_copy, skip_non_existing_files = False,
                                                        org_tensorflow = False):

    # Save the cur dir
    cur_dir = os.getcwd()

    # Change to src TF dir
    os.chdir(src_tf_repo)

    # Get all the files inside src TF dir based upon FileCopy enum
    files = []
    if (file_copy == FileCopy.ALL_FILES):
        files = glob.glob(dir + '/**', recursive=True)
    elif (file_copy == FileCopy.ALL_PYTHON_TEST_FILES):
        files = glob.glob(dir + '/**/*_test.py', recursive=True)
    elif (file_copy == FileCopy.SPECIFIC_FILES):
        files = dir

    # Revert to cur dir
    os.chdir(cur_dir)

    # Change to dst tf dir where we want the relevant files from the src TF dir
    os.chdir(dst_tf_repo)

    for file_path in files:
        tf_file_path = f'{src_tf_repo}/{file_path}'
        if (os.path.isfile(tf_file_path)):
            src_relative_dir, file_name = file_path.rsplit('/', 1)
            if (org_tensorflow):
                # Prepend org_tensorflow to the relative path
                dst_relative_dir = "org_tensorflow/" + src_relative_dir
            else:
                dst_relative_dir = src_relative_dir

            # Create dst relative dir
            os.makedirs(dst_relative_dir, exist_ok=True)

            dst_tf_file_path = f'{dst_relative_dir}/{file_name}'

            if skip_non_existing_files:
                if not os.path.exists(tf_file_path):
                    continue
            try:
                if os.path.exists(dst_tf_file_path) and not os.path.exists(tf_file_path):
                    print(f"Only in {dst_tf_file_path}. Hence removing it")
                    os.remove(dst_tf_file_path)
                    obsoleted_files.append(dst_tf_file_path)
                elif not os.path.exists(dst_tf_file_path) and os.path.exists(tf_file_path):
                    print(f"Only in {tf_file_path}. Hence copying it")
                    shutil.copy2(tf_file_path, f'./{dst_relative_dir}/')
                    new_files.append(dst_tf_file_path)
                elif os.path.exists(dst_tf_file_path) and filecmp.cmp(tf_file_path, dst_tf_file_path):
                    print(f"Files {tf_file_path} and {dst_tf_file_path} are identical")
                    identical_files.append(dst_tf_file_path)
                else:
                    print(f"Files {tf_file_path} and {dst_tf_file_path} differ")
                    shutil.copy2(tf_file_path, f'./{dst_relative_dir}/')
                    modified_files.append(dst_tf_file_path)
            except shutil.SameFileError:
                pass
    # Revert to cur dir
    os.chdir(cur_dir)

# Unused right now; double check
def copy_init_files():
    # refresh all required __init__.py files
    py_files = glob.glob('tensorflow/**/*.py', recursive=True)

    all_dirs = set()
    for file_path in py_files:
        folders = file_path.split('/')[:-1] # skip filename
        for i in range(len(folders), 1, -1):
            directory = '/'.join(folders[0:i])
            if directory in all_dirs:
                break
            else:
                all_dirs.add(directory)

    print(all_dirs)
    copy_files_from_tf_repo_and_keep_intermediate_paths(
        [f'{directory}/__init__.py' for directory in all_dirs],
        skip_non_existing_files=True
    )

def create_init_files(dirs_list):

    for dir in dirs_list:
        dir_full_path = f'{dst_tf_repo}/{dir}'
        if (os.path.exists(dir_full_path)):
            # Creating a file at specified location
            print(f"Creating __init__.py inside {dir}")
            with open(os.path.join(dir_full_path, '__init__.py'), 'w') as fp:
                pass

# find interesting files and print their relative path to tf_repo
python_tests_dirs = [
    'tensorflow/python/autograph',
    'tensorflow/python/client',
    'tensorflow/python/compat',
    'tensorflow/python/compiler',
    'tensorflow/python/data',
    'tensorflow/python/debug',
    'tensorflow/python/distribute',
    'tensorflow/python/dlpack',
    'tensorflow/python/eager',
    'tensorflow/python/feature_column',
    'tensorflow/python/framework',
    'tensorflow/python/grappler',
    'tensorflow/python/keras',
    'tensorflow/python/kernel_tests',
    'tensorflow/python/layers',
    'tensorflow/python/lib',
    'tensorflow/python/module',
    'tensorflow/python/ops',
    'tensorflow/python/platform',
    'tensorflow/python/profiler',
    'tensorflow/python/saved_model',
    'tensorflow/python/summary',
    'tensorflow/python/tf_program',
    'tensorflow/python/tools',
    'tensorflow/python/tpu',
    'tensorflow/python/training',
    'tensorflow/python/util',
]

test_data_dirs = [
    'tensorflow/core/lib/bmp/testdata',
    'tensorflow/core/lib/gif/testdata',
    'tensorflow/core/lib/jpeg/testdata',
    'tensorflow/core/lib/lmdb/testdata',
    'tensorflow/core/lib/png/testdata',
    'tensorflow/core/lib/psnr/testdata',
    'tensorflow/core/lib/ssim/testdata',
]

# Below dirs will be copied inside org_tensorflow dir
org_tf_test_data_dirs = [
    'tensorflow/cc/saved_model/testdata',
    'tensorflow/python/feature_column/testdata',
    'tensorflow/python/keras/mixed_precision/testdata',
    'tensorflow/python/kernel_tests/testdata',
]

# Below are the specific files mentioned along with the path that needs to be copied
tf_specific_files = [
    'tensorflow/python/data/experimental/kernel_tests/data_service_test_base.py',
    'tensorflow/python/data/experimental/kernel_tests/reader_dataset_ops_test_base.py',
    'tensorflow/python/data/experimental/kernel_tests/serialization/dataset_serialization_test_base.py',
    'tensorflow/python/data/experimental/kernel_tests/sql_dataset_test_base.py',
    'tensorflow/python/data/experimental/kernel_tests/stats_dataset_test_base.py',
    'tensorflow/python/data/kernel_tests/test_base.py',
]

# Below dirs are the ones where we need to create an __init__.py
init_tf_dirs = [
    'tensorflow/python/autograph/converters/',
    'tensorflow/python/autograph/operators/',
    'tensorflow/python/data/',
    'tensorflow/python/data/experimental/',
    'tensorflow/python/data/experimental/service/',
    'tensorflow/python/data/kernel_tests/',
    'tensorflow/python/distribute/v1/',
    'tensorflow/python/keras/',
    'tensorflow/python/keras/integration_test/',
    'tensorflow/python/keras/layers/',
    'tensorflow/python/keras/layers/preprocessing/',
    'tensorflow/python/keras/saving/saved_model/',
    'tensorflow/python/kernel_tests/',
    'tensorflow/python/kernel_tests/array_ops/',
    'tensorflow/python/kernel_tests/v1_compat_tests/',
    'tensorflow/python/ops/parallel_for/',
]

if __name__ == "__main__":

    for dir in python_tests_dirs:
        copy_files_from_tf_repo_and_keep_intermediate_paths(dir, FileCopy.ALL_PYTHON_TEST_FILES)

    print(f"Summary of copy operation for python test files:")
    print(f"No.of files that are identical: {len(identical_files)}")
    print(f"No.of files that are new: {len(new_files)}")
    print(f"No.of files that are modified: {len(modified_files)}")
    print(f"No.of files that are obsoleted: {len(obsoleted_files)}")
    with open(dst_tf_repo + "/copy_test_from_tf_summary.txt", 'w') as f:
        f.write(f"Summary of copy operation for python test files:\n")
        f.write(f"No.of files that are identical: {len(identical_files)}\n")
        f.write(f"No.of files that are new: {len(new_files)}\n")
        f.write(f"No.of files that are modified: {len(modified_files)}\n")
        f.write(f"No.of files that are obsoleted: {len(obsoleted_files)}\n")
        f.write("Identical Files:\n")
        f.write("\n".join(identical_files))
        f.write("\nNew Files:\n")
        f.write("\n".join(new_files))
        f.write("\nModified Files:\n")
        f.write("\n".join(modified_files))
        f.write("\nObsoleted Files:\n")
        f.write("\n".join(obsoleted_files))

    for dir in test_data_dirs:
        copy_files_from_tf_repo_and_keep_intermediate_paths(dir, FileCopy.ALL_FILES)

    for dir in org_tf_test_data_dirs:
        copy_files_from_tf_repo_and_keep_intermediate_paths(dir, FileCopy.ALL_FILES, org_tensorflow = True)

    # Pass specific files as a complete list
    copy_files_from_tf_repo_and_keep_intermediate_paths(tf_specific_files, FileCopy.SPECIFIC_FILES)

    # Pass init_tf_dirs as a complete list
    create_init_files(init_tf_dirs)
