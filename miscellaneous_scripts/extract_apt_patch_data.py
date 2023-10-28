import subprocess
import argparse
import json

from enum import Enum

class ModeType(str, Enum):
    GENERATE = 'generate'
    MODIFY = 'modify'

RETURN_CODE_OK = [0]

def list_of_strings(arg):
    return arg.split(",") if arg else []


def get_package_list_using_command(run_command="apt list --installed"):
    """
    Uses the input command to retrieve the list of installed/upgradable apt packages.
    :param run_command: str, the input apt command
    :return: list, the list of packages
    """
    otpt = subprocess.run(run_command, shell=True, capture_output=True, text=True, check=True)
    result = otpt.stdout.strip().split("\n")
    return [output_line.split("/")[0] for output_line in result if "/" in output_line]


def get_installed_version_for_packages(package_list = []):
    """
    Finds the currently installed version of the packages.

    :param package_list: list[str], List of packages
    :return: dict[str, str], Dict with (keys=package names) and (values=installed package versions)
    """
    run_command="apt list --installed"
    otpt = subprocess.run(run_command, shell=True, capture_output=True, text=True, check=True)
    result = otpt.stdout.strip().split("\n")
    package_dict = {}
    for output_line in result:
        if "/" not in output_line:
            continue
        package_name = output_line.split("/")[0]
        if package_name not in package_list:
            continue
        if package_name in package_dict:
            raise ValueError(f"Package {package_name} already exists in {package_dict}")
        package_dict[package_name] = {}
        installed_version = output_line.split()[1]
        package_dict[package_name]["installed_version"] = installed_version
    return package_dict


def process_packages(
    installed_packages,
    impacted_packages,
    upgradable_packages,
    patch_package_list,
    upgradable_packages_data_for_impacted_packages,
):
    """
    This method handles the processing of the apt packages.  The ECR Enhanced Scan gives the info about the source
    package that is impacted, but does not give the exact package/binary that is impacted. Hence, we need to find out
    the source of each package and see if the source or the package is impacted. If either is impacted and the package
    is upgradable, it is stored in the patch_package_list. The patch_package_list consists of all the packages that need
    to be upgraded and their current version. It also updates a dictionary called upgradable_packages_data_for_impacted_packages.
    This dict maintains the source package names as the keys and all their packages/binaries as values.

    :param installed_packages: list, List of all the installed apt packages
    :param impacted_packages: list, List of all the impacted apt packages (or source apt packages)
    :param upgradable_packages: list, List of all the upgradable apt packages
    :param patch_package_list: list, List of all the apt packages that need to be upgraded for patching. One of the core functionalities of this
                                     method is to add data to this list.
    :param upgradable_packages_data_for_impacted_packages: dict[List], Dict of all the impacted source apt packages as keys and a list of their upgradable binaries as value.
                                                                       One of the core functionalities of this method is to add data to this dict.
    :return patch_package_list: list, the same input parameter that is modified by this function
    :return upgradable_packages_data_for_impacted_packages: dict[list], the same input parameter that is modified by this function
    """
    for package in installed_packages:
        source_package = ""
        source_extraction_cmd_output = subprocess.run(
            f"dpkg -s {package} | grep ^Source", shell=True, capture_output=True, text=True
        )
        if source_extraction_cmd_output.returncode in RETURN_CODE_OK:
            source_package = source_extraction_cmd_output.stdout.strip().split()[1]

        if (
            any([pckg in impacted_packages for pckg in [package, source_package]])
            and package in upgradable_packages
        ):
            patch_package_list.append(package)
            ## Add package to the dict that maintains the upgradable package data
            if package in impacted_packages:
                if package not in upgradable_packages_data_for_impacted_packages:
                    upgradable_packages_data_for_impacted_packages[package] = []
                upgradable_packages_data_for_impacted_packages[package].append(package)
            ## Add source_package to the dict that maintains the upgradable package data
            if source_package in impacted_packages:
                if source_package not in upgradable_packages_data_for_impacted_packages:
                    upgradable_packages_data_for_impacted_packages[source_package] = []
                upgradable_packages_data_for_impacted_packages[source_package].append(package)
    return patch_package_list, upgradable_packages_data_for_impacted_packages


def process_generative_mode_type(args):
    """
    This method drives the functionality for the generative mode execution. It finds all the installed and the upgradable packages.
    Thereafter, it sends the relevant data to process_packages method to find out all the impacted packages that can be upgraded.
    In the ends, it dumps the data at save_result_path location.
    """
    impacted_packages = args.impacted_packages
    assert impacted_packages, "Impacted packages need to be passed for generative mode."
    installed_packages = get_package_list_using_command(run_command="apt list --installed")
    upgradable_packages = get_package_list_using_command(run_command="apt list --upgradable")

    upgradable_packages_data_for_impacted_packages = {}
    patch_package_list = []
    patch_package_dict = {}

    process_packages(
        installed_packages,
        impacted_packages,
        upgradable_packages,
        patch_package_list,
        upgradable_packages_data_for_impacted_packages,
    )

    patch_package_dict = get_installed_version_for_packages(patch_package_list)
    for k, _ in patch_package_dict.items():
        patch_package_dict[k]["previous_version"]=patch_package_dict[k].pop("installed_version")

    apt_patch_details = {
        "patch_package_dict": patch_package_dict,
        "upgradable_packages_data_for_impacted_packages": upgradable_packages_data_for_impacted_packages
    }
    with open(args.save_result_path, "w") as outfile:
        json.dump(apt_patch_details, outfile, indent=4)


def process_modify_mode_type(args):
    with open(args.save_result_path, "r") as readfile:
        json_data = json.load(readfile)
    patch_package_list = list(json_data["patch_package_dict"].keys())
    current_patch_package_dict = get_installed_version_for_packages(patch_package_list)
    for k,_ in json_data["patch_package_dict"].items():
        json_data["patch_package_dict"][k].update(current_patch_package_dict[k])
    with open(args.save_result_path, "w") as outfile:
        json.dump(json_data, outfile, indent=4)


def main():
    """
    This script takes in the list of impacted packages and finds all the apt packages that need to be and can be upgraded for patching.
    The script can work in 2 modes - genrate and modify. During the generate mode, the patchable packages are found out and are stored
    with their current version at the save-result-path location. During the modify phase, the patchable packages are read from the
    save-result-path location and their latest version is stored at the save-result-path location itself. The GENERATE mode is run on
    the released DLCs to find out the patchable packages and their version in the released DLCs. Once the released DLCs are patched,
    the same script is run in the MODIFY mode to add the latest version data of the patched packages.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode_type", type=str, choices=[ModeType.GENERATE, ModeType.MODIFY], required=True)
    parser.add_argument("--impacted-packages", type=list_of_strings, default="")
    parser.add_argument("--save-result-path", type=str, required=True)

    args = parser.parse_args()
    if args.mode_type == ModeType.GENERATE:
        process_generative_mode_type(args)
    else:
        process_modify_mode_type(args)


if __name__ == "__main__":
    main()
