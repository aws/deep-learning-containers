import subprocess
import argparse
import json

from enum import Enum


class ModeType(str, Enum):
    GENERATE = "generate"
    MODIFY = "modify"


RETURN_CODE_OK = [0]


def list_of_strings(arg):
    return arg.split(",") if arg else []


def get_package_list_using_command(run_command="apt list --installed"):
    """
    Uses the input command to retrieve the list of installed/upgradable apt packages.
    :param run_command: str, the input apt command
    :return: list, the list of packages
    """
    run_output = subprocess.run(run_command, shell=True, capture_output=True, text=True, check=True)
    result = run_output.stdout.strip().split("\n")
    return [output_line.split("/")[0] for output_line in result if "/" in output_line]


def get_installed_version_for_packages(package_list=[]):
    """
    Finds the currently installed version of the packages.

    :param package_list: list[str], List of packages
    :return: dict[str, str], Dict with (keys=package names) and (values=installed package versions)
    """
    run_command = "apt list --installed"
    run_output = subprocess.run(run_command, shell=True, capture_output=True, text=True, check=True)
    result = run_output.stdout.strip().split("\n")
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


def is_package_or_its_source_is_impacted_and_the_package_is_upgradable(
    package, source_package, impacted_packages, upgradable_packages
):
    """
    Checks if the package or it source is impacted and if the package is upgradable or not.

    :param package: str, package name
    :param source_package: str, source_package name
    :param impacted_packages: list, List of all the impacted apt packages (or source apt packages)
    :param upgradable_packages: list, List of all the upgradable apt packages
    :return: boolean
    """
    return (
        any([pckg in impacted_packages for pckg in [package, source_package]])
        and package in upgradable_packages
    )


def update_patch_package_list_and_upgradable_packages_data(
    installed_packages,
    impacted_packages,
    upgradable_packages,
    patch_package_list,
    upgradable_packages_data_for_impacted_packages,
):
    """
    This method iterates through the apt packages to figure out packages that are impacted and upgradable and updates
    `patch_package_list` and `upgradable_packages_data_for_impacted_packages` with releavant data.

    The ECR Enhanced Scan gives the info about the source package that is impacted, but does not give the exact package/binary that is impacted.
    Hence, we need to find out the source of each package and see if the source or the package is impacted.
    If either is impacted and the package is upgradable, it is stored in the patch_package_list.

    For eg., in one of the ECR scans, the impacted packages was cups2. However, cups2 (https://packages.ubuntu.com/source/focal/cups)
    was not present in the DLC. Instead, the package present in DLC was one of its binaries called libcups2 (as can be seen in the link shared).
    Enhanced Scan does not point us to the exact impacted binary. Thus, for any package, we need to make sure if that package is directly reported
    by enhanced scan, or, if one of its soruce packages is being reported as impacted (cups2 was the source package for libcups2). In case a package
    or its source package is impacted, we make the decision to upgrade it.

    The patch_package_list consists of all the packages that need to be upgraded and their current version. It also updates a dictionary
    called upgradable_packages_data_for_impacted_packages. The purpose of upgradable_packages_data_for_impacted_packages dict is to
    store the patch data regarding each package. In other words, a key in this dict is the package name and the value for that key is
    a list. The list contains of all the packages that were upgraded because of this package being a source package. In case the package
    is not a source pacakge for any package, the list will only contain the name of the package.

    For eg, in case of libcups2 and cups2, in case libcups2 is present in the dlc, the dict would look like:
        {
            "cups2":["libcups2"],
            "libcups2":["libcups2"]
        }
    It can essentially interpreted as - "To patch `key` (cups2) packages in the `values` (["libcups2") had to be upgraded.

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
        # dpkg -s <package> gives the output in the following format:
        #     Package: xxd
        #     Status: install ok installed
        #     Source: vim
        # We extract the source package from the command output
        source_extraction_cmd_output = subprocess.run(
            f"dpkg -s {package} | grep ^Source", shell=True, capture_output=True, text=True
        )
        if source_extraction_cmd_output.returncode in RETURN_CODE_OK:
            source_package = source_extraction_cmd_output.stdout.strip().split()[1]

        if is_package_or_its_source_is_impacted_and_the_package_is_upgradable(
            package=package,
            source_package=source_package,
            impacted_packages=impacted_packages,
            upgradable_packages=upgradable_packages,
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


def execute_generative_mode_type(args):
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

    (
        patch_package_list,
        upgradable_packages_data_for_impacted_packages,
    ) = update_patch_package_list_and_upgradable_packages_data(
        installed_packages,
        impacted_packages,
        upgradable_packages,
        patch_package_list,
        upgradable_packages_data_for_impacted_packages,
    )

    patch_package_dict = get_installed_version_for_packages(patch_package_list)
    for _, version_dict in patch_package_dict.items():
        version_dict["previous_version"] = version_dict.pop("installed_version")

    apt_patch_details = {
        "patch_package_dict": patch_package_dict,
        "upgradable_packages_data_for_impacted_packages": upgradable_packages_data_for_impacted_packages,
    }
    with open(args.save_result_path, "w") as outfile:
        json.dump(apt_patch_details, outfile, indent=4)


def execute_modify_mode_type(args):
    """
    This method is excuted when the Modify mode is enabled. In this case, it reads the already generated `apt_patch_details` report
    that was generated by execute_generative_mode_type method during generative mode execution. Thereafter, it goes through all the
    packages in the patch_package_dict and adds the current version of the package into the list.

    Generally, the generative mode is run before the image has been patched (in the Autopatch-prep stage), to store the package versions before patching.
    Modify mode is run during image build when the packages have been patched. It aims to find the latest version of the packages after patching and preserve the delta.

    :param: args, ArgParse Object
    """
    with open(args.save_result_path, "r") as readfile:
        json_data = json.load(readfile)
    patch_package_list = list(json_data["patch_package_dict"].keys())
    current_patch_package_dict = get_installed_version_for_packages(patch_package_list)
    for k, _ in json_data["patch_package_dict"].items():
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
    parser.add_argument(
        "--mode_type", type=str, choices=[mode_type.value for mode_type in ModeType], required=True
    )
    parser.add_argument("--impacted-packages", type=list_of_strings, default="")
    parser.add_argument("--save-result-path", type=str, required=True)

    args = parser.parse_args()
    if args.mode_type == ModeType.GENERATE:
        execute_generative_mode_type(args)
    else:
        execute_modify_mode_type(args)


if __name__ == "__main__":
    main()
