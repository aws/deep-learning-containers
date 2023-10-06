from invoke import run
import argparse
import json


def list_of_strings(arg):
    return arg.split(",")


def get_package_list_using_command(run_command="apt list --installed"):
    otpt = run(run_command, hide=True)
    result = otpt.stdout.strip().split("\n")
    return [output_line.split("/")[0] for output_line in result if "/" in output_line]


def process_packages(
    installed_packages,
    impacted_packages,
    upgradable_packages,
    patch_package_list,
    upgradable_packages_data_for_impacted_packages,
):
    for package in installed_packages:
        source_package = ""
        source_extraction_cmd_output = run(
            f"dpkg -s {package} | grep ^Source", warn=True, hide=True
        )
        if source_extraction_cmd_output.ok:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impacted-packages", type=list_of_strings)
    parser.add_argument("--save-result-path", type=str)

    args = parser.parse_args()
    impacted_packages = args.impacted_packages
    installed_packages = get_package_list_using_command(run_command="apt list --installed")
    upgradable_packages = get_package_list_using_command(run_command="apt list --upgradable")

    upgradable_packages_data_for_impacted_packages = {}
    patch_package_list = []

    process_packages(
        installed_packages,
        impacted_packages,
        upgradable_packages,
        patch_package_list,
        upgradable_packages_data_for_impacted_packages,
    )

    apt_patch_details = {
        "patch_package_list": patch_package_list,
        "upgradable_packages_data_for_impacted_packages": upgradable_packages_data_for_impacted_packages
    }
    with open(args.save_result_path, "w") as outfile:
        json.dump(apt_patch_details, outfile, indent=4)


if __name__ == "__main__":
    main()
