import os

ARCHIVE_FOLDER_PATH = "/opt/aws/dlc/patch-details-archive"
LATEST_PATCH_DETAILS_FOLDER_PATH = "/opt/aws/dlc/patch-details"
CONTENTS_WITHIN_ARCHIVE_FOLDER = os.listdir(ARCHIVE_FOLDER_PATH)
OVERALL_HISTORY_DUMP_LOCATION = os.path.join(
    os.sep, LATEST_PATCH_DETAILS_FOLDER_PATH, "overall_history.txt"
)


def generate_overall_history(folder_path, overall_history, release_count=0):
    """
    In this method we look into the install scripts within the patch-details folder and extract the contents of the install
    scripts that are appended to the overall_history.txt

    :param folder_path: str, Path of the patch-details folder
    :param overall_history: list, contains contents that will finally be dumped into the overall_history.txt file
    :param release_count: int, stores the release number after the first manual release
    """
    if ARCHIVE_FOLDER_PATH in folder_path:
        f = open(os.path.join(os.sep, folder_path, "image_sha.txt"), "r")
        overall_history.append(f"#### Patch contents for patch-{release_count}: {f.read().strip()} ####")
    elif LATEST_PATCH_DETAILS_FOLDER_PATH in folder_path:
        overall_history.append("#### Current Patch contents ####")
    f = open(os.path.join(os.sep, folder_path, "install_script_language.sh"), "r")
    overall_history.append(f"{f.read().strip()}")
    f = open(os.path.join(os.sep, folder_path, "install_script_os.sh"), "r")
    overall_history.append(f"{f.read().strip()}")


def main():
    """
    This script is run during the build phase of the AutoPatch DLC and is used by the AutoPatch dockerfile.
    The script looks into the entire patch history starting from the first release and put the data in the
    overall_history.txt file.
    """
    overall_history = []
    f = open(os.path.join(os.sep, ARCHIVE_FOLDER_PATH, "first_image_sha.txt"), "r")
    first_image_sha = f.read().strip()
    overall_history.append(f"#### First image: {first_image_sha} ####")
    patch_details_folder_names = [
        artifact_name
        for artifact_name in CONTENTS_WITHIN_ARCHIVE_FOLDER
        if os.path.isdir(os.path.join(os.sep, ARCHIVE_FOLDER_PATH, artifact_name))
        and artifact_name.startswith("patch-details-")
    ]
    patch_details_folder_names = sorted(
        patch_details_folder_names, key=lambda x: int(x.split("-")[-1])
    )

    for release_count, patch_details_folder_name in enumerate(patch_details_folder_names):
        patch_details_folder_path = os.path.join(
            os.sep, ARCHIVE_FOLDER_PATH, patch_details_folder_name
        )
        generate_overall_history(
            patch_details_folder_path, overall_history, release_count
        )
    generate_overall_history(LATEST_PATCH_DETAILS_FOLDER_PATH, overall_history)

    with open(OVERALL_HISTORY_DUMP_LOCATION, "w") as f:
        for line in overall_history:
            f.write(f"{line}\n")


if __name__ == "__main__":
    main()
