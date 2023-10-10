import os

archive_folder_path = "/opt/aws/dlc/patch-details-archive"
latest_patch_details_folder_path = "/opt/aws/dlc/patch-details"
contents_within_archive_folder = os.listdir(archive_folder_path)
overall_history_dump_location = os.path.join(
    os.sep, latest_patch_details_folder_path, "overall_history.txt"
)


def process_patch_folders_and_add_contents(folder_path, overall_history, release_count):
    if archive_folder_path in folder_path:
        f = open(os.path.join(os.sep, folder_path, "image_sha.txt"), "r")
        overall_history.append(f"#### Release {release_count}: {f.read().strip()} ####")
    elif latest_patch_details_folder_path in folder_path:
        overall_history.append("#### Latest Release ####")
    f = open(os.path.join(os.sep, folder_path, "install_script.sh"), "r")
    overall_history.append(f"{f.read().strip()}")
    f = open(os.path.join(os.sep, folder_path, "install_script_second.sh"), "r")
    overall_history.append(f"{f.read().strip()}")


def main():
    overall_history = []
    f = open(os.path.join(os.sep, archive_folder_path, "first_image_sha.txt"), "r")
    first_image_sha = f.read().strip()
    overall_history.append(f"#### First image: {first_image_sha} ####")
    patch_details_folder_names = [
        artifact_name
        for artifact_name in contents_within_archive_folder
        if os.path.isdir(os.path.join(os.sep, archive_folder_path, artifact_name))
        and artifact_name.startswith("patch-details-")
    ]
    patch_details_folder_names = sorted(
        patch_details_folder_names, key=lambda x: int(x.split("-")[-1])
    )
    release_count = 0
    for patch_details_folder_name in patch_details_folder_names:
        patch_details_folder_path = os.path.join(
            os.sep, archive_folder_path, patch_details_folder_name
        )
        release_count += 1
        process_patch_folders_and_add_contents(
            patch_details_folder_path, overall_history, release_count
        )
    process_patch_folders_and_add_contents(
        latest_patch_details_folder_path, overall_history, release_count
    )

    with open(overall_history_dump_location, "w") as f:
        for line in overall_history:
            f.write(f"{line}\n")


if __name__ == "__main__":
    main()
