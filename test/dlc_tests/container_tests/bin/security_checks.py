import sys
import logging
import os
import time
import calendar

LOGGER = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def main():
    home_dir = os.path.expanduser('~')
    check_that_cache_dir_is_removed(home_dir)
    check_that_global_tmp_dir_is_empty()
    check_vim_info_does_not_exists(home_dir)
    check_bash_history(home_dir)
    check_if_any_files_in_subfolder_with_mask_was_last_modified_before_the_boottime(home_dir,
                                                                                    "history",
                                                                                    recursive=False)
    check_if_any_files_in_subfolder_with_mask_was_last_modified_before_the_boottime("/var/lib/cloud/instances/")
    return 0



def check_that_cache_dir_is_removed(home_dir):
    cache_dir_path = os.path.join(home_dir, ".cache")
    if os.path.exists(cache_dir_path):
        content_of_cache_dir = [f for f in os.listdir(cache_dir_path)]
        LOGGER.info("Contents of cache directory: %s", content_of_cache_dir)
        if len(content_of_cache_dir) > 1:
            raise ValueError("cache dir includes more than 1 file (not only motd)")
        if not content_of_cache_dir[0].startswith("pip"):
            raise ValueError("cache dir include file that it probably should not have: {}"
                             .format(content_of_cache_dir[0]))


def check_that_global_tmp_dir_is_empty():
    global_tmp_dir_path = "/tmp/"
    global_tmp_dir_content = [f for f in os.listdir(global_tmp_dir_path)]
    for f in global_tmp_dir_content:
        if not f.startswith(".") and "system" not in f.lower() and "dkms" not in f.lower() and "ccNPSUr9.s" not in f and "hsperfdata" not in f:
            raise ValueError("/tmp folder includes file that probably should not be there: {}".format(f))

def check_vim_info_does_not_exists(home_dir):
    viminfo_path = os.path.join(home_dir, ".viminfo")
    if os.path.exists(viminfo_path):
        raise ValueError("{} still exists".format(viminfo_path))


def check_bash_history(home_dir):
    bash_history_path = os.path.join(home_dir, ".bash_history")
    if os.path.exists(bash_history_path):
        with open(bash_history_path, "r") as bash_history_file:
            if bash_history_file.read():
                raise ValueError("{} contains history".format(bash_history_path))


def check_if_any_files_in_subfolder_with_mask_was_last_modified_before_the_boottime(folder, mask=None, recursive=True):
    uptime_seconds = 0
    if recursive:
        # Recursive travel and get all files under given folder
        all_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(folder) for f in filenames]
    else:
        all_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    # Get the bootime
    with open('/proc/uptime', 'r') as uptime_process:
        uptime_seconds = int(round(float(uptime_process.readline().split()[0])))
    current_time_seconds = int(calendar.timegm(time.gmtime()))
    boot_time_seconds = current_time_seconds - uptime_seconds

    # Filter the files need to be checked
    if mask is not None:
        all_files = [f for f in all_files if mask in f]

    for f in all_files:
        last_modified_time_seconds = int(round(os.path.getmtime(f)))
        if last_modified_time_seconds < boot_time_seconds:
            raise ValueError("Looks like {} was modified before the current boot".format(f))

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
