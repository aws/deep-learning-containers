import subprocess as sp
import argparse, os
from time import tzset
from os.path import join, dirname


def git_checkout(paths, date):
    """check out the git modules located in paths one by one in a subprocess with a commmit before a certain date

    Keyword arguments:
    paths -- a list of absolute paths to the git repos
    date -- a date (%Y-%M-%D) of which the immediately prior commit will be checked out
    """
    tz_cmds = 'echo "America/New_York" > /etc/timezone && ln -sf /usr/share/zoneinfo/America/New_York /etc/localtime && '
    checkout_cmds = []
    for p in paths:
        checkout_cmds.append(
            f'cd {p} && git checkout -f $(git rev-list -1 --before="{date}" HEAD)'
        )

    if len(checkout_cmds) == 1:
        cmd = checkout_cmds[0]
    else:
        cmd = " && ".join(checkout_cmds)
    cmd = tz_cmds + cmd
    # git checkout all the modules in one process
    sp.run(cmd, shell=True, check=True)


def init_submodules(path):
    """initialize a repo's submodules recursively"""
    sp.run(
        f"cd {path} && git submodule update --init --recursive", shell=True, check=True
    )


def find_submodules(path):
    """explore and return a list of submodule paths from a certain git repo at `path`

    Keyword arguments:
    path -- the aboslute path of git repo used as a root of finding its submodules
    """
    res = sp.run(
        f"find {path} -name .gitmodules", shell=True, check=True, capture_output=True
    )
    if res.stdout == "":
        return False

    submodules = []
    module_files = res.stdout.decode("utf-8").split("\n")
    for m in module_files:
        if m:
            with open(m) as f:
                submodule_root = dirname(m)
                module_paths = f.read().split("path = ")[1:]
                for p in module_paths:
                    submodule_path = join(submodule_root, p.split("\n")[0])
                    submodules.append(submodule_path)
    return submodules


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument(
        "--date",
        required=True,
        help="date string used to find the commit (format: YYYY-MM-DD)",
    )
    args = parser.parse_args()

    git_checkout([args.src], args.date)
    init_submodules(args.src)
    files = find_submodules(args.src)
    git_checkout(files, args.date)
