import argparse
import glob
import os
import subprocess as sp
import time
from os.path import dirname, join

# set the timezone once in-process
os.environ["TZ"] = "America/New_York"
time.tzset()


def run(cmd, cwd=None, **kwargs):
    """simple wrapper around subprocess.run with common defaults"""
    return sp.run(cmd, cwd=cwd, text=True, capture_output=True, check=True, **kwargs)


def git_checkout(paths, date):
    """checkout each repo in paths to the last commit before date

    Keyword arguments:
    paths -- a list of absolute paths to the git repos
    date -- a date (%Y-%M-%D) of which the immediately prior commit will be checked out
    """
    for p in paths:
        rev = run(["git", "rev-list", "-1", f"--before={date}", "HEAD"], cwd=p).stdout.strip()
        run(["git", "checkout", "-f", rev], cwd=p)


def init_submodules(path):
    """initialize a repo's submodules recursively"""
    run(["git", "submodule", "update", "--init", "--recursive"], cwd=path)


def find_submodules(path):
    """explore and return a list of submodule paths from a certain git repo at `path`

    Keyword arguments:
    path -- the aboslute path of git repo used as a root of finding its submodules
    """
    module_files = glob.glob(f"{path}/**/.gitmodules", recursive=True)
    if not module_files:
        return False

    subs = []
    for m in module_files:
        root = dirname(m)
        for line in open(m):
            if line.startswith("path = "):
                subs.append(join(root, line.split("=", 1)[1].strip()))
    return subs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument(
        "--date", required=True, help="date string used to find the commit (format: YYYY-MM-DD)"
    )
    args = parser.parse_args()

    git_checkout([args.src], args.date)
    init_submodules(args.src)
    files = find_submodules(args.src)
    git_checkout(files, args.date)
