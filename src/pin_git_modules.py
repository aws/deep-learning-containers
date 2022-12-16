import subprocess as sp
import argparse
from os.path import join, dirname

def git_checkout(paths, date):
    checkout_cmds = []
    for p in paths:
        checkout_cmds.append(f"cd {p} && git checkout -f $(git rev-list -1 --before=\"{date}\" main)")

    if len(checkout_cmds) == 1:
        cmd = checkout_cmds[0]
    else:
        cmd = (" && ").join(checkout_cmds)

    # git checkout all the modules in one process
    sp.run(cmd, shell=True, check=True)

def find_submodules(path):
    res = sp.run(f"find {path} -name .gitmodules", shell=True, check=True, capture_output=True)
    if res.stdout == '':
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
    parser.add_argument("--date", required=True, help="date string used to find the commit (format: YYYY-MM-DD)")
    args = parser.parse_args()
    
    git_checkout([args.src], args.date)
    files = find_submodules(args.src)
    git_checkout(files, args.date)
    
    
    