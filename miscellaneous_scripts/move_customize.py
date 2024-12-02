import sys
import shutil
import sysconfig
import os


def move_customize(source):
    destination = os.path.join(sysconfig.get_path("stdlib"), "sitecustomize.py")
    shutil.move(source, destination)


if __name__ == "__main__":
    move_customize(sys.argv[1])
