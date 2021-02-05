# Standard Library
import inspect
import os


def log_result(label, res):
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    filename = os.path.basename(module.__file__)
    filename = filename.split(".")[0]

    print("[SMP_METRIC]__" + filename + "__" + str(label) + "__" + str(res) + "\n")
