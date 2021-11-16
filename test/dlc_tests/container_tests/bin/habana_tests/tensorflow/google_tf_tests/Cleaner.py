import glob
import os

from .consts import (
    POST_PART_FILES_TO_REMOVE,
    ALL_GRAPHS_DUMPS,
)

class Cleaner(object):
    def __init__(self, logger):
        self.logger = logger

    @staticmethod
    def remove_post_part_files():
        for pattern_to_remove in POST_PART_FILES_TO_REMOVE:
            file_list = glob.glob(pattern_to_remove)
            for file_path in file_list:
                os.remove(file_path)

    @staticmethod
    def remove_graph_dumps():
        graph_dumps_path = os.path.join(os.getcwd(), ALL_GRAPHS_DUMPS)
        file_list = glob.glob(graph_dumps_path)
        for file_path in file_list:
            os.remove(file_path)

    @staticmethod
    def remove_bin_files():
        for file in os.listdir("."):
            if file.endswith(".bin"):
                os.remove(file)
                base = os.path.splitext(file)[0]
                if os.path.isfile(base):
                    os.remove(base)

    def clean_before(self):
        Cleaner.remove_post_part_files()
        Cleaner.remove_graph_dumps()

    def clean_after(self):
        Cleaner.remove_post_part_files()
        Cleaner.remove_graph_dumps()
