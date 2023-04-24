"""
Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"). You
may not use this file except in compliance with the License. A copy of
the License is located at

    http://aws.amazon.com/apache2.0/

or in the "license" file accompanying this file. This file is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
"""
import time
import sys
import shutil
import logging
from collections import defaultdict

import pyfiglet
import reprint
import constants


class OutputFormatter:
    """
    This class is responsible for having a unified interface to print to stdout
    """

    def __init__(self, padding=0):
        """
        Constructor that defines the attributes of the formatter class
        """
        self.width = shutil.get_terminal_size().columns
        self.padding_length = padding
        self.max_line_length = self.width - ((self.padding_length + 1) * 2)

        self.padding = " " * self.padding_length
        self.left_padding = "=" + self.padding
        self.right_padding = self.padding + "="

        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    def log(self, level, message):
        if level == constants.INFO:
            logging.info(message)
        if level == constants.ERROR:
            logging.error(message)
        if level == constants.DEBUG:
            logging.debug(message)

    def separator(self):
        """
        Print separator between blocks of output
        """
        # TODO: Make decorator
        # TODO: Make "=" dynamic
        print("=" * self.width)

    def title(self, title):
        """
        Print title for the block
        """
        title = title.center(self.width, "=")
        print(title)

    def banner(self, title):
        """
        Print banner (title for the program
        """
        title = pyfiglet.figlet_format(title)
        lines = title.split("\n")
        self.separator()
        for line in lines:
            line = line.center(self.max_line_length)
            print(f"{self.left_padding}{line}{self.right_padding}")
        self.separator()

    def progress(self, futures):
        """
        Print a progressbar
        Note: futures is a dictionary. Keys = Name of the thread,
        Value = concurrent.futures object. The function being executed
        MUST return a dictionary with 'status' key that defines the status code.
        """

        done = defaultdict(bool)

        isatty = sys.stdout.isatty()

        with reprint.output(
            output_type="list", initial_len=len(futures.items()), interval=0
        ) as output:
            num_iterations = 0
            self.print_lines(output)
            while True:
                i = 0
                num_iterations += 1
                for image, thread in futures.items():
                    output[i] = image
                    if thread.done():
                        output[i] += (
                            "." * 10 + constants.STATUS_MESSAGE[futures[image].result()]
                        )
                        done[image] = True
                    else:
                        output[i] += "." * (num_iterations % 10)
                        done[image] = False
                    i += 1

                if all(done.values()):
                    break
                time.sleep(1)

        self.print_lines(output)

    def table(self, rows):
        """
        Print a table from dictionary
        rows = iter of tuples
        """
        for (key, value) in rows:
            # TODO: left and right align key and value
            line = f"{key}:{value}".ljust(self.max_line_length)
            print(f"{line}")

    def print(self, line):
        """
        To keep all output to stdout consistent. Gives room to format each line in the future.
        """
        print(line)

    def print_lines(self, lines):
        """
        Print multiple lines
        """
        self.print("\n".join(lines))
