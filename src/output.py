'''
This file defines the output formatter class
that has all functions for printing output to stdout
'''
import time
import sys
import shutil
from collections import defaultdict

import pyfiglet
import reprint


class OutputFormatter:
    '''
    This class is responsible for having a unified interface to print to stdout
    '''
    def __init__(self, padding=0):
        '''
        Constructor that defines the attributes of the formatter class
        '''
        self.width = shutil.get_terminal_size().columns
        self.padding_length = padding
        self.max_line_length = self.width - ((self.padding_length + 1) * 2)

        self.padding = " " * self.padding_length
        self.left_padding = "=" + self.padding
        self.right_padding = self.padding + "="

    def separator(self):
        '''
        Print separator between blocks of output
        '''
        # TODO: Make decorator
        # TODO: Make "=" dynamic
        print("=" * self.width)

    def title(self, title):
        '''
        Print title for the block
        '''
        title = title.center(self.width, "=")
        print(title)

    def banner(self, title):
        '''
        Print banner (title for the program
        '''
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

        status_code = {0: "S", 1: "F"}

        isatty = sys.stdout.isatty()

        with reprint.output(
            output_type="list", initial_len=len(futures.items()), interval=0
        ) as output:
            num_iterations = 0
            if not isatty:
                output = [""] * len(futures.items())
                running_iterator = 0
                for image, _ in futures.items():
                    output[running_iterator] = f"{image}.........R"
                    running_iterator += 1
            self.print_lines(output)
            while True:
                i = 0
                num_iterations += 1
                for image, thread in futures.items():
                    output[i] = image
                    if thread.done():
                        output[i] += (
                            "." * 10 + status_code[futures[image].result()]
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
        '''
        Print multiple lines
        '''
        self.print("\n".join(lines))
