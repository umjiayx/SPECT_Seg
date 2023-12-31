"""
author: Xiaojian Xu, 2018
Transcript - direct print output to a file, in addition to terminal.
"""

import sys
class Transcript(object):

    def __init__(self, filename, mode="a"):
        self.terminal = sys.stdout
        self.logfile = open(filename, mode)

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def start(filename, mode='a'):
    """Start transcript, appending print output to given filename"""
    sys.stdout = Transcript(filename, mode=mode)

def stop():
    """Stop transcript and return print functionality to normal"""
    sys.stdout.logfile.close()
    sys.stdout = sys.stdout.terminal

