import os
import time


class Logger:
    def __init__(self, log_file):
        self.log_file = log_file

        # check if log_file exists
        if os.path.exists(log_file):
            raise FileExistsError("The log file already exists.")

    def write_log(self, msg):
        with open(self.log_file, 'a') as ww:
            prompt = '[' + time.ctime() + '] : '
            ww.write(prompt + msg + "\n")