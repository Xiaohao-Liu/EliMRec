import time
import os

class Logger():
    # static varies
    logger = None
    def __init__(self, name=None, show_in_console=False,  is_creat_log_file = False, path=""):
        if name is None:
            self.name = str(int(time.time()))
        else:
            self.name = name +'_'+str(int(time.time()))
        self.show_in_console = show_in_console
        self.log_file_name = path +"/" + self.name + '.log'
        self.in_file = is_creat_log_file
        if self.in_file is True:
            self.create_log_file()
            print("log to file: ",self.log_file_name)
        
    def create_log_file(self):
        with open(self.log_file_name,"w",encoding="utf-8") as f:
            f.write("============Start Logging============\n")
            f.write("[Created At]:"+str(time.asctime(time.localtime(time.time())))+"\n")
            f.write("=====================================\n")

    def log(self,*msg):
        msg = "\t".join([str(m) for m in msg])
        if self.show_in_console:
            print(msg)
        if self.in_file is True:
            with open(self.log_file_name, "a") as f:
                f.write(msg+"\n")

    @staticmethod
    def info(*msg):
        if Logger.logger is None:
            print(*msg)
            AssertionError('The logger is not initialized')
        else:
            Logger.logger.log(*msg)
"""
Usage:
logger = Logger(train_type="NN",show_in_console=True)
logger.log('log something')
"""