import os
import glob
class Grabber:
    """Returns a list of .edf files in ../data/ directory"""
    def __init__(self, directory=None) -> list:
        self.directory = directory

    def grab_edfs(self) -> list:
        pwd = os.getcwd()
        dir = os.path.join(pwd, self.directory, "*.edf")
        return glob.glob(dir)
    
    def grab_subject_name(self) -> str:
        return os.getcwd().split(folder_splitter)[-2].split("VEEG_Event_Processor-")[-1]