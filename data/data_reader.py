"""
This is a interface of data
mainly to help dataloader to register data
"""


class Dataset():
    def __init__(self):
        self.data_type = ''  # useless yet ... rotated / positive
        self.data_set = []  # here is a container of formed data
        self.data_path = ''  # data root
