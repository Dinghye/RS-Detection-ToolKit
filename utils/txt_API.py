import numpy as np

"""
data type : X1,Y1,X2,Y2,X3,Y3,X4,Y4,TYPE
"""


def read_labels(path):
    f = open(path)
    line = f.readline()
    data_list = []
    while line:
        num = list(map(float, line.split()))
        data_list.append(num)
        line = f.readline()
    f.close()
    data_array = np.array(data_list)
    return data_array

# NOT FINISHED YET
