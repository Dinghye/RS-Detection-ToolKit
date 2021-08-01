import numpy as np
from detectron2.structures import BoxMode

"""
data type : TYPE,X1,Y1,X2,Y2,X3,Y3,X4,Y4
"""


def read_labels(path):
    """
    purpose: 读取单个存放标注的txt数据
    :param path: 文件所在文件夹
    :param name: 文件名称（不带后缀名）
    :return: 带标注的数组
    """

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
def txt_to_register(info_group):
    dataset = []
    return dataset




def get_dicts(self, ids):
    """
    purpose: 用于定义自己的数据集格式，返回指定对象的数据
    :param ids: 需要返回数据的名称（id）号
    :return: 指定格式的数据字典
    """
    dataset_dicts = []
    data = {}
    count = 0
    for i in ids:
        count += 1
        # 构建图！
        img = self.read_labels(self.DATA_PATH, i)
        data["ids"] = count
        data["image_id"] = int(i)

        data["height"] = 1024
        data["width"] = 1024
        data["file_name"] = self.DATA_PATH + "images/" + str(i) + ".tif"
        # 对于每一个图里面的annotation来说
        annotations = []
        for j in img:
            ann = {}
            ann["bbox_mode"] = BoxMode.XYWHA_ABS
            ann["category_id"] = int(j[0])  # 根据给出来的数据格式，0
            ann["bbox"] = self._get_abbox(j)
            annotations.append(ann)
        data["annotations"] = annotations
        dataset_dicts.append(data)
        data = {}
    return dataset_dicts
