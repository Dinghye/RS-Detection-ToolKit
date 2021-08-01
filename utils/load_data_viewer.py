import random

from detectron2.data import DatasetCatalog, MetadataCatalog
import cv2
from sklearn.model_selection import train_test_split
# import data.txt_data_loader as ld
from data.json_API import JSON
from data.rotated_data_loader import info_Register
from data.rotated_visualization import myVisualization

"""
Use the data display that comes with detectron2 to view annotation data
Mainly used to check if the data is correctly registered。

(txt AIP support)
"""

#
# dataset_dicts = ld.Register().get_dicts(range(1, 2009))
# # 特别的，这里之所以直接用range来生成，是因为前面输入的id就是按顺序拍下来的num
# total_csv_annotations = range(1, len(dataset_dicts) + 1)
#
# total_keys = list(total_csv_annotations)
# train_keys, val_keys = train_test_split(total_keys, test_size=0.2)
# print("train_n:", len(train_keys), 'val_n:', len(val_keys))
#
# # 注册数据集
# DatasetCatalog.register("ship_train", lambda: ld.Register().get_dicts(train_keys))
# MetadataCatalog.get("ship_train").set(thing_classes=['__background__', '1', '2', '3', '4', '5'])
#
# DatasetCatalog.register("ship_val", lambda: ld.Register().get_dicts(val_keys))
# MetadataCatalog.get("ship_val").set(thing_classes=['__background__', '1', '2', '3', '4', '5'])


# 进行预览
# 测试一下结果

json_data = JSON('../../dataset/split')

info_Register(json_data, 0.2).register_dataset()
ship_metadata = MetadataCatalog.get("train")


for d in info_Register(json_data, 0.2).data_info.data_set:
    # e = dict(d)
    # name = e.get("file_name")
    # print(name)
    img = cv2.imread(d['file_name'])
    visualizer = myVisualization(img[:, :, ::-1], metadata={}, scale=1)
    vis = visualizer.draw_dataset_dict(d, )
    cv2.imshow("hi", vis.get_image()[:, :, ::-1])
    cv2.waitKey(0)
