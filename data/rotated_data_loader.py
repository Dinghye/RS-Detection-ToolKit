import math
from collections import Counter

import cv2
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from sklearn.model_selection import train_test_split
from detectron2.utils.visualizer import Visualizer, _create_text_labels, GenericMask, ColorMode

from data.rotated_visualization import myVisualization

"""
data={
    "file_name": filename,  # instance的位置，建议使用绝对路径
    "image_id": idx,        # 对于每一个instance须有有一个唯一id
    "height": height,		# image的高度
    "width": width,			# image的宽度
    "annotations":[			# 对于这一张图片的标注,可以是多个
        {
            "bbox": [cx,cy,w,h,a]	# 和bbox_mode相关联
        	"bbox_mode": BoxMode.
        	"category_id": 
        }
        {
            "bbox": [cx,cy,w,h,a]
        	"bbox_mode": BoxMode.
        	"category_id": 
        }
    ],
}
"""


class info_Register:
    """用于注册自己的数据集"""

    def __init__(self, data_info, test_size):
        # self.CLASS_NAMES = Register.CLASS_NAMES or ['__background__', ]
        self.data_info = data_info
        self.test_size = test_size
        # self.cls_name = self.get_cls_name(data_info.data_set)

    # @todo: move to data API
    # def get_cls_name(self, dataset):
    #     cls = []
    #
    #     for i in dataset:
    #         for j in i['annotations']:
    #             cls.append(j['category_id'])
    #     a = list(np.unique(cls))
    #     a = list(map(int, a))
    #     a.append('__background__')
    #     return a

    def register_dataset(self):
        """
        purpose: register all splits of datasets with PREDEFINED_SPLITS_DATASET
        注册数据集（这一步就是将自定义数据集注册进Detectron2）
        """

        # 注册进去数据
        l_keys = range(1, len(self.data_info.data_set) + 1)
        train_keys, val_keys = train_test_split(l_keys, test_size=0.2)

        print("train:" + str(len(train_keys)) + "," + "val:" + str(len(val_keys)))

        # train_keys = list(map(int, train_keys))
        # val_keys = list(map(int, val_keys))
        # print(train_keys)

        self.plain_register_dataset(train_keys, val_keys)

    # basic
    def get_dicts(self, ids):
        data = []
        count = 0

        for i in ids:
            ist = self.data_info.data_set[i - 1]
            ist['ids'] = count

            data.append(ist)
            count += 1
        return data

    def plain_register_dataset(self, train_key, val_key):
        """prepare cls num 4 register"""
        for i in range(0, len(self.data_info.data_set)):
            for j in range(0, len(self.data_info.data_set[i]['annotations'])):
                self.data_info.data_set[i]['annotations'][j]['category_id'] = self.data_info.cls_dir[
                    self.data_info.data_set[i]['annotations'][j][
                        'category_id']]  # @todo: look so complicated and stupid!

        """注册数据集和元数据"""
        # 训练集
        DatasetCatalog.register("train", lambda: self.get_dicts(train_key))

        MetadataCatalog.get("train").set(evaluator_type='RotatedCOCOEvaluator',
                                         thing_classes=self.data_info.data_cls)  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭

        # 验证/测试集
        DatasetCatalog.register("val", lambda: self.get_dicts(val_key))
        MetadataCatalog.get("val").set(evaluator_type='RotatedCOCOEvaluator',
                                       thing_classes=self.data_info.data_cls)  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭

        # print(self.data_info.data_cls)

    def checkout_dataset_annotation(self, name="coco_my_val"):
        """
        !!! 这个只针对正框数据，斜框数据请使用myVisualization
        查看数据集标注，可视化检查数据集标注是否正确，
        这个也可以自己写脚本判断，其实就是判断标注框是否超越图像边界
        可选择使用此方法
        """
        dataset_dicts = self.data_info.data_set(range(1, 11))  # self.get_dicts(range(1, 11))
        print(len(dataset_dicts))
        for d in dataset_dicts:
            e = dict(d)
            name = e.get("file_name")
            print(name)
            print(e.items())
            img = cv2.imread(name)
            visualizer = myVisualization(img[:, :, ::-1], metadata={}, scale=1)
            vis = visualizer.draw_dataset_dict(e, )
            cv2.imshow("check", vis.get_image()[:, :, ::-1])
            cv2.waitKey(0)
