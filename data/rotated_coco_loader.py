import os

import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json

# NOT FINISHED YET!!!!!
# TEST ONLY
from detectron2.utils.visualizer import Visualizer
CLS_N= ['__background__','0','1','2', '3', '4', '5', '6', '7', '8', '9', '10'] 

class coco_Register:
    """用于注册自己的数据集"""

    # @todo: ugly code here
    # CLASS_NAMES = ['__background__', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']  # 保留 background 类
    ROOT = "../dataset/splitMyDataset"
    CLASS_NAMES= ['__background__','0','1','2', '3', '4', '5', '6', '7', '8', '9', '10'] 
    #
    def __init__(self):
        self.CLASS_NAMES = coco_Register.CLASS_NAMES or ['__background__', ]
        # 数据集路径
        self.DATASET_ROOT = coco_Register.ROOT
        # ANN_ROOT = os.path.join(self.DATASET_ROOT, 'COCOformat')
        self.ANN_ROOT = self.DATASET_ROOT

        self.TRAIN_PATH = os.path.join(self.DATASET_ROOT, 'images/train')
        self.VAL_PATH = os.path.join(self.DATASET_ROOT, 'images/val')
        self.TRAIN_JSON = os.path.join(self.ANN_ROOT, 'annotations/train.json')
        self.VAL_JSON = os.path.join(self.ANN_ROOT, 'annotations/val.json')
        # VAL_JSON = os.path.join(self.ANN_ROOT, 'test.json')

        # 声明数据集的子集
        self.PREDEFINED_SPLITS_DATASET = {
            "train": (self.TRAIN_PATH, self.TRAIN_JSON),
            "val": (self.VAL_PATH, self.VAL_JSON),
        }

    def register_dataset(self):
        """
        purpose: register all splits of datasets with PREDEFINED_SPLITS_DATASET
        注册数据集（这一步就是将自定义数据集注册进Detectron2）
        """
        for key, (image_root, json_file) in self.PREDEFINED_SPLITS_DATASET.items():
            self.register_dataset_instances(name=key,
                                            json_file=json_file,
                                            image_root=image_root)

    # @staticmethod
    def register_dataset_instances(self, name, json_file, image_root):
        """
        purpose: register datasets to DatasetCatalog,
                 register metadata to MetadataCatalog and set attribute
        注册数据集实例，加载数据集中的对象实例
        """

        DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
        MetadataCatalog.get(name).set(json_file=json_file,
                                      image_root=image_root,
                                      evaluator_type="RotatedCOCOEvaluator")
                                      # evaluator_type="coco") # 这里注意！如果是rotated coco要选择对应的evaluator_type!

    def plain_register_dataset(self):
        """注册数据集和元数据"""
        # 训练集
        DatasetCatalog.register("train", lambda: load_coco_json(self.TRAIN_JSON, self.TRAIN_PATH))
        MetadataCatalog.get("train").set(thing_classes=CLS_N,  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
                                         #evaluator_type='coco',  # 指定评估方式
                                         evaluator_type='RotatedCOCOEvaluator',
                                         json_file=self.TRAIN_JSON,
                                         image_root=self.TRAIN_PATH)
        # DatasetCatalog.register("coco_my_val", lambda: load_coco_json(VAL_JSON, VAL_PATH, "coco_2017_val"))
        # 验证/测试集
        DatasetCatalog.register("val", lambda: load_coco_json(self.VAL_JSON, self.VAL_PATH))
        MetadataCatalog.get("val").set(thing_classes=CLS_N,  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
                                    #    evaluator_type='coco',  # 指定评估方式
                                        evaluator_type='RotatedCOCOEvaluator',

                                        json_file=self.VAL_JSON,
                                        image_root=self.VAL_PATH)
        # print(self.CLASS_NAMES)

    def checkout_dataset_annotation(self, name="coco_my_val"):
        """
        查看数据集标注，可视化检查数据集标注是否正确，
        这个也可以自己写脚本判断，其实就是判断标注框是否超越图像边界
        可选择使用此方法
        """
        # dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH, name)
        dataset_dicts = load_coco_json(self.TRAIN_JSON, self.TRAIN_PATH)
        print(len(dataset_dicts))
        for i, d in enumerate(dataset_dicts, 0):
            # print(d)
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(name), scale=1.5)
            vis = visualizer.draw_dataset_dict(d)
            # cv2.imshow('show', vis.get_image()[:, :, ::-1])
            cv2.imwrite('out/' + str(i) + '.jpg', vis.get_image()[:, :, ::-1])
            # cv2.waitKey(0)
            if i == 200:
                break
