import json
import math
import os
import numpy as np
from detectron2.structures import BoxMode
from data.data_info import DATA_INFO

"""
Data structure example:
{
    "version": "4.5.6",
    "flags": {},
    "shapes": [
        {
            "label": "I",
            "points": [
                [
                    1439.4602654098444,
                    2771.754849493459
                ],
                [
                    1416.8179779890258,
                    2735.3149181755794
                ],
                [
                    1447.4909114325212,
                    2716.2560080747667
                ],
                [
                    1470.1331988533398,
                    2752.6959393926463
                ]
            ],
            "group_id": null,
            "shape_type": "polygon",
            "flags": {}
        }
    ],
    "imagePath": "3.png",
    "imageHeight": 4096,
    "imageWidth": 4096,
    "imageData": null
}

"""


# @todo: NOT hard code
CLS_NUM = {
    # "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10, "K": 11
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9, "K": 10
}


class JSON(DATA_INFO):

    def __init__(self, data_path):
        self.data_path = data_path
        self.data_set = self.get_dataset(data_path)
        # print(self.data_set)
        self.data_cls, self.cls_dir = self.get_cls_name(self.data_set)

    def get_cls_name(self, dataset):
        data_cls = []
        for i in dataset:
            for j in i['annotations']:
                data_cls.append(j['category_id'])
        data_cls = list(np.unique(data_cls))
        print(data_cls)
        cls_dir = {}
        cls_num = 0

        cls = []
        for i in data_cls:
            cls_dir[i] = cls_num
            cls.append(cls_num)
            cls_num += 1
        cls.append('__background__')
        return cls, cls_dir

    def get_dataset(self, data_path):
        info_group = self.get_file_info(data_path)
        return self.json_to_dataset(info_group)

    def _get_info_json(self, json_file):
        """
        :param json_file: *.json file name
        :return: json info with a dict
        """
        # jf = os.path.dirname(json_file)
        with open(json_file) as i:
            info = json.load(i)
        return info

    def get_file_info(self, filename):
        """
        :param filename:
        :return: a info group in this file
        """
        info_group = []

        for root, dirs, files in os.walk(filename):
            for file in files:
                if file.endswith(".json"):
                    info_group.append(self._get_info_json(os.path.join(root, file)))
        return info_group

    def json_to_txt(self, info_group):
        """
        function: turn json to txt info form
        """
        dataset = []
        for i in info_group:
            single_img = []
            for j in i['shapes']:
                single_obj = []
                points = j['points']
                single_obj.append(j['label'])
                single_obj.append(points[0][0])
                single_obj.append(points[0][1])
                single_obj.append(points[1][0])
                single_obj.append(points[1][1])
                single_obj.append(points[2][0])
                single_obj.append(points[2][1])
                single_obj.append(points[3][0])
                single_obj.append(points[3][1])

                single_img.append(single_obj)

            dataset.append(single_img)

        return dataset

    def _get_abbox(self, points):
        """
                purpose: 用于返回对应的bbox数据
                :param points: 输入的数组，其格式为：[label,x1,y1,x2,y2,x3,y3,x4,y4]
                :return: 输出XYWHA_ABS格式的bbox，其格式为：[centerX, centerY, w, h, a]（a为旋转角度）
                """
        # annotation = self.get_sorted(annotation)
        centerx = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4
        centery = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4
        h = math.sqrt(math.pow((points[0][1] - points[1][1]), 2) + math.pow(
            (points[0][0] - points[1][0]), 2))
        w = math.sqrt(math.pow((points[0][0] - points[3][0]), 2) + math.pow(
            (points[0][1] - points[3][1]), 2))
        if h < w:
            a = - math.degrees(math.atan2((points[3][1] - points[0][1]), (points[3][0] - points[0][0])))
        else:
            temp = h
            h = w
            w = temp
            a = - math.degrees(math.atan2((points[1][1] - points[0][1]), (points[1][0] - points[0][0])))

        return [centerx, centery, w, h, a]

    # DIR = '../../dataset/test'
    # print(get_file_info(DIR))
    def json_to_dataset(self, info_group):
        """
        get detectron2 register form data
        """
        dataset = []
        id_generator = 0
        for img in info_group:
            id_generator += 1
            single_img = {}
            # print(os.path.join(self.data_path, img['imagePath']))
            single_img['file_name'] = os.path.join(self.data_path, img['imagePath'])
            single_img['image_id'] = id_generator
            single_img['height'] = img['imageHeight']
            single_img['width'] = img['imageWidth']
            single_img['annotations'] = []
            if len(img['shapes']) != 0:
                for obj in img['shapes']:
                    box = {}
                    box['bbox'] = self._get_abbox(obj['points'])
                    box['bbox_mode'] = BoxMode.XYWHA_ABS
                    # print(CLS_NUM[obj['label']])
                    box['category_id'] = obj['label']
                    single_img['annotations'].append(box)
            dataset.append(single_img)

        return dataset
    """
    directly change data to coco form
    @TODO: independent class/function
    """

# example
# js = JSON('../../dataset/train')
# js.info_to_rotated_coco()
