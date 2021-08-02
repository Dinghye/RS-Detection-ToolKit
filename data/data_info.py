"""
This is a interface of data
mainly to help dataloader to register data
"""
import json
import os
import shutil

from sklearn.model_selection import train_test_split


class DATA_INFO():
    def __init__(self):
        self.data_type = ''  # useless yet ... rotated / positive
        self.data_set = []  # here is a container of formed data
        self.data_path = ''  # data root
        self.data_cls = []  # a list of cls name(including '__background__'
        self.cls_dir = {}  # a dict of cls ,eg: 'A':0

    def info_to_rotated_coco(self):
        saved_coco_path = self.data_path
        # 创建必须的文件夹
        if not os.path.exists('%sMyDataset/annotations/' % saved_coco_path):
            os.makedirs('%sMyDataset/annotations/' % saved_coco_path)
        if not os.path.exists('%sMyDataset/images/train/' % saved_coco_path):
            os.makedirs('%sMyDataset/images/train/' % saved_coco_path)
        if not os.path.exists('%sMyDataset/images/val/' % saved_coco_path):
            os.makedirs('%sMyDataset/images/val/' % saved_coco_path)

        # 按照键值划分数据
        total_keys = range(1, len(self.data_set) + 1)
        # print(total_keys)
        train_keys, val_keys = train_test_split(total_keys, test_size=0.1)
        print("train_n:", len(train_keys), 'val_n:', len(val_keys))

        # 把训练集转化为COCO的json格式
        train_instance = self._to_coco(train_keys)


        self._save_coco_json(train_instance, '%sMyDataset/annotations/train.json' % saved_coco_path)
        for key in train_keys:
            shutil.copy(self.data_set[key - 1]['file_name'], "%sMyDataset/images/train/" % saved_coco_path)
        for key in val_keys:
            shutil.copy(self.data_set[key - 1]['file_name'], "%sMyDataset/images/val/" % saved_coco_path)

        # 把验证集转化为COCO的json格式
        val_instance = self._to_coco(val_keys)
        self._save_coco_json(val_instance, '%sMyDataset/annotations/val.json' % saved_coco_path)

    def _to_coco(self, keys):
        instance = {}
        images = []
        annotation = []
        a_id = 0
        for key in keys:
            # create image
            image = {}
            image['height'] = self.data_set[key - 1]['height']
            image['width'] = self.data_set[key - 1]['width']
            image['id'] = self.data_set[key - 1]['image_id']
            image['file_name'] = self.data_set[key - 1]['file_name'].split('/')[-1]
            images.append(image)

            # create annotations

            for shape in self.data_set[key - 1]['annotations']:
                a = {}
                a_id += 1
                a['id'] = a_id
                a['image_id'] = key
                a['category_id'] = self.cls_dir[shape['category_id']]
                a['bbox'] = shape['bbox']
                a['bbox_mode'] = shape['bbox_mode']
                a['iscrowd'] = 0
                a['area'] = shape['bbox'][2] * shape['bbox'][3]
                annotation.append(a)

        # create categories
        categories = []
        for k, v in self.cls_dir.items():
            category = {}
            category['id'] = int(v)
            category['name'] = k
            categories.append(category)

        instance['info'] = 'geoxlab create'
        instance['license'] = ['license']
        instance['images'] = images
        instance['annotations'] = annotation
        instance['categories'] = categories
        # inst.append(instance)
        return instance

    def _save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w'), ensure_ascii=False, indent=2)  # indent=2 更加美观显示
