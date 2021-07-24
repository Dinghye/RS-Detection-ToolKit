import math
from collections import Counter

import cv2
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from sklearn.model_selection import train_test_split
from detectron2.utils.visualizer import Visualizer, _create_text_labels, GenericMask, ColorMode


class myVisualization(Visualizer):
    """用于显示旋转过后的数据（继承Visualizer）"""

    def draw_dataset_dict(self, dic):
        annos = dic.get("annotations", None)
        if annos:
            if "segmentation" in annos[0]:
                masks = [x["segmentation"] for x in annos]
            else:
                masks = None
            if "keypoints" in annos[0]:
                keypts = [x["keypoints"] for x in annos]
                keypts = np.array(keypts).reshape(len(annos), -1, 3)
            else:
                keypts = None

            boxes = [BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYWHA_ABS) for x in annos]

            labels = [x["category_id"] for x in annos]
            names = self.metadata.get("thing_classes", None)
            if names:
                labels = [names[i] for i in labels]
            labels = [
                "{}".format(i) + ("|crowd" if a.get("iscrowd", 0) else "")
                for i, a in zip(labels, annos)
            ]
            self.overlay_instances(labels=labels, boxes=boxes, masks=masks, keypoints=keypts)

        sem_seg = dic.get("sem_seg", None)
        if sem_seg is None and "sem_seg_file_name" in dic:
            sem_seg = cv2.imread(dic["sem_seg_file_name"], cv2.IMREAD_GRAYSCALE)
        if sem_seg is not None:
            self.draw_sem_seg(sem_seg, area_threshold=0, alpha=0.5)
        return self.output

    def draw_instance_predictions(self, predictions):
        """
        Draw instance-level prediction results on an image.
        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").
        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None
        # print(classes)
        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            assert predictions.has("pred_masks"), "ColorMode.IMAGE_BW requires segmentations"
            self.output.img = self._create_grayscale_image(
                (predictions.pred_masks.any(dim=0) > 0).numpy()
            )
            alpha = 0.3

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=classes,
            # classes=classes,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output


class Register:
    """用于注册自己的数据集"""
    CLASS_NAMES = ['__background__', '1', '2', '3', '4', '5']  # 保留 background 类
    DATA_PATH = "/home/dinghye/下载/科目四初赛第一阶段/train/"  # 数据的路径
    TEST_SIZE = 0.2  # 测试集比例

    def __init__(self):
        self.CLASS_NAMES = Register.CLASS_NAMES or ['__background__', ]

    def read_labels(self, path, name):
        """
        purpose: 读取单个存放标注的txt数据
        :param path: 文件所在文件夹
        :param name: 文件名称（不带后缀名）
        :return: 带标注的数组
        """
        f = open(path + "labels/" + str(name) + ".txt")
        line = f.readline()
        data_list = []
        while line:
            num = list(map(float, line.split()))
            data_list.append(num)
            line = f.readline()
        f.close()
        data_array = np.array(data_list)
        # print(data_array)
        return data_array

    def get_length(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

    def get_type(self, array):
        type = []
        for i in array:
            type.append(i[0])
        print(type)
        return type

    def get_sorted(self, annotation):
        # gt rank by y
        list = [annotation[1], annotation[3], annotation[5], annotation[7]]
        # print(rank)
        rank = sorted(range(len(list)), key=lambda k: list[k])
        ann = []
        ann.append(annotation[0])
        ann.append(annotation[rank[0] * 2 + 1])
        ann.append(annotation[rank[0] * 2 + 2])
        ann.append(annotation[rank[1] * 2 + 1])
        ann.append(annotation[rank[1] * 2 + 2])
        ann.append(annotation[rank[3] * 2 + 1])
        ann.append(annotation[rank[3] * 2 + 2])
        ann.append(annotation[rank[2] * 2 + 1])
        ann.append(annotation[rank[2] * 2 + 2])
        return ann

    def get_abbox(self, annotation):
        """
        purpose: 用于返回对应的bbox数据
        :param annotation: 输入的数组，其格式为：[label,x1,y1,x2,y2,x3,y3,x4,y4]
        :return: 输出XYWHA_ABS格式的bbox，其格式为：[centerX, centerY, w, h, a]（a为旋转角度）
        """
        # annotation = self.get_sorted(annotation)
        centerx = (annotation[1] + annotation[3] + annotation[5] + annotation[7]) / 4
        centery = (annotation[2] + annotation[4] + annotation[6] + annotation[8]) / 4
        h = math.sqrt(math.pow((annotation[1] - annotation[3]), 2) + math.pow(
            (annotation[2] - annotation[4]), 2))
        w = math.sqrt(math.pow((annotation[1] - annotation[7]), 2) + math.pow(
            (annotation[2] - annotation[8]), 2))
        if h < w:
            a = - math.degrees(math.atan2((annotation[8] - annotation[2]), (annotation[7] - annotation[1])))
        else:
            temp = h
            h = w
            w = temp
            a = - math.degrees(math.atan2((annotation[4] - annotation[2]), (annotation[3] - annotation[1])))
        # a = ((annotation[1] + annotation[3]) / 2 - centerx) / self.get_length((annotation[1] + annotation[3]) / 2,
        #                                                                       (annotation[2] + annotation[4]) / 2,
        #                                                                       centerx, centery)

        # print('bad')

        # a = math.acos(((annotation[1] + annotation[3]) / 2 - centerx) / (h / 2))
        # a = 0
        return [centerx, centery, w, h, a]

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
                ann["bbox"] = self.get_abbox(j)
                annotations.append(ann)
                ann = {}
            data["annotations"] = annotations
            dataset_dicts.append(data)
            data = {}
        return dataset_dicts

    def train_test_split(self, total_keys, total_type, tesize):
        types = total_type.unique()
        type_key = [] * len(types)
        print(types)
        count = 0
        for t in total_type:
            index = types.index(t)
            type_key[index].append(total_keys[count])
            count = count + 1
        print(type_key)

    def instance_type(self, annotations):
        count = Counter(annotations)
        return max(zip(count.values(), count.keys()))

    def register_dataset(self):
        """
        purpose: register all splits of datasets with PREDEFINED_SPLITS_DATASET
        注册数据集（这一步就是将自定义数据集注册进Detectron2）
        """

        # 这里利用train_test_split函数（sklearn包）进行测试集和训练集的划分
        total_csv_annotations = range(1, 2009)
        # total_keys = list(total_csv_annotations)
        type = []
        # read data and annotation to identified instance type(in helping split train and val)
        for i in range(1, 2009):
            this = self.read_labels(self.DATA_PATH, i)
            annotations = this[:, 0]
            type.append(self.instance_type(annotations)[1])

        # get train and val key by type
        train = []
        val = []
        for i in range(1, 6):
            location = 0
            find = i
            l_keys = []
            for i in range(type.count(find)):
                location += type[location:].index(find)
                l_keys.append(location + 1)
                location += 1
            try:
                train_keys, val_keys = train_test_split(l_keys, test_size=0.2)
            except:
                train_keys = []
                val_keys = []
            train.extend(train_keys)
            val.extend(val_keys)
        # 注册进去数据
        print("train:" + str(len(train)) + "," + "val:" + str(len(val)))

        self.plain_register_dataset(train, val)

    def plain_register_dataset(self, train_key, val_key):
        """注册数据集和元数据"""

        # 训练集
        DatasetCatalog.register("ship_train", lambda: self.get_dicts(train_key))
        MetadataCatalog.get("ship_train").set(evaluator_type='RotatedCOCOEvaluator',
                                              thing_classes=self.CLASS_NAMES)  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭

        # 验证/测试集
        DatasetCatalog.register("ship_val", lambda: self.get_dicts(val_key))
        MetadataCatalog.get("ship_val").set(evaluator_type='RotatedCOCOEvaluator',
                                            thing_classes=self.CLASS_NAMES)  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭

    def checkout_dataset_annotation(self, name="coco_my_val"):
        """
        !!! 这个只针对正框数据，斜框数据请使用myVisualization
        查看数据集标注，可视化检查数据集标注是否正确，
        这个也可以自己写脚本判断，其实就是判断标注框是否超越图像边界
        可选择使用此方法
        """
        dataset_dicts = self.get_dicts(range(1, 11))
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

# Register().register_dataset()
