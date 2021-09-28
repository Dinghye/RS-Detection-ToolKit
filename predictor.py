import cv2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import os
import utils.ImageSplit as sp
import utils.ImageMerge as mr
import math

# @todo : fix ugly code
TEST_PATH = "../dataset/trainMyDataset/images/val"
OUT_PATH = " ../dataset/trainMyDataset/image/outcome"
CLASS_LIST = ['__background__', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
CLASS_CHAR_LIST = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']


# you can construct outcome here as ur need!
def data_construct(outputs):
    js = {}
    js["image_name"] = d
    labels = []
    if outputs['instances'] is not None:
        instances = outputs['instances'].get_fields()
        # instances = outputs['instances']['_fields']
        instan_num = len(instances['pred_boxes'])
        for i in range(instan_num):
            label = {}
            categoty_id = int(instances['pred_classes'].tolist()[i])
            label['category_id'] = CLASS_CHAR_LIST[categoty_id]
            label['points'] = rotateTo4Point(instances['pred_boxes'].tensor.tolist()[i])
            label['confidence'] = instances['scores'].tolist()[i]
            labels.append(label)
    js['labels'] = labels
    return js


def rotateTo4Point(params):
    # new code for convert
    cnt_x, cnt_y, w, h, angle = params
    area = w * h
    theta = angle * math.pi / 180.0
    c = math.cos(theta)
    s = math.sin(theta)
    rect = [(-w / 2, h / 2), (-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2)]
    for (xx, yy) in rect:
        x = s * yy + c * xx + cnt_x
        y = c * yy - s * xx + cnt_y
    rotated_rect = [[s * yy + c * xx + cnt_x, c * yy - s * xx + cnt_y] for (xx, yy) in rect]

    return rotated_rect


if __name__ == "__main__":
    setup_logger()

    cfg = get_cfg()
    cfg.merge_from_file("model/my_config.yaml")
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 模型阈值
    cfg.MODEL.WEIGHTS = "output/model_final.pth"
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    predictor = DefaultPredictor(cfg)
    MetadataCatalog.get("train").set(things_class=CLASS_LIST)

    split = sp.splitbase(TEST_PATH, os.path.join(OUT_PATH, 'split'))
    split.splitdata(1)

    result = []
    for d in os.listdir(TEST_PATH):
        im = cv2.imread(os.path.join(TEST_PATH, d))
        out = predictor(im)
        out_js = data_construct(out)
        if not len(out_js.get('labels')) == 0:
            result.append(out_js)

    mr.mergebase(result, OUT_PATH + 'result.json', nms=True)

    # for d in random.sample(os.listdir(TEST_PATH), 20):
    #     # im = cv2.imread(TEST_PATH + str(d) + ".png")
    #     im = cv2.imread(os.path.join(TEST_PATH,d))
    #
    #     outputs = predictor(im)
    #
    #     print(outputs)
    #     # print(TEST_PATH + str(d) + ".png")
    #     print(d)
    #
    #     # ship_mentadata = MetadataCatalog.get("ship_train")
    #     # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.8)
    #     # v = myVisualization(im[:, :, ::-1],  MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.8)
    #     v = myVisualization(im[:, :, ::-1], MetadataCatalog.get("train"), scale=0.8)
    #     v = v.draw_instance_predictions(outputs['instances'].to("cpu"))
    #
    #     cv2.imshow("test", v.get_image()[:, :, ::-1])
    #     cv2.waitKey(0)
