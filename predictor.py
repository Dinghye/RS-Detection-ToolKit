import random
import cv2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from data.rotated_visualization import myVisualization

# TEST_PATH = "/home/dinghye/下载/科目四初赛第一阶段/test1/"
# CLASS_LIST = {'__background__', 1, 2, 3, 4, 5}
TEST_PATH = "../dataset/trainMyDataset"
CLASS_LIST= ['__background__','0','1','2', '3', '4', '5', '6', '7', '8', '9', '10']

if __name__ == "__main__":
    setup_logger()

    cfg = get_cfg()
    cfg.merge_from_file("model/my_config.yaml")
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 模型阈值
    cfg.MODEL.WEIGHTS = "output/model_final.pth"
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    predictor = DefaultPredictor(cfg)
    MetadataCatalog.get("train").set(things_class=CLASS_LIST)

    for d in random.sample(range(1, 261), 20):
        im = cv2.imread(TEST_PATH + str(d) + ".tif")

        outputs = predictor(im)

        print(outputs)
        print(TEST_PATH + str(d) + ".tif")

        # ship_mentadata = MetadataCatalog.get("ship_train")
        # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.8)
        # v = myVisualization(im[:, :, ::-1],  MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.8)
        v = myVisualization(im[:, :, ::-1], MetadataCatalog.get("train"), scale=0.8)
        v = v.draw_instance_predictions(outputs['instances'].to("cpu"))

        cv2.imshow("test", v.get_image()[:, :, ::-1])
        cv2.waitKey(0)
