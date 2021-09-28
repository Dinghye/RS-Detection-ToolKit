import cv2
# import data.txt_data_loader as ld
from data.json_API import JSON
from data.rotated_data_loader import info_Register
from utils.rotated_visualization import myVisualization

"""
Use the data display that comes with detectron2 to view annotation data
Mainly used to check if the data is correctly registered。

(txt AIP support)
"""

json_data = JSON('../../dataset/testfull')


# todo :fix ugly code
CLASS_CHAR_LIST = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
CLS_NUM = {
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9, "K": 10
}
COLOR = ['aliceblue', 'azure', 'coral', 'firebrick', 'green', 'greenyellow', 'honeydew', 'lightcoral', 'lightpink',
         'purple', 'red', 'violet']
for d in info_Register(json_data, 0.2).data_info.data_set:

    # print(d)
    img = cv2.imread(d['file_name'])

    metadata = {}
    metadata['thing_classes'] = CLASS_CHAR_LIST
    metadata['thing_colors'] = COLOR
    for i in d['annotations']:
        i['category_id'] = CLS_NUM[i['category_id']]

    visualizer = myVisualization(img[:, :, ::-1], metadata=metadata, scale=1)
    vis = visualizer.draw_dataset_dict(d, )
    # cv2.imshow("hi", vis.get_image()[:, :, ::-1])
    cv2.imwrite(d['file_name'].split('/')[-1].split('.')[0] + '_' + '.png', vis.get_image()[:, :, ::-1])
    # cv2.waitKey(0)
