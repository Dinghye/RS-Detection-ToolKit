import os
import cv2 as cv
import numpy as np

from data.json_API import JSON

"""
Use CV to display annotation images.
Depending on the ordering of the points, 
the sides of the annotation box are displayed 
in different colors.

Mainly used to check data.
(support JSON API,txt API)
"""

# img_path = '/home/dinghye/下载/科目四初赛第一阶段/train/'  # 把图片直接放在同一文件夹下
# from data.json_API import get_file_info


img_path = '../../dataset/testfull'

def read_labels(path, name):
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


def get_sorted(annotation):
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

    # print(ann)

    return ann


# for i in range(1, 300):
#     img = read_labels(img_path, i)
#     img_raw = cv.imread(img_path + 'images/' + str(i) + '.tif')
#     for j in img:
#         j = [int(x) for x in j]
#         j = get_sorted(j)
#         # ann = np.array([[j[1], j[2]], [j[3], j[4]]], np.int32)
#         # ann = ann.reshape(-1, 1, 2)/
#         cv.line(img_raw, (j[1], j[2]), (j[3], j[4]), (0, 255, 255), thickness=2)
#         # ann = np.array([j[3], j[4]], [j[5], j[6]], np.int32)
#         # ann = ann.reshape(-1, 1, 2)
#         cv.line(img_raw, (j[3], j[4]), (j[5], j[6]), (255, 0, 255), thickness=2)
#         # ann = np.array([j[5], j[6]], [j[7], j[8]], np.int32)
#         # ann = ann.reshape(-1, 1, 2)
#         cv.line(img_raw, (j[5], j[6]), (j[7], j[8]), (255, 255, 0), thickness=2)
#         # ann = np.array([j[7], j[8]], [j[1], j[2]], np.int32)
#         # ann = ann.reshape(-1, 1, 2)
#         cv.line(img_raw, (j[7], j[8]), (j[1], j[2]), (0, 0, 255), thickness=2)
#     cv.imshow('image', img_raw)
#     cv.waitKey(0)
js = JSON(img_path)
info_group = js.data_set

for i in info_group:
    img_raw = cv.imread(os.path.join(img_path, i["imagePath"]))

    for j in i['shapes']:
        p = []
        for a in range(0, 4):
            p.append(list(map(int, j['points'][a])))

        # p = get_sorted(j['points'])
        # ann = np.array([[j[1], j[2]], [j[3], j[4]]], np.int32)
        # ann = ann.reshape(-1, 1, 2)/
        cv.line(img_raw, (p[0][0], p[0][1]), (p[1][0], p[1][1]), (0, 255, 255), thickness=2)
        # ann = np.array([j[3], j[4]], [j[5], j[6]], np.int32)
        # ann = ann.reshape(-1, 1, 2)
        cv.line(img_raw, (p[1][0], p[1][1]), (p[2][0], p[2][1]), (255, 0, 255), thickness=2)
        # ann = np.array([j[5], j[6]], [j[7], j[8]], np.int32)
        # ann = ann.reshape(-1, 1, 2)
        cv.line(img_raw, (p[2][0], p[2][1]), (p[3][0], p[3][1]), (255, 255, 0), thickness=2)
        # ann = np.array([j[7], j[8]], [j[1], j[2]], np.int32)
        # ann = ann.reshape(-1, 1, 2)
        cv.line(img_raw, (p[3][0], p[3][1]), (p[0][0], p[0][1]), (0, 0, 255), thickness=2)
        # print(j['difficult'])
    img_raw = cv.resize(img_raw, (1024, 1024))
    cv.imshow('image', img_raw)
    cv.waitKey(0)
