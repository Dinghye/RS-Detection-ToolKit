import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.append("..")

# import data.json_API

from data.json_API import JSON



"""

this statistic is based on txt form data...


Perform 6 statistics of target detection data
1. the amount of data contained in each graph
2. the aspect ratio distribution of the overall target
3. the area distribution of the overall target
4. bibliographic statistics of different categories of targets
5. distribution of aspect ratios within a single category
6. area distribution within a single category
7. 旋转角度统计(info type only)
"""


# 每张图含有实例分布图
def get_object_number_rank(dataset):
    obj_num = []
    for i in dataset:
        obj_num.append(len(i))
    print(obj_num)
    plt.hist(obj_num, bins=40, color='steelblue')
    plt.xlabel('Instance contained in each diagram')
    plt.ylabel('Frequency')
    plt.savefig('instance.png')
    plt.show()


# 单个框长宽比分布(数据是一样的,只是方便设置xlabel)
def get_lenWid_rank_type(dataset, type):
    ratio = []
    for img in dataset:
        for obj in img:
            l1 = clc_length(obj[1], obj[2], obj[3], obj[4])
            l2 = clc_length(obj[3], obj[4], obj[5], obj[6])
            length = max(l1, l2)
            width = min(l1, l2)
            ratio.append(length / width)

    plt.hist(ratio, bins=60, color='steelblue')
    plt.title("Class '" + str(type) + "'")
    plt.xlabel("The length-width ratio of each target(length/width) ")
    plt.ylabel("Frequency")
    plt.savefig("Class '" + str(type) + 'width_length.png')
    plt.show()


# 总体框的长宽比分布
def get_lenWid_rank(dataset):
    ratio = []
    for img in dataset:
        for obj in img:
            l1 = clc_length(obj[1], obj[2], obj[3], obj[4])
            l2 = clc_length(obj[3], obj[4], obj[5], obj[6])
            length = max(l1, l2)
            width = min(l1, l2)
            if width == 0 or length == 0:
                print(obj[1], obj[2], obj[3], obj[4])
            else:
                ratio.append(length / width)
    plt.hist(ratio, bins=60, color='steelblue')
    plt.xlabel("The length-width ratio of each target(length/width)")
    plt.ylabel("Frequency")
    plt.savefig('ratio.png')
    plt.show()


def get_area_rank_type(dataset, type):
    area = []
    for img in dataset:
        for obj in img:
            l1 = clc_length(obj[1], obj[2], obj[3], obj[4])
            l2 = clc_length(obj[3], obj[4], obj[5], obj[6])
            length = max(l1, l2)
            width = min(l1, l2)
            area.append(length * width)
    plt.title("Class '" + str(type) + "'")
    plt.hist(area, bins=60, color='steelblue')
    plt.xlabel("The area of each target")
    plt.ylabel("Frequency")
    plt.savefig('Class' + str(type) + '.png')
    plt.show()


# 总体框的面积分布
def get_area_rank(dataset):
    area = []
    for img in dataset:
        for obj in img:
            l1 = clc_length(obj[1], obj[2], obj[3], obj[4])
            l2 = clc_length(obj[3], obj[4], obj[5], obj[6])
            length = max(l1, l2)
            width = min(l1, l2)
            area.append(length * width)

    plt.hist(area, bins=60, color='steelblue')
    plt.xlabel("The area of each target")
    plt.ylabel("Frequency")
    plt.savefig('area.png')
    plt.show()


# 不同类别框的数目统计
def get_type_rank(dataset):
    TYPE = []
    for img in dataset:
        for obj in img:
            TYPE.append(obj[0])

    statstic = pd.value_counts(TYPE)
    plt.bar([i for i, v in statstic.items()], [v for i, v in statstic.items()])
    plt.ylabel("Frequency")
    plt.savefig('frequency.png')
    plt.show()


# 单个类别内框的长宽比分布
def get_single_type_lenWid_ratio(dataset):
    cdataset = dataset.copy()
    for t in TYPE:
        for img_index in range(0, len(dataset)):
            img = dataset[img_index].copy()
            newimg = img.copy()
            deletcounter = 0

            for obj_index in range(0, len(img)):
                # if int(img[obj_index][0]) != int(t):
                if img[obj_index][0] != t:
                    newimg = np.delete(newimg, obj_index - deletcounter, axis=0)
                    deletcounter += 1

            cdataset[img_index] = newimg.copy()

        get_lenWid_rank_type(cdataset, t)
        cdataset = dataset.copy()


# 单个类别内框的面积分布
def get_single_type_area(dataset):
    # type = [1.0, 2.0, 3.0, 4.0, 5.0]
    cdataset = dataset.copy()
    for t in TYPE:
        for img_index in range(0, len(dataset)):
            img = dataset[img_index].copy()
            newimg = img.copy()
            deletcounter = 0

            for obj_index in range(0, len(img)):
                # if int(img[obj_index][0]) != int(t):
                if img[obj_index][0] != t:
                    newimg = np.delete(newimg, obj_index - deletcounter, axis=0)
                    deletcounter += 1

            cdataset[img_index] = newimg.copy()

        get_area_rank_type(cdataset, t)
        cdataset = dataset.copy()


# 计算长度
def clc_length(x1, y1, x2, y2):
    x1 = float(x1)
    x2 = float(x2)
    y1 = float(y1)
    y2 = float(y2)

    return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))



# here is data_info type stastic !
# this function is 4 rotated dataset
def get_rotated_angle(data_info):
    # +180°~-180°，以30°为一个区间做统计
    angle_30 = []
    angle_60 = []
    for i in data_info:
        for j in i['annotations']:
            flag_30 = 0
            flag_60 = 0
            if j['bbox'][-1]< -150:
                flag_30 = 0
                flag_60 = 0
            elif -150 <= j['bbox'][-1] < -120 :
                flag_30 = 1
                flag_60 = 0
            elif -120 <= j['bbox'][-1] < -90 :
                flag_30 = 2
                flag_60 = 1
            elif -90 <= j['bbox'][-1] < -60 :
                flag_30 = 3
                flag_60 = 1
            elif -60 <= j['bbox'][-1] < -30 :
                flag_30 = 4
                flag_60 = 2 
            elif -30 <= j['bbox'][-1] < 0 :
                flag_30 = 5
                flag_60 = 2
            elif 0 <= j['bbox'][-1] < 30 :
                flag_30 = 6
                flag_60 = 3
            elif 30 <= j['bbox'][-1] < 60 :
                flag_30 = 7
                flag_60 = 3
            elif 60 <= j['bbox'][-1] < 90 :
                flag_30 = 8
                flag_60 = 4
            elif 90 <= j['bbox'][-1] < 120 :
                flag_30 = 9
                flag_60 = 4
            elif 120 <= j['bbox'][-1] < 150 :
                flag_30 = 10
                flag_60 = 5
            elif 150 <= j['bbox'][-1] < 180 :
                flag_30 = 11
                flag_60 = 5

            angle_30.append(flag_30)
            angle_60.append(flag_60)

    # 绘制30
    statstic = pd.value_counts(angle_30)
    plt.bar([i for i, v in statstic.items()], [v for i, v in statstic.items()])
    plt.ylabel("Frequency")
    plt.savefig('angle30_frequency.png')
    # plt.show()
    plt.close()

    # 绘制60
    statstic = pd.value_counts(angle_60)
    plt.bar([i for i, v in statstic.items()], [v for i, v in statstic.items()])
    plt.ylabel("Frequency")
    plt.savefig('angle60_frequency.png')
    

# dataset = []
# for i in range(1, 2009):
#     path = "labels/" + str(i) + ".txt"
#     img = read_labels(path)
#     dataset.append(img)

TYPE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
DIR = '../../dataset/train'
# dataset = get_file_info(DIR)
# dataset = json_to_txt(dataset)

#
# get_single_type_lenWid_ratio(dataset)

# get_single_type_area(dataset)

js = JSON(DIR)
get_rotated_angle(js.data_set)
