import json
import os


def get_info_json(json_file):
    """
    :param json_file: *.json file name
    :return: json info with a dict
    """
    # jf = os.path.dirname(json_file)
    with open(json_file) as i:
        info = json.load(i)
    return info


def get_file_info(filename):
    """
    :param filename:
    :return: a info group in this file
    """
    info_group = []

    for root, dirs, files in os.walk(filename):
        for file in files:
            if file.endswith(".json"):
                info_group.append(get_info_json(os.path.join(root, file)))
    return info_group


def json_to_txt(info_group):
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

# DIR = '../../dataset/test'
# print(get_file_info(DIR))
