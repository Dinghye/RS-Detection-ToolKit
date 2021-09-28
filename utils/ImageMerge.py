import os
import numpy as np
import utils.data_utils as util
import re
import time

import json

from multiprocessing import Pool
from functools import partial

# due to different object might be split as 2(even more object),
# we use a nms when merging (but not necessary!)

nms_thresh = 0.01
pic_type = '.png'
namebox = {}


def nmsbynamedict(nameboxdict, thresh):
    nameboxnmsdict = {x: [] for x in nameboxdict}
    for imgname in nameboxdict:
        # print('imgname:', imgname)
        # keep = py_cpu_nms(np.array(nameboxdict[imgname]), thresh)
        # print('type nameboxdict:', type(nameboxnmsdict))
        # print('type imgname:', type(imgname))
        # print('type nms:', type(nms))
        keep = py_cpu_nms(np.array(nameboxdict[imgname]), thresh)
        # print('keep:', keep)
        outdets = []
        # print('nameboxdict[imgname]: ', nameboxnmsdict[imgname])
        for index in keep:
            # print('index:', index)
            outdets.append(nameboxdict[imgname][index])
        nameboxnmsdict[imgname] = outdets
    return nameboxnmsdict

def mergesingle(nms, js_res, fullname):
        name = util.custombasename(fullname)
        print('name:', name)
        # dstname = os.path.join(dstpath, name + '.txt')
        global namebox
        # namebox = {}
        lable = []
        for i in js_res:
            if i["image_name"].split('__')[0] == fullname.split('/')[-1].split('.')[
                0]:  # fullname:'../input_path/1.png'
                lable.append(i)
        for img in lable:
            subname = img['image_name'].split('.')[0]  # image_name = "xxx__0___1.png"
            splitname = subname.split('__')
            oriname = splitname[0]
            pattern1 = re.compile(r'__\d+___\d+')

            x_y = re.findall(pattern1, subname)
            x_y_2 = re.findall(r'\d+', x_y[0])
            x, y = int(x_y_2[0]), int(x_y_2[1])

            pattern2 = re.compile(r'__([\d+\.]+)__\d+___')
            rate = re.findall(pattern2, subname)[0]

            # dets = []
            for object in img['labels']:
                confidence = object['confidence']
                # todo: time consuming....
                poly = []
                for coordinate in object['points']:
                    poly.append(float(coordinate[0]))
                    poly.append(float(coordinate[1]))
                origpoly = poly2originpoly(poly, x, y, rate)
                det = origpoly
                det.append(confidence)
                det = list(map(float, det))
                det.append(object['category_id'])
                # dets.append(det)
                try:
                    namebox[oriname].append(det)
                except:
                    namebox[oriname] = []  # init

        if nms:
            namebox = nmsbynamedict(namebox, nms_thresh)
            print(namebox)


def poly2originpoly(poly, x, y, rate):
    origpoly = []
    for i in range(int(len(poly) / 2)):
        tmp_x = float(poly[i * 2] + x) / float(rate)
        tmp_y = float(poly[i * 2 + 1] + y) / float(rate)
        origpoly.append(tmp_x)
        origpoly.append(tmp_y)
    return origpoly

    # nameboxdic.append(namebox)
    # print(len(nameboxdic))
    # piclist = np.array(piclist)


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    # print('dets:', dets)

    x1 = dets[:, 0].astype(float)
    y1 = dets[:, 1].astype(float)
    x2 = dets[:, 2].astype(float)
    y2 = dets[:, 3].astype(float)
    scores = dets[:, 4].astype(float)

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    ## index for dets
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def mergebypoly(js_res, srcpath, dstpath, nms=True, num_process=16):
    """
       js_res : outcome
       dstpath: result files after merge and nms
       """
    # namebox = {}
    global namebox

    #
    # pool = Pool(num_process)
    filelist = util.GetFileFromThisRootDir(srcpath)


    for file in filelist:
        mergesingle(nms,js_res,file)
    # worker = partial(mergesingle, nms, js_res)
    # # pdb.set_trace()
    # pool.map(worker, filelist)

    final_out = []
    # print(nameboxdic)
    # for namebox in nameboxdic:
    for n, box in namebox.items():
        # img = filelist.split['.'][0]
        out = {}
        out['image_name'] = n + pic_type
        out['labels'] = []
        for det in box:
            obj = {}
            obj['category_id'] = det[-1]
            obj['points'] = np.array(det[0:len(det) - 2]).reshape(4, 2).tolist()
            obj['confidence'] = det[len(det) - 2]
            out['labels'].append(obj)
        final_out.append(out)

    if dstpath != '':
        with open(dstpath, 'w') as f:
            json.dump(final_out, f, indent=2)

# class mergebase():
#     def __init__(self, js_res, outpath='', nms=True, num_process=16):
#         self.namebox = {}
#         self.piclist = []
#         self.final_out = []
#         self.num_process = num_process
#         pool = Pool(16)
#         mergesingle_fn = partial(mergesingle,js_res)
