import os
import numpy as np
import utils.data_utils as util
import re
import time

import json

# due to different object might be split as 2(even more object),
# we use a nms when merging (but not necessary!)

nms_thresh = 0.3
pic_type = '.png'

class mergebase():
    def __init__(self, js_res, outpath ='', nms=True):
        self.namebox = {}
        self.piclist = []
        self.final_out = []

        for img in js_res:
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
                origpoly = self.poly2originpoly(poly, x, y, rate)
                det = origpoly
                det.append(confidence)
                det = list(map(float, det))
                det.append(object['category_id'])
                # dets.append(det)
                try:
                    self.namebox[oriname].append(det)
                except:
                    self.namebox[oriname] = []  # init
            # self.namebox[oriname]append(det)
            self.piclist.append(oriname)

        if nms:
            self.namebox = self.nmsbynamedict(self.namebox, nms_thresh)

        self.piclist = np.array(self.piclist)


        for img in (np.unique(self.piclist)):
            out = {}
            out['image_name'] = img + pic_type
            out['labels'] = []
            for det in self.namebox[img]:
                obj = {}
                obj['category_id'] = det[-1]
                obj['points'] = np.array(det[0:len(det)-2]).reshape(4,2).tolist()
                obj['confidence'] = det[len(det)-2]
                out['labels'].append(obj)
            self.final_out.append(out)

        if outpath != '':
            with open (outpath,'w') as f:
                json.dump(self.final_out,f,indent=2)


    def poly2originpoly(self, poly, x, y, rate):
        origpoly = []
        for i in range(int(len(poly) / 2)):
            tmp_x = float(poly[i * 2] + x) / float(rate)
            tmp_y = float(poly[i * 2 + 1] + y) / float(rate)
            origpoly.append(tmp_x)
            origpoly.append(tmp_y)
        return origpoly

    def nmsbynamedict(self, nameboxdict, thresh):
        nameboxnmsdict = {x: [] for x in nameboxdict}
        for imgname in nameboxdict:
            # print('imgname:', imgname)
            # keep = py_cpu_nms(np.array(nameboxdict[imgname]), thresh)
            # print('type nameboxdict:', type(nameboxnmsdict))
            # print('type imgname:', type(imgname))
            # print('type nms:', type(nms))
            keep = self.py_cpu_nms(np.array(nameboxdict[imgname]), thresh)
            # print('keep:', keep)
            outdets = []
            # print('nameboxdict[imgname]: ', nameboxnmsdict[imgname])
            for index in keep:
                # print('index:', index)
                outdets.append(nameboxdict[imgname][index])
            nameboxnmsdict[imgname] = outdets
        return nameboxnmsdict



    # beta version : @todo: time consuming !
    def py_cpu_nms(self, dets, thresh):
        """Pure Python NMS baseline."""
        # print('dets:', dets)

        x1 = list(map(float, dets[:, 0]))
        y1 = list(map(float, dets[:, 1]))
        x2 = list(map(float, dets[:, 2]))
        y2 = list(map(float, dets[:, 3]))
        scores = list(map(float, dets[:, 4]))

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

    # beta version: to get single image object nms outcome!
    def py_cpu_nms_poly(self, dets, thresh):
        import utils.extension_polyiou.polyiou as polyiou

        scores = dets[:, 8]
        polys = []
        # category_id = []
        for i in range(len(dets)):
            tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                               dets[i][2], dets[i][3],
                                               dets[i][4], dets[i][5],
                                               dets[i][6], dets[i][7]])
            polys.append(tm_polygon)

        order = scores.argsort()[::-1]  # reverse! I hate python :(

        keep = []
        while order.size > 0:
            ovr = []
            i = order[0]
            keep.append(i)
            for j in range(order.size - 1):
                iou = polyiou.iou_poly(polys[i], polys[order[j + 1]])
                ovr.append(iou)
            ovr = np.array(ovr)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

