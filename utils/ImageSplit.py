import json
import os
import numpy as np
import math
import cv2
import shapely.geometry as shgeo
import utils.data_utils as util
import copy
from multiprocessing import Pool
from functools import partial
import sys

sys.path.append("..")

# from data.json_API import get_file_info
# from data.json_API import get_file_info
import data.json_API

"""
This function is aim to split the data (origin from DOTA_devkit)
1. split image by given size
2. choose best point order to help regression (optional)
3. image who was split will be renamed by "originname__1__left x__left y". 
   And it helps Merge image. A new json file will also be generate, which 
   contain the object info of the new image. A new attribute will also be 
   writen called 'difficult'. If obj is broken by the crop box, the attribute
    is marked as "2".  

(support Json api)
"""


def choose_best_pointorder_fit_another(poly1, poly2):
    """
        To make the two polygons best fit with each point
    """
    x1 = poly1[0]
    y1 = poly1[1]
    x2 = poly1[2]
    y2 = poly1[3]
    x3 = poly1[4]
    y3 = poly1[5]
    x4 = poly1[6]
    y4 = poly1[7]
    combinate = [np.array([x1, y1, x2, y2, x3, y3, x4, y4]), np.array([x2, y2, x3, y3, x4, y4, x1, y1]),
                 np.array([x3, y3, x4, y4, x1, y1, x2, y2]), np.array([x4, y4, x1, y1, x2, y2, x3, y3])]
    dst_coordinate = np.array(poly2)
    distances = np.array([np.sum((coord - dst_coordinate) ** 2) for coord in combinate])
    sorted = distances.argsort()
    return combinate[sorted[0]]


def cal_line_length(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


def split_single_warp(info, split_base, rate):
    split_base.SplitSingle(info, rate)


class splitbase():
    def __init__(self,
                 basepath,
                 outpath,
                 code='utf-8',
                 gap=256,
                 subsize=512,
                 thresh=0.7,
                 choosebestpoint=True,
                 padding = True,
                 ext='.png',
                 num_process=8,
                 datatype='json',  # notsupportyet

                 ):
        """
        :param basepath: base path for dota data
        :param outpath: output base path for dota data,
        the basepath and outputpath have the similar subdirectory, 'images' and 'labelTxt'
        :param code: encodeing format of txt file
        :param gap: overlap between two patches
        :param subsize: subsize of patch
        :param thresh: the thresh determine whether to keep the instance if the instance is cut down in the process of split
        :param choosebestpoint: used to choose the first point for the
        :param ext: ext for the image format
        """
        self.basepath = basepath
        self.outpath = outpath
        self.code = code
        self.gap = gap
        self.subsize = subsize
        self.slide = self.subsize - self.gap
        self.thresh = thresh
        self.padding = padding
        self.info_group = []
        # self.imagepath = os.path.join(self.basepath, 'images')
        # self.labelpath = os.path.join(self.basepath, 'labelTxt')
        # self.outimagepath = os.path.join(self.outpath, 'images')
        # self.outlabelpath = os.path.join(self.outpath, 'labelTxt')
        self.outimagepath = self.outpath
        self.outlabelpath = self.outpath
        self.choosebestpoint = choosebestpoint
        self.ext = ext
        self.num_process = num_process
        self.pool = Pool(num_process)

        if not os.path.exists(self.outimagepath):
            os.makedirs(self.outimagepath)
        if not os.path.exists(self.outlabelpath):
            os.makedirs(self.outlabelpath)

        ## point: (x, y), rec: (xmin, ymin, xmax, ymax)
        # def __del__(self):
        #     self.f_sub.close()
        ## grid --> (x, y) position of grids
        # @todo: determined by data type! default:json
        js = data.json_API.JSON(self.basepath)
        self.info_group = js.get_file_info(self.basepath)  # get_file_info(self.basepath)  # save dataset info

    def polyorig2sub(self, left, up, poly):
        polyInsub = np.zeros(len(poly))
        for i in range(int(len(poly) / 2)):
            polyInsub[i * 2] = int(poly[i * 2] - left)
            polyInsub[i * 2 + 1] = int(poly[i * 2 + 1] - up)
        return polyInsub

    def calchalf_iou(self, poly1, poly2):
        """
            It is not the iou on usual, the iou is the value of intersection over poly1
        """

        inter_poly = poly1.intersection(poly2)
        inter_area = inter_poly.area
        poly1_area = poly1.area
        half_iou = inter_area / poly1_area

        return inter_poly, half_iou

    def saveimagepatches(self, img, subimgname, left, up):
        subimg = copy.deepcopy(img[up: (up + self.subsize), left: (left + self.subsize)])
        outdir = os.path.join(self.outimagepath, subimgname + self.ext)
        cv2.imwrite(outdir, subimg)

    def GetPoly4FromPoly5(self, poly):
        distances = [cal_line_length((poly[i * 2], poly[i * 2 + 1]), (poly[(i + 1) * 2], poly[(i + 1) * 2 + 1])) for i
                     in range(int(len(poly) / 2 - 1))]
        distances.append(cal_line_length((poly[0], poly[1]), (poly[8], poly[9])))
        pos = np.array(distances).argsort()[0]
        count = 0
        outpoly = []
        while count < 5:
            # print('count:', count)
            if (count == pos):
                outpoly.append((poly[count * 2] + poly[(count * 2 + 2) % 10]) / 2)
                outpoly.append((poly[(count * 2 + 1) % 10] + poly[(count * 2 + 3) % 10]) / 2)
                count = count + 1
            elif (count == (pos + 1) % 5):
                count = count + 1
                continue

            else:
                outpoly.append(poly[count * 2])
                outpoly.append(poly[count * 2 + 1])
                count = count + 1
        return outpoly

    def save_file(self, path, item):

        # 先将字典对象转化为可写入文本的字符串
        item = json.dumps(item)

        try:
            if not os.path.exists(path):
                with open(path, "w", encoding='utf-8') as f:
                    f.write(item + ",\n")
                    print("^_^ write success")
            else:
                with open(path, "a", encoding='utf-8') as f:
                    f.write(item + ",\n")
                    print("^_^ write success")
        except Exception as e:
            print("write error==>", e)

    def savepatches(self, resizeimg, objects, subimgname, left, up, right, down):
        # outdir = os.path.join(self.outlabelpath, subimgname + '.txt')
        os.mknod(os.path.join(self.outlabelpath, subimgname + '.json'))
        outdir = os.path.join(self.outlabelpath, subimgname + '.json')

        # mask_poly = []
        imgpoly = shgeo.Polygon([(left, up), (right, up), (right, down),
                                 (left, down)])
        # print([(left, up), (right, up), (right, down),
        #        (left, down)])
        # with codecs.open(outdir, 'w', self.code) as f_out:
        out_json_dict = {}
        out_json_dict["version"] = "4.5.6"
        out_json_dict["flags"] = {}

        # print(len(objects))
        shapes = []

        for obj in objects:
            single_o = {}
            gtpoly = shgeo.Polygon([(obj['poly'][0], obj['poly'][1]),
                                    (obj['poly'][2], obj['poly'][3]),
                                    (obj['poly'][4], obj['poly'][5]),
                                    (obj['poly'][6], obj['poly'][7])])
            # print(obj['poly'])
            if (gtpoly.area <= 0):
                continue
            inter_poly, half_iou = self.calchalf_iou(gtpoly, imgpoly)
            # print(inter_poly)
            # print(half_iou)

            # print('writing...')
            if (half_iou == 1):
                polyInsub = self.polyorig2sub(left, up, obj['poly'])

                single_o["lable"] = obj['name']
                points = [[polyInsub[0], polyInsub[1]],
                          [polyInsub[2], polyInsub[3]],
                          [polyInsub[4], polyInsub[5]],
                          [polyInsub[6], polyInsub[7]]
                          ]
                single_o["points"] = points
                single_o["group_id"] = "null"
                single_o["shape_type"] = "polygon"
                single_o["flags"] = {}
                single_o["difficult"] = obj['difficult']
                # outline = ' '.join(list(map(str, polyInsub)))
                # outline = outline + ' ' + obj['name'] + ' ' + str(obj['difficult'])
                # f_out.write(outline + '\n')

            elif (half_iou > 0):
                inter_poly = shgeo.polygon.orient(inter_poly, sign=1)
                out_poly = list(inter_poly.exterior.coords)[0: -1]
                if len(out_poly) < 4:
                    continue

                out_poly2 = []
                for i in range(len(out_poly)):
                    out_poly2.append(out_poly[i][0])
                    out_poly2.append(out_poly[i][1])

                if (len(out_poly) == 5):
                    out_poly2 = self.GetPoly4FromPoly5(out_poly2)
                elif (len(out_poly) > 5):
                    """
                        if the cut instance is a polygon with points more than 5, we do not handle it currently
                    """
                    continue
                if (self.choosebestpoint):
                    out_poly2 = choose_best_pointorder_fit_another(out_poly2, obj['poly'])

                polyInsub = self.polyorig2sub(left, up, out_poly2)

                for index, item in enumerate(polyInsub):
                    if (item <= 1):
                        polyInsub[index] = 1
                    elif (item >= self.subsize):
                        polyInsub[index] = self.subsize
                # outline = ' '.join(list(map(str, polyInsub)))
                if (half_iou > self.thresh):
                    # outline = outline + ' ' + obj['name'] + ' ' + str(obj['difficult'])
                    single_o = {}
                    single_o["lable"] = obj['name']
                    points = [[polyInsub[0], polyInsub[1]],
                              [polyInsub[2], polyInsub[3]],
                              [polyInsub[4], polyInsub[5]],
                              [polyInsub[6], polyInsub[7]]
                              ]
                    single_o["points"] = points
                    single_o["group_id"] = "null"
                    single_o["shape_type"] = "polygon"
                    single_o["flags"] = {}
                    single_o["difficult"] = obj['difficult']
                else:
                    ## if the left part is too small, label as '2'
                    # outline = outline + ' ' + obj['name'] + ' ' + '2'
                    single_o = {}
                    single_o["label"] = obj['name']
                    points = [[polyInsub[0], polyInsub[1]],
                              [polyInsub[2], polyInsub[3]],
                              [polyInsub[4], polyInsub[5]],
                              [polyInsub[6], polyInsub[7]]
                              ]
                    single_o["points"] = points  # polyInsub
                    single_o["group_id"] = "null"
                    single_o["shape_type"] = "polygon"
                    single_o["flags"] = {}
                    single_o["difficult"] = 2

            # print(single_o)
            # f_out.write(outline + '\n')
            # else:
            #   mask_poly.append(inter_poly)
            if len(single_o):
                shapes.append(single_o)
        # print(len(shapes))
        out_json_dict["shapes"] = shapes
        out_json_dict["imagePath"] = subimgname + '.png'
        out_json_dict["imageWidth"] = right - left
        out_json_dict["imageHeight"] = down - up
        out_json_dict["imageData"] = "null"

        json_str = json.dumps(out_json_dict, indent=4)
        with open(outdir, 'w') as json_file:
            json_file.write(json_str)
        # self.save_file(outdir, out_json_dict)
        self.saveimagepatches(resizeimg, subimgname, left, up)

    def SplitSingle(self, info, rate):
        """
            split a single image and ground truth
        :param info: image info
        :param rate: the resize scale for the image
        :param extent: the image format
        :return:
        """
        try:
            img = cv2.imread(os.path.join(self.basepath, info['imagePath']))
            objects = util.parse_dota_poly2(info)
            for obj in objects:
                obj['poly'] = list(map(lambda x: rate * x, obj['poly']))
            outbasename = info["imagePath"].split('.')[0] + '__' + str(rate) + '__'
        except:
            # if np.shape(img) == ():
            img = cv2.imread(os.path.join(info))
            # objects = []
            print(info)
            outbasename = info.split('/')[-1].split('.')[0] + '__' + str(rate) + '__'

        if (rate != 1):
            resizeimg = cv2.resize(img, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
        else:
            resizeimg = img

        weight = np.shape(resizeimg)[1]
        height = np.shape(resizeimg)[0]

        left, up = 0, 0
        while (left < weight):
            if (left + self.subsize >= weight):
                left = max(weight - self.subsize, 0)
            up = 0
            while (up < height):
                if (up + self.subsize >= height):
                    up = max(height - self.subsize, 0)
                right = min(left + self.subsize, weight - 1)
                down = min(up + self.subsize, height - 1)
                subimgname = outbasename + str(left) + '___' + str(up)
                # self.f_sub.write(name + ' ' + subimgname + ' ' + str(left) + ' ' + str(up) + '\n')
                try:
                    self.savepatches(resizeimg, objects, subimgname, left, up, right, down)
                except:
                    self.saveimagepatches(resizeimg, subimgname, left, up)
                if (up + self.subsize >= height):
                    break
                else:
                    up = up + self.slide
            if (left + self.subsize >= weight):
                break
            else:
                left = left + self.slide

    def splitdata(self, rate):
        """
        :param rate: resize rate before cut
        """
        # imagelist = []
        # imagenames = []
        # for i in self.info_group:
        #     imagenames.append(i["imagePath"])
        #     imagelist.append(os.path.join(self.basepath, i["imagePath"]))

        if self.num_process == 1:
            if len(self.info_group) != 0:
                for name in self.info_group:
                    self.SplitSingle(name, rate)
            else:
                names = []
                for root, dirs, files in os.walk(self.basepath):
                    for file in files:
                        if os.path.splitext(file)[1] == '.png':
                            names.append(os.path.join(root, file))
                # print(names)
                for name in names:
                    self.SplitSingle(name, rate)
        else:
            if len(self.info_group) != 0:
                worker = partial(split_single_warp, split_base=self, rate=rate)
                self.pool.map(worker, self.info_group)
            else:
                names = []
                for root, dirs, files in os.walk(self.basepath):
                    for file in files:
                        if os.path.splitext(file)[1] == '.png':
                            names.append(os.path.join(root, file))

                worker = partial(split_single_warp, split_base=self, rate=rate)
                self.pool.map(worker, names)

        # imagenames = [util.custombasename(x) for x in imagelist if (util.custombasename(x) != 'Thumbs')]
        # for name in imagenames:
        #     self.SplitSingle(name, rate, self.ext)
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

if __name__ == '__main__':
    # example usage of ImgSplit
    split = splitbase(r'../../dataset/testfull',
                      r'../../dataset/testsplit')
    split.splitdata(1)
