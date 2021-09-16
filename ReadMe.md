# RS-Detection-ToolKit-zh

## Introduction

本项目主要提供一个针对旋转目标检测的项目模板。项目基于pytorch和detectron2。在原框架的基础上，本项目：

* 设计了单独的数据格式接口
* 提供一些针对遥感数据的预处理和统计方法
* 提供一些斜框目标检测模型config
* 提供影像拆分和合并工具

本项目正在建设中！欢迎一起维护～



## 文件组织

```
Project
--data
    data_reader.py  #数据接口，父类
    json_API.py  #继承数据接口，json数据读取
    txt_API.py  #继承数据接口，json数据读取
    rotated_data_loader.py #旋转数据注册类
    rotated_visualization  #旋转数据显示类别（非必要
    
--model
    faster_rcnn_R_50_FPN_3x.yaml  # 官方代码库 copy
    Base-RCNN-FPN.yaml  # 同上
    my_config.yaml  # 自己的训练配置文件
    
--utils
    data_statistic.py # 目标检测数据统计，包括面积、长宽比、数据分布等等6项指标
    data_utils.py  # 草稿
    ImageSplit.py  # 对目标检测数据进行指定大小的切分，并提供坐标排序等增强（详细见后）
    ImageMerge.py # 与ImageSplit相对，复原切割数据
    load_data_viewer.py # 对注册进去的数据进行预览（基于visualization），保证数据训练正确
    origin_data_viewer.py # 对原始数据进行预览（基于cv），依次绘制边，确定坐标点顺序的正确性
   
   -dehaze  #包含两种去雾算法，FFA-Net和GCANet
   # -extension_polyiou # c++ extension of nms 
train_net.py  # 官方代码库 copy 并自行修改
predictor.py  # 模型预测 输出
single_predictor.py # 单张图片模型输出
train.sh
train_resume.sh
eval.sh

```

博客：dinghye.gitee.io / dinghye.github.io(不一定稳定)



## 数据接口说明

1. 通用数据接口：本项目设计一个data_reader的接口，下设不同种类数据继承该接口，以规范数据读取、注册、统计等工作。该接口包含：

   * self.data_set # 存储数据，数据格式为：

     > ```
     > data={
     >     "file_name": filename,  # instance的位置，建议使用绝对路径
     >     "image_id": idx,        # 对于每一个instance须有有一个唯一id
     >     "height": height,     # image的高度
     >     "width": width,          # image的宽度
     >     "annotations":[          # 对于这一张图片的标注,可以是多个
     >         {
     >             "bbox": [cx,cy,w,h,a]  # 和bbox_mode相关联
     >            "bbox_mode": BoxMode.  
     >            "category_id": 
     >         }
     >         {
     >             "bbox": [cx,cy,w,h,a]
     >            "bbox_mode": BoxMode.
     >            "category_id": 
     >         }
     >     ],
     > }
     > ```

   * self.data_path # 数据读取的根目录
   * self.data_type # 暂未使用，类似bboxmodel（考虑直接使用这个，区分正框数据和斜框）

   

# 实现的功能

1. 数据格式转换：COCO，txt（DOTA），json
2. 数据统计基础统计，包括：每张图含有实例分布图；单个框长宽比分布(数据是一样的,只是方便设置xlabel)；总体框的长宽比分布；总体框的面积分布；不同类别框的数目统计；单个类别内框的长宽比分布；单个类别内框的面积分布;(对于旋转框)标注框旋转统计
3. 数据集的裁剪与标签重写
4. 旋转框数据注册与预览
5. 旋转框的模型设置



# Log
2021/9/16 dinghye:
What's new:

1. 实现合并图像时的非极大值抑制（nms=True）python version

   值得注意的是这里的nms_thresh可以改小一点，disable py_cpu_nms_poly

2. 完善load_data_viewer,rotated_visualizatiion

   （比较重要）之前版本的viewer只能显示目标的位置，其类别显示和颜色都是错误的。针对该问题，我们进行了修改完善。

Future work：

1. 常参数定义的过于混乱，比如说CLS_Name这种，与任务相关的参数贯穿始终，这样的东西应该统一管理！
2. 关于evaluator：由于现在的evaluator的test数据还是针对的小图，其实好像拼图与否影响是比较小的ho……所以打算先这样吧
3. 关于坐标转换：建议使用Detectron2 的转换！





2021/9/13 dinhye:
What's new: 

1. 新添加了ImageMerge功能，能够根据split文件拼出完整图用于大图预测。

   @该部分参考[DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) 值得注意的是，由于原图被拆分，造成一些目标被分割，所以在这一步当中可能出现一个目标被多次检测的情况。针对这一问题有效的策略是使用nms来提高结果的精度。ImageMerge.py 这一部分提供两种计算nms的方法：py_cpu_nms和py_cpu_nms_poly 。后者受到swig/c++的支持，需要在utils/extension_polyiou 下安装：

   ```shell
   # step1: install swig
   sudo  apt-get install swig
   
   # step2: create the c++ extension for python 
   swig -c++ -python polyiou.i
   python setup.py build_ext --inplace
   
   ```

   > 2021/9/16 nms实现
   >
   > 2021/9/14 nms未完全完善，暂时建议在使用的过程中nms=False. Detectorn2 封装好的nms！

2. 修改完善ImageSplit，现支持纯图片拆分（以方便在预测的过程中使用）

3. 修改完善predictor，按照指定格式保存结果，预测流程为：拆分图片->预测->结果合并（暂时disable预览，可通过修改utils/origin_data_viewer和预测结果来显示。

4. 移动dataprocess（dehaze）到data

Future work：

* evaluator没有修改好，需要根据predictor的流程来修改evaluator的process
* time comsuming的问题：由于在预测过程中需要拆解图片和合并图片，使得预测时间大大延长。可以考虑用c++的一些extension来帮助实现这部分功能（或者multi-process）





2021/8/2 dinghye: 
coco转换时会出现height和width不精准的情况，提交coco_editor.py patch 修改矫正；添加cascade模型(model/cascade)；添加旋转框角度统计(data_statistic.py)；
重要修改: eval-only 旋转框验证完善（RotatedCOCOEvalutate）

# 感谢&相关工作
* https://github.com/CAPTAIN-WHU/DOTA_devkit
* https://github.com/zhilin007/FFA-Net#ffa-net-feature-fusion-attention-network-for-single-image-dehazing-aaai-2020
* https://github.com/cddlyf/GCANet
