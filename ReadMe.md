# RS-Detection-ToolKit-zh

## Introduction

本项目主要提供一个针对旋转目标检测的项目模板。项目基于pytorch和detectron2。在原框架的基础上，本项目：

* 设计了单独的数据格式接口
* 提供一些针对遥感数据的预处理和统计方法
* 提供一些斜框目标检测模型config
* .....

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



# TODO List

* 数据接口设计不完全，命名不清晰

* 一些针对性的数据增强方法

* 一些针对遥感的模型修改与测试

  

# LOG
2021/8/2 dinghye: 
coco转换时会出现height和width不精准的情况，提交coco_editor.py patch 修改矫正；添加cascade模型(model/cascade)；添加旋转框角度统计(data_statistic.py)；
重要修改: eval-only 旋转框验证完善（RotatedCOCOEvalutate）
