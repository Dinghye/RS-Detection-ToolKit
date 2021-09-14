### Dehaze

本项目针对图像去雾进行数据增强，包括FFA-Net算法和GCANet算法，这里两个算法均只使用原算法作者训练的模型对图像进行去雾，若要重新训练模型，则参考原作者github

FFA-Net算法：https://github.com/zhilin007/FFA-Net#ffa-net-feature-fusion-attention-network-for-single-image-dehazing-aaai-2020

GCANet算法：https://github.com/cddlyf/GCANet 

------

#### FFA-Net

训练模型：百度云 https://pan.baidu.com/s/1-pgSXN6-NXLzmTp21L_qIg  密码4gat

谷歌云 https://drive.google.com/drive/folders/19_lSUPrpLDZl9AyewhHBsHidZEpTMIV5?usp=sharing

模型存放在trained_models/下，图片默认存放在test_imgs/下

```
python test.py --task ots --test_imgs home/xxx/data/images
```

（这里选用遥感图像，所以推荐使用ots，即户外训练模型）

------

#### GCANet

图片默认存放在examples/下

```
python test.py --indir home/xxx/data/images
```

