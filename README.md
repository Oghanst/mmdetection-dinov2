# 开始
本项目主要基于mmengine-template实现 DINO-DINOv2模型

## 安装
请首先安装 pytorch1.12.1，cuda11.3和 cudnn8-devel。
```
git clone https://github.com/Oghanst/mmdetection-dinov2.git
cd mmdetection-dinov2.git
pip install -r requirements.txt
```

## 训练
1. 设置数据集：

在路径 `./configs/_base_/datasets` 创建自己的数据集配置文件，具体可以参考文件`coco_detection_visdrone`的设置。

对于里面的`dataset_type = 'CocoDataset'`需要在`./mmdet/datasets/coco.py`设计一个自己的新类，可以参考里面的`class CocoDataset_Smokefire(CocoDataset): `的设置。

然后在datasets的__init__.py文件里面对自己设计的新类进行注册。

2. 设置PYTHONPATH
```
export PYTHONPATH = $PWD:$PYTHONPATH
```

3. 如何训练
参考configs里面的注释输入命令行`./configs/dino`
```
python tools/train.py configs/dino/yourConfigName.py --work-dir work_dirs/dino 
```



