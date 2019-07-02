# Automatic Spine segmentation by 3D DeepLab

## Abstract

This model segment  CT spine images automatically by three steps. First, locate every vertebrae and find center coordinates by YOLOv3. Second, generate patches by coordinates and segment vertebrae by 3D DeepLabv3+. Third, identify vertebrae from segmented image patches by 3D Xception.

## Folders and Files

darknet/
: (folder for YOLOv3 model)

data/
: (folder for data)

weights/
: (folder for trained weights)

cross_weights/
: (folder for cross validation weights)

result/
: (folder for output images)

main.py
: (main file for training and cross validation)

model3d.py
: (model of 3D DeepLab)

xception3d.py
: (model of 3D Xception)

result.ipynb
: (demo result for training dataset)

result_point.ipynb
: (demo result for pointrobotics dataset)

## Commands for main.py

```
python3 main.py [mode] -m [model_name] -e [epoch_number] -gpu [gpu_id] -data [data_path]
```


* [mode]: 'train' or 'cross'
    * train or cross validation
    * (cross only work for Deeplab)
* [model_name]: 'deeplab' or 'xception'
* [epoch_number]: number of epoch (default 100)
* [gpu_id]: integer, id of gpu
* [data_path]: path to data, ex: './data/seg_data_original'

### Example commands

#### Deeplab training

```
python3 main.py train -m deeplab -gpu 0 -data ./data/seg_data_original
```

#### Xception training

```
python3 main.py train -m xception -gpu 0 -data ./data/seg_data_original
```

#### Cross validation for deeplab

```
python3 main.py cross -gpu 1 -data ./data/seg_data_original
```

## Recommend library version

**tensorflow-gpu (2.0.0a0)**

**Keras (2.1.0)**

**h5py (2.9.0)**


opencv-python (3.4.4.19)

matplotlib (3.0.2)

numpy (1.16.2)

SimpleITK (1.1.0)

scipy (1.2.1)

pydicom (1.2.2)

cuda 9.0

## Requirements

### Ram

Training: 9GB
Testing: 7GB

### Build

You may need to makefile for darknet.


