# Faster VoxelPose
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/end-to-end-estimation-of-multi-person-3d/3d-multi-person-pose-estimation-on-campus)](https://paperswithcode.com/sota/3d-multi-person-pose-estimation-on-campus?p=end-to-end-estimation-of-multi-person-3d) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/end-to-end-estimation-of-multi-person-3d/3d-multi-person-pose-estimation-on-shelf)](https://paperswithcode.com/sota/3d-multi-person-pose-estimation-on-shelf?p=end-to-end-estimation-of-multi-person-3d) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/end-to-end-estimation-of-multi-person-3d/3d-multi-person-pose-estimation-on-cmu)](https://paperswithcode.com/sota/3d-multi-person-pose-estimation-on-cmu?p=end-to-end-estimation-of-multi-person-3d)



This is the official implementation for:
> [**Faster VoxelPose: Real-time 3D Human Pose Estimation by Orthographic Projection**](https://arxiv.org/abs/2004.06239),            
> Hang Ye, Wentao Zhu, Chunyu Wang, Rujie Wu, and Yizhou Wang       
> *ECCV 2022 ([arXiv -TODO-]())*


<img src="data/panoptic2.gif" width="800"/>


## Installation
1. Clone this repo, and we'll call the directory that you cloned as ${ROOT}.
2. Install dependencies.

## Data preparation

### Shelf/Campus datasets
1. Download the datasets from http://campar.in.tum.de/Chair/MultiHumanPose and extract them under `${ROOT}/data/Shelf` and `${ROOT}/data/Campus`, respectively.

2. We have processed the camera parameters to our formats and you can download them from this repository. They lie in `${ROOT}/data/Shelf/` and `${ROOT}/data/Campus/`,  respectively.

3. Due to the limited and incomplete annotations of the two datasets, we don't train our model using this dataset. Instead, we directly use the 2D pose estimator trained on COCO, and use independent 3D human poses from the Panoptic dataset to train our 3D model. It lies in `${ROOT}/data/panoptic_training_pose.pkl`. See our paper for more details.

4. For testing, we first estimate 2D poses and generate 2D heatmaps for these two datasets in this repository.  The predicted poses can also download from the repository. They lie in `${ROOT}/data/Shelf/` and `${ROOT}/data/Campus/`,  respectively. You can also use the models trained on COCO dataset (like HigherHRNet) to generate 2D heatmaps directly.

The directory tree should look like this:
```
${ROOT}
|-- data
    |-- Shelf
    |   |-- Camera0
    |   |-- ...
    |   |-- Camera4
    |   |-- actorsGT.mat
    |   |-- calibration_shelf.json
    |   |-- pred_shelf_maskrcnn_hrnet_coco.pkl
    |-- Campus
    |   |-- Camera0
    |   |-- Camera1
    |   |-- Camera2
    |   |-- actorsGT.mat
    |   |-- calibration_campus.json
    |   |-- pred_campus_maskrcnn_hrnet_coco.pkl
    |-- panoptic_training_pose.pkl
```


### CMU Panoptic dataset
1. Download the dataset by following the instructions in [panoptic-toolbox](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox) and extract them under `${ROOT}/data/Panoptic/`.
- You can only download those sequences you need. You can also just download a subset of camera views by specifying the number of views (HD_Video_Number) and changing the camera order in `./scripts/getData.sh`. The sequences and camera views used in our project can be obtained from our paper.
- Note that we only use HD videos,  calibration data, and 3D Body Keypoint in the codes. You can comment out other irrelevant codes such as downloading 3D Face data in `./scripts/getData.sh`.
2. Download the pretrained backbone model from [pretrained backbone](https://1drv.ms/u/s!AjX41AtnTHeTjn3H9PGSLcbSC0bl?e=cw7SQg) and place it here: `${ROOT}/models/pose_resnet50_panoptic.pth.tar` (ResNet-50 pretrained on COCO dataset and finetuned jointly on Panoptic dataset and MPII).

The directory tree should look like this:
```
${ROOT}
|-- models
|   |-- pose_resnet50_panoptic.pth.tar
|-- data
    |-- Panoptic
        |-- 16060224_haggling1
        |   |-- hdImgs
        |   |-- hdvideos
        |   |-- hdPose3d_stage1_coco19
        |   |-- calibration_160224_haggling1.json
        |-- 160226_haggling1  
        |-- ...
```

## Training
### CMU Panoptic dataset

Train and validate on the five selected camera views. You can specify the GPU devices and batch size per GPU in the config file. 
```
python run/train.py --cfg configs/panoptic/jln64.yaml
```
### Shelf/Campus datasets
```
python run/train.py --cfg configs/shelf/jln64.yaml
python run/train.py --cfg configs/campus/jln64.yaml
```

## Evaluation
### CMU Panoptic dataset

Evaluate the models. It will print evaluation results to the screen./
```
python run/validate.py --cfg configs/panoptic/jln64.yaml
```
### Shelf/Campus datasets

It will print the PCP results to the screen.
```
python run/validate.py --cfg configs/shelf/jln64.yaml
python run/validate.py --cfg configs/campus/jln64.yaml
```

## Citation
If you use our code or models in your research, please cite with:
```

```


# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
# Faster-VoxelPose
