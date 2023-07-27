<h1 align="center">Faster VoxelPose: Real-time 3D Human Pose Estimation by Orthographic Projection <br> (ECCV 2022)</h1>

</div>

<div align="left">

  <a>![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)</a>
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
  [![arXiv](https://img.shields.io/badge/arXiv-2207.10955-b31b1b.svg)](https://arxiv.org/pdf/2207.10955.pdf)
  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/faster-voxelpose-real-time-3d-human-pose/3d-multi-person-pose-estimation-on-campus)](https://paperswithcode.com/sota/3d-multi-person-pose-estimation-on-campus?p=faster-voxelpose-real-time-3d-human-pose)
  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/faster-voxelpose-real-time-3d-human-pose/3d-multi-person-pose-estimation-on-shelf)](https://paperswithcode.com/sota/3d-multi-person-pose-estimation-on-shelf?p=faster-voxelpose-real-time-3d-human-pose)
  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/faster-voxelpose-real-time-3d-human-pose/3d-multi-person-pose-estimation-on-cmu)](https://paperswithcode.com/sota/3d-multi-person-pose-estimation-on-cmu?p=faster-voxelpose-real-time-3d-human-pose)
</div>

<img src="assets/band.jpg" width="800"/>


## Introduction
This is the official implementation of our paper:
> **[Faster VoxelPose: Real-time 3D Human Pose Estimation by Orthographic Projection](https://arxiv.org/pdf/2207.10955.pdf)**,            
> Hang Ye, Wentao Zhu, Chunyu Wang, Rujie Wu, and Yizhou Wang       
> *ECCV 2022*

The overall framework of Faster-VoxelPose is presented below.
<img src="assets/teaser.jpg" width="800"/>

## Environment
This project is developed using python 3.8, PyTorch 1.12.0, CUDA 11.3 (not necessary this version) on Ubuntu 16.04. 

```bash
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

## Data Preparation

### Download Dataset
Following [VoxelPose](https://github.com/microsoft/voxelpose-pytorch#data-preparation), we use the [CMU Panoptic](http://domedb.perception.cs.cmu.edu/), [Shelf](https://campar.in.tum.de/Chair/MultiHumanPose) and [Campus](https://campar.in.tum.de/Chair/MultiHumanPose) datasets in our experiments. 

1. We provide the scripts to download the datasets automatically:

```bash
bash scripts/download_panoptic.sh
bash scripts/download_shelf.sh
bash scripts/download_campus.sh
```

2. Due to incomplete annotations of the Shelf/Campus datasets, we synthesize **extra data** to provide training supervision for our 3D pose estimator on these two datasets. The pose sequences come from the Panoptic dataset. You need to download it ([Google drive](https://drive.google.com/file/d/1yGoDO73X2hsV9QAjS1Lb2TBNikERQRJT/view?usp=sharing)) and put it under the `data/` directory.

3. Download the [pretrained backbone model](https://drive.google.com/file/d/1Qlt_S5BoJcUIyXO5s6FEvVJ_ucQ1yN_L/view?usp=sharing) (ResNet-50 pretrained on COCO dataset and finetuned jointly on Panoptic dataset and MPII) for 2D heatmap estimation and place it under the `backbone/` directory.

**Note**: As for the Shelf/Campus datasets, we directly test our model using 2D pose predictions from pre-trained [Mask R-CNN](https://github.com/matterport/Mask_RCNN) on [COCO Dataset](https://cocodataset.org/). We've already included the annotations in the `data/Campus` and `data/Shelf` directory.


### Preprocess Data
To generate 2D heatmap predictions, you need to resize the RGB images in the pre-processing step. You can run the following code to preprocess the dataset. The supported argument `[DATASET_NAME]` includes `Panoptic`, `Shelf` and `Campus`.

```bash
python preprocess.py --dataset [DATASET_NAME]
```

After downloading and pre-processing data, your directory tree should be like this:

```
${Project}
|-- data
    |-- Panoptic
        |-- 16060224_haggling1
        |   |-- hdImgs
        |   |-- hdvideos
        |   |-- hdPose3d_stage1_coco19
        |   |-- calibration_160224_haggling1.json
        |-- 160226_haggling1  
        |-- ...
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


## Train & Eval

### Training
Every experiment is defined by config files. You can specify the path of the config file (e.g.`configs/panoptic/jln64.yaml`) and run the following code to start training the model. Note that we only support single-GPU training now.

```bash
python run/train.py --cfg [CONFIG_FILE]
```

#### Training on your own data
To train Faster-VoxelPose model on your own data, you need to follow the steps below:

1. Implement the code to process your own dataset under the `lib/dataset/` directory. You can refer to `lib/dataset/shelf.py` and rewrite the `_get_db` and `_get_cam` functions to take RGB images and camera params as input. 

2. Modify the config file based on `configs/shelf/jln64.yaml`. Remember to alter the `TEST_HEATMAP_SRC` attribute to `image` if no 2D predictions are given.

3. Start training the model and visualize the evaluation results.

### Evaluation
To evaluate the model, specify the path of the config file. By default, 
the `model_best.pth.tar` checkpoint under the corresponding working directory 
will be selected for evaluation. And the results will be printed on the screen.

```bash
python run/validate.py --cfg [CONFIG_FILE]
```

### Model Zoo
You can download our pre-trained checkpoint from Google Drive.

| Dataset  | MPJPE | AP25  | AP50  | AP100 | AP150 | Model weight                                                 | Config                               |
| -------- | ----- | ----- | ----- | ----- | ----- | ------------------------------------------------------------ | ------------------------------------ |
| Panoptic | 18.41 | 86.66 | 98.08 | 99.26 | 99.53 | [Google drive](https://drive.google.com/file/d/1ETtd2isq11oiqqkEPfCQaOSmtU0MO9kH/view?usp=sharing) | [cfg](./configs/panoptic/jln64.yaml) |


| Dataset | PCP3D | Model weight                                                 | Config                             |
| ------- | ----- | ------------------------------------------------------------ | ---------------------------------- |
| Shelf   | 97.6  | [Google drive](https://drive.google.com/file/d/1K550SE_x-0GvQuW-OUdAA-FWLlFJ0d72/view?usp=sharing) | [cfg](./configs/shelf/jln64.yaml)  |
| Campus  | 96.9  | [Google drive](https://drive.google.com/file/d/1unws01RK8Dq9dg3B8AKapPp2SJMUAM09/view?usp=sharing) | [cfg](./configs/campus/jln64.yaml) |

**Important Note**: Our implementation is slightly different from the one proposed in the original paper. Through lots of experiments, considering the speed-performance tradeoffs, we remove the **offset** branch in HDN and retrain the models. We'll modify the paper and upload the final version on arXiv.

## Visualization
We also provide a demo demonstrating how to visualize results on your own sequences. Please refer to the [ipynb](./demo/visualize.ipynb) file.

## Citation
If you use our code or models in your research, please cite with:
```bibtex
@inproceedings{fastervoxelpose,
    author={Ye, Hang and Zhu, Wentao and Wang, Chunyu and Wu, Rujie and Wang, Yizhou},
    title={Faster VoxelPose: Real-time 3D Human Pose Estimation by Orthographic Projection},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2022}
}
```

## Acknowledgment
This repo is built on the excellent work [VoxelPose](https://github.com/microsoft/voxelpose-pytorch). Thank the authors for releasing their codes.