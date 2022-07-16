# Faster VoxelPose
This is the official implementation for:
> **Faster VoxelPose: Real-time 3D Human Pose Estimation by Orthographic Projection**,            
> Hang Ye, Wentao Zhu, Chunyu Wang, Rujie Wu, and Yizhou Wang       
> *ECCV 2022*

<img src="assets/band.jpg" width="800"/>

## Environment
```bash
conda install pytorch torchvision cudatoolkit=<your cuda version>
pip install -r requirements.txt
```

## Data 
We use the Shelf/Campus and CMU Panoptic datasets. Please refer to [VoxelPose](https://github.com/microsoft/voxelpose-pytorch#data-preparation) for detailed instructions.


## Training
<img src="assets/teaser.jpg" width="800"/>
### CMU Panoptic dataset

Train and validate on the five selected camera views. You can specify the GPU devices and batch size per GPU in the config file. 
```bash
python run/train.py --cfg configs/panoptic/jln64.yaml
```
### Shelf/Campus datasets
```bash
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
```bibtex
@inproceedings{fastervoxelpose,
    author={Ye, Hang and Zhu, Wentao and Wang, Chunyu and Wu, Rujie and Wang, Yizhou},
    title={Faster VoxelPose: Real-time 3D Human Pose Estimation by Orthographic Projection},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2022}
}
```