# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml

import numpy as np
from easydict import EasyDict as edict

config = edict()

config.BACKBONE = 'pose_resnet'
config.GPUS = '0,1'
config.WORKERS = 8
config.PRINT_FREQ = 100

config.OUTPUT_DIR = 'output'
config.LOG_DIR = 'log'
config.DATA_DIR = 'data'
config.MODEL = 'voxelpose'

# higherhrnet definition
config.HIGHER_HRNET = edict()
config.HIGHER_HRNET.PRETRAINED_LAYERS = ['*']
config.HIGHER_HRNET.FINAL_CONV_KERNEL = 1
config.HIGHER_HRNET.STEM_INPLANES = 64

config.HIGHER_HRNET.STAGE2 = edict()
config.HIGHER_HRNET.STAGE2.NUM_MODULES = 1
config.HIGHER_HRNET.STAGE2.NUM_BRANCHES= 2
config.HIGHER_HRNET.STAGE2.BLOCK = 'BASIC'
config.HIGHER_HRNET.STAGE2.NUM_BLOCKS = [4, 4]
config.HIGHER_HRNET.STAGE2.NUM_CHANNELS = [48, 96]
config.HIGHER_HRNET.STAGE2.FUSE_METHOD = 'SUM'

config.HIGHER_HRNET.STAGE3 = edict()
config.HIGHER_HRNET.STAGE3.NUM_MODULES = 4
config.HIGHER_HRNET.STAGE3.NUM_BRANCHES = 3
config.HIGHER_HRNET.STAGE3.BLOCK = 'BASIC'
config.HIGHER_HRNET.STAGE3.NUM_BLOCKS = [4, 4, 4]
config.HIGHER_HRNET.STAGE3.NUM_CHANNELS = [48, 96, 192]
config.HIGHER_HRNET.STAGE3.FUSE_METHOD = 'SUM'

config.HIGHER_HRNET.STAGE4 = edict()
config.HIGHER_HRNET.STAGE4.NUM_MODULES = 3
config.HIGHER_HRNET.STAGE4.NUM_BRANCHES = 4
config.HIGHER_HRNET.STAGE4.BLOCK = 'BASIC'
config.HIGHER_HRNET.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
config.HIGHER_HRNET.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
config.HIGHER_HRNET.STAGE4.FUSE_METHOD = 'SUM'

config.HIGHER_HRNET.DECONV = edict()
config.HIGHER_HRNET.DECONV.NUM_DECONVS = 1
config.HIGHER_HRNET.DECONV.NUM_CHANNELS = 32
config.HIGHER_HRNET.DECONV.KERNEL_SIZE = 4
config.HIGHER_HRNET.DECONV.NUM_BASIC_BLOCKS = 4
config.HIGHER_HRNET.DECONV.CAT_OUTPUT = True

# pose_resnet related params
config.RESNET = edict()
config.RESNET.NUM_LAYERS = 50
config.RESNET.DECONV_WITH_BIAS = False
config.RESNET.NUM_DECONV_LAYERS = 3
config.RESNET.NUM_DECONV_FILTERS = [256, 256, 256]
config.RESNET.NUM_DECONV_KERNELS = [4, 4, 4]
config.RESNET.FINAL_CONV_KERNEL = 1

# Cudnn related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# common params for NETWORK
config.NETWORK = edict()
config.NETWORK.PRETRAINED_BACKBONE = ''
config.NETWORK.HEATMAP_SIZE = np.array([80, 80])
config.NETWORK.IMAGE_SIZE = np.array([320, 320])
config.NETWORK.NUM_JOINTS = 17
config.NETWORK.SIGMA = 2
config.NETWORK.BETA = 100

# DATASET related params
config.DATASET = edict()
config.DATASET.TRAIN_DATASET = 'campus_synthetic'
config.DATASET.TRAIN_HEATMAP_SRC = 'image'
config.DATASET.TEST_DATASET = 'campus'
config.DATASET.TEST_HEATMAP_SRC = 'image'
config.DATASET.DATA_AUGMENTATION = False
config.DATASET.COLOR_RGB = False
config.DATASET.CAMERA_NUM = 3
config.DATASET.ORI_IMAGE_WIDTH = 360
config.DATASET.ORI_IMAGE_HEIGHT = 288
config.DATASET.ROOTIDX = 2
config.DATASET.ROOT = ''

# Synthetic dataset
config.SYNTHETIC = edict()
config.SYNTHETIC.CAMERA_FILE = ''
config.SYNTHETIC.POSE_FILE = ''
config.SYNTHETIC.MAX_PEOPLE = 10
config.SYNTHETIC.NUM_DATA = 1000
config.SYNTHETIC.DATA_AUGMENTATION = True

# train
config.TRAIN = edict()
config.TRAIN.ONLY_3D_MODULE = True
config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [90, 110]
config.TRAIN.LR = 0.001
config.TRAIN.OPTIMIZER = 'adam'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0001
config.TRAIN.NESTEROV = False
config.TRAIN.GAMMA1 = 0.99
config.TRAIN.GAMMA2 = 0.0
config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 140
config.TRAIN.RESUME = False
config.TRAIN.BATCH_SIZE = 8
config.TRAIN.SHUFFLE = True

# testing
config.TEST = edict()
config.TEST.BATCH_SIZE = 8
config.TEST.STATE = 'best'
config.TEST.MODEL_FILE = ''

# specification of the whole motion capture space
config.CAPTURE_SPEC = edict()
config.CAPTURE_SPEC.SPACE_SIZE = np.array([4000.0, 5200.0, 2400.0])
config.CAPTURE_SPEC.SPACE_CENTER = np.array([300.0, 300.0, 300.0])
config.CAPTURE_SPEC.VOXELS_PER_AXIS = np.array([24, 32, 16])
config.CAPTURE_SPEC.MAX_PEOPLE = 10
config.CAPTURE_SPEC.MIN_SCORE = 0.1

# specification of each individual
config.INDIVIDUAL_SPEC = edict()
config.INDIVIDUAL_SPEC.SPACE_SIZE = np.array([2000.0, 2000.0, 2000.0])
config.INDIVIDUAL_SPEC.VOXELS_PER_AXIS = np.array([64, 64, 64])


config.LIMBS_DEF = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4],
                    [3, 5], [4, 6], [5, 7], [7, 9], [6, 8],
                    [8, 10], [5, 11], [11, 13], [13, 15],
                    [6, 12], [12, 14], [14, 16], [5, 6], [11, 12]]


def _update_dict(k, v):
    if k == 'DATASET':
        if 'MEAN' in v and v['MEAN']:
            v['MEAN'] = np.array(
                [eval(x) if isinstance(x, str) else x for x in v['MEAN']])
        if 'STD' in v and v['STD']:
            v['STD'] = np.array(
                [eval(x) if isinstance(x, str) else x for x in v['STD']])
    if k == 'NETWORK':
        if 'HEATMAP_SIZE' in v:
            if isinstance(v['HEATMAP_SIZE'], int):
                v['HEATMAP_SIZE'] = np.array(
                    [v['HEATMAP_SIZE'], v['HEATMAP_SIZE']])
            else:
                v['HEATMAP_SIZE'] = np.array(v['HEATMAP_SIZE'])
        if 'IMAGE_SIZE' in v:
            if isinstance(v['IMAGE_SIZE'], int):
                v['IMAGE_SIZE'] = np.array([v['IMAGE_SIZE'], v['IMAGE_SIZE']])
            else:
                v['IMAGE_SIZE'] = np.array(v['IMAGE_SIZE'])
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def update_dir(model_dir, log_dir, data_dir):
    if model_dir:
        config.OUTPUT_DIR = model_dir

    if log_dir:
        config.LOG_DIR = log_dir

    if data_dir:
        config.DATA_DIR = data_dir

    config.DATASET.ROOT = os.path.join(config.DATA_DIR, config.DATASET.ROOT)

    config.NETWORK.PRETRAINED = os.path.join(config.DATA_DIR,
                                             config.NETWORK.PRETRAINED)


def get_model_name(cfg):
    name = '{model}_{num_layers}'.format(
        model=cfg.MODEL, num_layers=cfg.RESNET.NUM_LAYERS)
    deconv_suffix = ''.join(
        'd{}'.format(num_filters)
        for num_filters in cfg.RESNET.NUM_DECONV_FILTERS)
    full_name = '{height}x{width}_{name}_{deconv_suffix}'.format(
        height=cfg.NETWORK.IMAGE_SIZE[1],
        width=cfg.NETWORK.IMAGE_SIZE[0],
        name=name,
        deconv_suffix=deconv_suffix)

    return name, full_name


if __name__ == '__main__':
    import sys
    gen_config(sys.argv[1])
