# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim


def create_logger(cfg, cfg_name, phase='train'):
    tensorboard_log_dir = cfg.LOG_DIR
    
    # create the root output directory if not exists
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.mkdir(cfg.OUTPUT_DIR)
        print('=> creating the output directory: {}'.format(cfg.OUTPUT_DIR))

    # create the output directory for this config file
    dataset = cfg.DATASET.TEST_DATASET
    cfg_name = os.path.basename(cfg_name).split('.')[0]
    final_output_dir = os.path.join(cfg.OUTPUT_DIR, dataset, cfg_name)
    os.makedirs(final_output_dir, exist_ok=True)
    print('=> the output will be saved in: {}'.format(final_output_dir))

    # set up log file
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(phase, time_str)
    final_log_file = os.path.join(final_output_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    # create the log directory for tensorboard visualization (only for training)
    if phase == 'train':
        tensorboard_log_dir = os.path.join(cfg.LOG_DIR, dataset, (cfg_name + "_" + time_str))
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        print('=> creating the training log directory: {}'.format(tensorboard_log_dir))
    return logger, final_output_dir, tensorboard_log_dir


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )
    else:
        raise ValueError("optimizer type not supported")

    return optimizer


def load_checkpoint(model, optimizer, output_dir, filename='checkpoint.pth.tar'):
    file = os.path.join(output_dir, filename)
    if os.path.isfile(file):
        checkpoint = torch.load(file)
        start_epoch = checkpoint['epoch']
        precision = checkpoint['precision'] if 'precision' in checkpoint else 0
        model.load_state_dict(checkpoint['state_dict'])
        optimizer['pose'].load_state_dict(checkpoint['pose_optimizer'])
        optimizer['joint'].load_state_dict(checkpoint['joint_optimizer'])
        print('=> load checkpoint {} (epoch {})'.format(file, start_epoch))
        return start_epoch, model, optimizer, precision
    else:
        raise ValueError("no checkpoint found in the directory")


def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        
        # not save the parameters of the backbone
        new_states_dict = {}
        for k, v in states['state_dict'].items(): 
            if "backbone" not in k:
                new_states_dict[k] = v
        torch.save(new_states_dict, os.path.join(output_dir, 'model_best.pth.tar'))