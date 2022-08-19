# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from core.config import config, update_config
from core.function import train_3d, validate_3d
from utils.utils import create_logger, save_checkpoint, load_checkpoint, load_backbone
import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train VoxelPose Network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)

    args, _ = parser.parse_known_args()
    return args


def get_optimizer(model):
    lr = config.TRAIN.LR
    if model.module.backbone is not None:
        for params in model.module.backbone.parameters():
            # Set to True If you want to train the whole model.
            params.requires_grad = False

    for params in model.module.pose_net.parameters():
        params.requires_grad = True
    for params in model.module.joint_net.parameters():
        params.requires_grad = True
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.module.parameters()), lr=lr)

    return model, optimizer


def get_data_loaders(config):
    ngpus = len(config.GPUS.split(','))
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = eval('dataset.' + config.DATASET.TRAIN_DATASET)(
        config, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE * ngpus,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True)

    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE * ngpus,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)
    return train_dataset, train_loader, test_dataset, test_loader


def main():
    torch.cuda.empty_cache()
    args = parse_args()
    update_config(args.cfg)

    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'train')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    gpus = [int(i) for i in config.GPUS.split(',')]
    train_dataset, train_loader, test_dataset, test_loader = get_data_loaders(config)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    torch.autograd.set_detect_anomaly(True)

    print('=> Constructing models ..')
    model = eval('models.' + config.MODEL + '.get')(config, is_train=True)

    with torch.no_grad():
        model = torch.nn.DataParallel(model.cuda(), device_ids=gpus)
        model.to(f'cuda:{model.device_ids[0]}')

    model, optimizer = get_optimizer(model)

    start_epoch = config.TRAIN.BEGIN_EPOCH
    end_epoch = config.TRAIN.END_EPOCH

    best_precision = 0
    if config.NETWORK.PRETRAINED_BACKBONE:
        model = load_backbone(model, config.NETWORK.PRETRAINED_BACKBONE)
        print('=> Loading weights for backbone')
    if config.TRAIN.RESUME:
        start_epoch, model, optimizer, best_precision = load_checkpoint(
            model, optimizer, final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    print('=> Training...')
    for epoch in range(start_epoch, end_epoch):
        print('Epoch: {}'.format(epoch))

        train_3d(config, model, optimizer, train_loader, epoch, final_output_dir, writer_dict)

        best_model = True
        if test_loader != None and test_dataset.has_evaluate_function:
            precision = validate_3d(config, model, test_loader, final_output_dir,\
                                    test_dataset.has_evaluate_function)

            if precision >= best_precision:
                best_precision = precision
                best_model = True
            else:
                best_model = False

        logger.info('=> saving checkpoint to {} (Best: {})'.format(
            final_output_dir, best_model))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict(),
            'precision': best_precision,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(final_output_dir, 'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
