from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import torch

from utils.vis import train_vis_all, test_vis_all

logger = logging.getLogger(__name__)


def train(config, backbone, model, optimizer, loader, epoch, output_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_2d = AverageMeter()
    losses_1d = AverageMeter()
    losses_bbox = AverageMeter()
    losses_joint = AverageMeter()

    model.train()
    if backbone is not None:
        backbone.train()

    accumulation_steps = 4
    accu_loss = 0

    end = time.time()

    # loading constants of the dataset
    cameras = loader.dataset.cameras
    resize_transform = torch.as_tensor(loader.dataset.resize_transform, dtype=torch.float, device=config.DEVICE)

    for i, (inputs, targets, meta, input_heatmaps) in enumerate(loader):
        if config.DATASET.TEST_HEATMAP_SRC == 'image':
            inputs = inputs.to(config.DEVICE)
            targets = dict((k, v.to(config.DEVICE)) for k, v in targets.items())
            fused_poses, plane_poses, proposal_centers, input_heatmaps, loss_dict = model(backbone=backbone, views=inputs, 
                                                                                          meta=meta, targets=targets,
                                                                                          cameras=cameras, 
                                                                                          resize_transform=resize_transform)
        else:
            input_heatmaps = input_heatmaps.to(config.DEVICE)
            targets = dict((k, v.to(config.DEVICE)) for k, v in targets.items())
            fused_poses, plane_poses, proposal_centers, _, loss_dict  = model(backbone=backbone, meta=meta, 
                                                                              input_heatmaps=input_heatmaps, 
                                                                              targets=targets, cameras=cameras, 
                                                                              resize_transform=resize_transform)
                
        loss = loss_dict["total"]
        loss_2d = loss_dict["2d_heatmaps"]
        loss_1d = loss_dict["1d_heatmaps"]
        loss_bbox = loss_dict["bbox"]
        loss_joint = loss_dict["joint"]

        losses.update(loss.item())
        losses_2d.update(loss_2d.item())
        losses_1d.update(loss_1d.item())
        losses_bbox.update(loss_bbox.item())
        losses_joint.update(loss_joint.item())
        
        if loss_joint > 0:
            optimizer['joint'].zero_grad()
            loss_joint.backward()
            optimizer['joint'].step()

        if accu_loss > 0 and (i + 1) % accumulation_steps == 0:
            optimizer['pose'].zero_grad()
            accu_loss.backward()
            optimizer['pose'].step()
            accu_loss = 0.0
        else:
            accu_loss += (loss_2d + loss_1d + loss_bbox) / accumulation_steps

        del loss_joint, loss_2d, loss_1d, loss_bbox
        
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            gpu_memory_usage = torch.cuda.memory_allocated(config.DEVICE)
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed: {speed:.1f} samples/s\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss: {loss.val:.6f} ({loss.avg:.6f})\t' \
                  'Loss_2d: {loss_2d.val:.7f} ({loss_2d.avg:.7f})\t' \
                  'Loss_1d: {loss_1d.val:.7f} ({loss_1d.avg:.7f})\t' \
                  'Loss_bbox: {loss_bbox.val:.6f} ({loss_bbox.avg:.6f})\t' \
                  'Loss_joint: {loss_joint.val:.6f} ({loss_joint.avg:.6f})\t' \
                  'Memory {memory:.1f}'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    speed=len(inputs) * inputs[0].size(0) / batch_time.val,
                    data_time=data_time, loss=losses, loss_2d=losses_2d, 
                    loss_1d=losses_1d, loss_bbox=losses_bbox, 
                    loss_joint=losses_joint, memory=gpu_memory_usage)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss_2d', losses_2d.val, global_steps)
            writer.add_scalar('train_loss_1d', losses_1d.val, global_steps)
            writer.add_scalar('train_loss_bbox', losses_bbox.val, global_steps)
            writer.add_scalar('train_loss_joint', losses_joint.val, global_steps)
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1
            
            # visualization
            if config.TRAIN.VISUALIZATION:
                prefix = '{}_{:08}'.format(os.path.join(output_dir, 'train'), i)
                train_vis_all(config, meta, cameras, resize_transform, inputs, input_heatmaps, fused_poses, plane_poses, proposal_centers, prefix)


def validate(config, backbone, model, loader, output_dir, has_evaluate_function=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    model.eval()
    if backbone is not None:
        backbone.eval()

    # loading constants of the dataset
    cameras = loader.dataset.cameras
    resize_transform = torch.as_tensor(loader.dataset.resize_transform, dtype=torch.float, device=config.DEVICE)

    with torch.no_grad():
        all_fused_poses = []
        end = time.time()
        for i, (inputs, _, meta, input_heatmaps) in enumerate(loader):
            data_time.update(time.time() - end)
            if config.DATASET.TEST_HEATMAP_SRC == 'image':
                inputs = inputs.to(config.DEVICE)
                fused_poses, plane_poses, proposal_centers, input_heatmaps, _ = model(backbone=backbone, views=inputs, 
                                                                                      meta=meta, cameras=cameras, 
                                                                                      resize_transform=resize_transform)
            else:
                input_heatmaps = input_heatmaps.to(config.DEVICE)
                fused_poses, plane_poses, proposal_centers, _, _  = model(backbone=backbone, meta=meta, 
                                                                          input_heatmaps=input_heatmaps, 
                                                                          cameras=cameras, 
                                                                          resize_transform=resize_transform)
            
            all_fused_poses.append(fused_poses)

            batch_time.update(time.time() - end)
            end = time.time()
            if i % config.PRINT_FREQ == 0 or i == len(loader) - 1:
                gpu_memory_usage = torch.cuda.memory_allocated(0)
                msg = 'Test: [{0}/{1}]\t' \
                      'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed: {speed:.1f} samples/s\t' \
                      'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Memory {memory:.1f}'.format(
                        i, len(loader), batch_time=batch_time,
                        speed=len(inputs) * inputs[0].size(0) / batch_time.val,
                        data_time=data_time, memory=gpu_memory_usage)
                logger.info(msg)

                # visualization
                if config.TEST.VISUALIZATION:
                    prefix = '{}_{:08}'.format(os.path.join(output_dir, 'validation'), i)
                    test_vis_all(config, meta, cameras, resize_transform, inputs, input_heatmaps, fused_poses, plane_poses, proposal_centers, prefix)
        
        all_fused_poses = torch.cat(all_fused_poses, dim=0)
    
    if not has_evaluate_function:
        return 0.0

    metric, msg = loader.dataset.evaluate(all_fused_poses)
    logger.info(msg)
    return metric


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count