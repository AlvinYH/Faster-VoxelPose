# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import resnet
from models.human_detection_net import HumanDetectionNet
from models.joint_localization_net import JointLocalizationNet

class FasterVoxelPoseNet(nn.Module):
    def __init__(self, cfg):
        super(FasterVoxelPoseNet, self).__init__()
        self.max_people = cfg.CAPTURE_SPEC.MAX_PEOPLE
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.device = torch.device(cfg.DEVICE)
       
        self.pose_net = HumanDetectionNet(cfg)
        self.joint_net = JointLocalizationNet(cfg)

        self.lambda_loss_2d = cfg.TRAIN.LAMBDA_LOSS_2D
        self.lambda_loss_1d = cfg.TRAIN.LAMBDA_LOSS_1D
        self.lambda_loss_bbox = cfg.TRAIN.LAMBDA_LOSS_BBOX
        self.lambda_loss_fused = cfg.TRAIN.LAMBDA_LOSS_FUSED


    def forward(self, backbone=None, views=None, meta=None, targets=None, input_heatmaps=None, cameras=None, resize_transform=None):
        # generate input heatmaps given RGB images with the backbone
        if views is not None:
            num_views = views.shape[1]
            input_heatmaps = torch.stack([backbone(views[:, c]) for c in range(num_views)], dim=1)
        
        batch_size = input_heatmaps.shape[0]
 
        # human detection network
        proposal_heatmaps_2d, proposal_heatmaps_1d, proposal_centers, \
                              bbox_preds = self.pose_net(input_heatmaps, meta, cameras, resize_transform)
        mask = (proposal_centers[:, :, 3] >= 0)

        # joint localization network
        fused_poses, plane_poses = self.joint_net(meta, input_heatmaps, proposal_centers.detach(), mask, cameras, resize_transform)

        # compute the training loss
        if self.training:
            assert targets is not None, 'proposal ground truth not set'
            proposal2gt = proposal_centers[:, :, 3]
            proposal2gt = torch.where(proposal2gt >= 0, proposal2gt, torch.zeros_like(proposal2gt))

            # compute 2d loss of proposal heatmaps
            loss_2d = self.lambda_loss_2d * F.mse_loss(proposal_heatmaps_2d[:, 0], targets['2d_heatmaps'], reduction='mean')
            
            # unravel the 1d gt heatmaps and compute 1d loss
            matched_heatmaps_1d = torch.gather(targets['1d_heatmaps'], dim=1, index=proposal2gt.long()\
                                               .unsqueeze(2).repeat(1, 1, proposal_heatmaps_1d.shape[2]))
            loss_1d = self.lambda_loss_1d * F.mse_loss(proposal_heatmaps_1d[mask], matched_heatmaps_1d[mask], reduction='mean')
            
            # compute the loss of bbox regression, only apply supervision on gt positions
            bbox_preds = torch.gather(bbox_preds, 1, targets['index'].long().view(batch_size, -1, 1).repeat(1, 1, 2))
            loss_bbox = self.lambda_loss_bbox * F.l1_loss(bbox_preds[targets['mask']], targets['bbox'][targets['mask']], reduction='mean')

            del proposal_heatmaps_2d, proposal_heatmaps_1d, bbox_preds
            
            if torch.sum(mask) == 0:  # no valid proposals
                loss_dict = {
                    "2d_heatmaps": loss_2d,
                    "1d_heatmaps": loss_1d,
                    "bbox": loss_bbox,
                    "joint": torch.zeros(1, device=input_heatmaps.device),
                    "total": loss_2d + loss_1d + loss_bbox
                }
                return None, None, proposal_centers, input_heatmaps, loss_dict
                
            # weighted L1 loss of joint localization
            gt_joints_3d = meta['joints_3d'].float().to(self.device)
            gt_joints_3d_vis = meta['joints_3d_vis'].float().to(self.device)
            joints_3d = torch.gather(gt_joints_3d, dim=1, index=proposal2gt.long().view\
                                     (batch_size, -1, 1, 1).repeat(1, 1, self.num_joints, 3))[mask]
            joints_vis = torch.gather(gt_joints_3d_vis, dim=1, index=proposal2gt.long().view\
                                     (batch_size, -1, 1).repeat(1, 1, self.num_joints))[mask].unsqueeze(2)
            loss_joint = F.l1_loss(plane_poses[0][mask] * joints_vis, joints_3d[:, :, :2] * joints_vis, reduction="mean") +\
                         F.l1_loss(plane_poses[1][mask] * joints_vis, joints_3d[:, :, ::2] * joints_vis, reduction="mean") +\
                         F.l1_loss(plane_poses[2][mask] * joints_vis, joints_3d[:, :, 1:] * joints_vis, reduction="mean") +\
                         self.lambda_loss_fused * F.l1_loss(fused_poses[mask] * joints_vis, joints_3d * joints_vis, reduction="mean")

            loss_dict = {
                "2d_heatmaps": loss_2d,
                "1d_heatmaps": loss_1d,
                "bbox": loss_bbox,
                "joint": loss_joint,
                "total": loss_2d + loss_1d + loss_bbox + loss_joint
            }
        else:
            loss_dict = None

        fused_poses = torch.cat([fused_poses, proposal_centers[:, :, 3:5].reshape(batch_size,\
                                 -1, 1, 2).repeat(1, 1, self.num_joints, 1)], dim=3)

        return fused_poses, plane_poses, proposal_centers, input_heatmaps, loss_dict


def get(cfg):
    model = FasterVoxelPoseNet(cfg)
    return model