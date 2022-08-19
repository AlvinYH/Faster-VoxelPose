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

class VoxelPoseNet(nn.Module):
    def __init__(self, backbone, cfg):
        super(VoxelPoseNet, self).__init__()
        self.max_people = cfg.CAPTURE_SPEC.MAX_PEOPLE
        self.num_joints = cfg.NETWORK.NUM_JOINTS
       
        self.backbone = backbone
        self.pose_net = HumanDetectionNet(cfg)
        self.joint_net = JointLocalizationNet(cfg)

    def forward(self, views=None, meta=None, targets=None, input_heatmaps=None, cameras=None, resize_transform=None):
        # views: [batch_size, num_views, num_channels, height, width]
        # input_heatmaps: [batch_size, num_views, num_joints, hm_height, hm_width]
        if views is not None:
            num_views = views.shape[1]
            input_heatmaps = torch.stack([self.backbone(views[:, c]) for c in range(num_views)], dim=1)
        batch_size = input_heatmaps.shape[0]
 
        # human detection network
        proposal_heatmaps_2d, proposal_heatmaps_1d, proposal_centers, \
                              bbox_preds = self.pose_net(input_heatmaps, meta, cameras, resize_transform)
        mask = (proposal_centers[:, :, 3] >= 0)

        # joint localization network
        fused_poses, poses = self.joint_net(meta, input_heatmaps, proposal_centers.detach(), mask, cameras, resize_transform)

        # compute the training loss
        if self.training:
            assert targets is not None, 'proposal ground truth not set'
            proposal2gt = proposal_centers[:, :, 3]
            proposal2gt = torch.where(proposal2gt >= 0, proposal2gt, torch.zeros_like(proposal2gt))

            # compute 2d loss of proposal heatmaps
            loss_2d = F.mse_loss(proposal_heatmaps_2d[:, 0], targets['2d_heatmaps'], reduction='mean')
            
            # unravel the 1d gt heatmaps and compute 1d loss
            matched_heatmaps_1d = torch.gather(targets['1d_heatmaps'], dim=1, index=proposal2gt.long()\
                                               .unsqueeze(2).repeat(1, 1, proposal_heatmaps_1d.shape[2]))
            loss_1d = F.mse_loss(proposal_heatmaps_1d[mask], matched_heatmaps_1d[mask], reduction='mean')
            
            # compute the loss of bbox regression, only apply supervision on gt positions
            bbox_preds = torch.gather(bbox_preds, 1, targets['index'].long().view(batch_size, -1, 1).repeat(1, 1, 2))
            loss_bbox = F.l1_loss(bbox_preds[targets['mask']], targets['bbox'][targets['mask']], reduction='mean')

            del proposal_heatmaps_2d, proposal_heatmaps_1d, bbox_preds
            
            # weighted L1 loss of joint localization
            joints_3d = torch.gather(meta['joints_3d'].float(), dim=1, index=proposal2gt.long().view\
                                             (batch_size, -1, 1, 1).repeat(1, 1, self.num_joints, 3))[mask]
            joints_vis = torch.gather(meta['joints_3d_vis'].float(), dim=1, index=proposal2gt.long().view\
                                             (batch_size, -1, 1).repeat(1, 1, self.num_joints))[mask].unsqueeze(2)
            loss_joint = F.l1_loss(poses[0][mask] * joints_vis, joints_3d[:, :, :2] * joints_vis, reduction="mean") +\
                         F.l1_loss(poses[1][mask] * joints_vis, joints_3d[:, :, ::2] * joints_vis, reduction="mean") +\
                         F.l1_loss(poses[2][mask] * joints_vis, joints_3d[:, :, 1:] * joints_vis, reduction="mean") +\
                         2 * F.l1_loss(fused_poses[mask] * joints_vis, joints_3d * joints_vis, reduction="mean")

            loss_dict = {
                "2d_heatmaps": loss_2d,
                "1d_heatmaps": loss_1d,
                "bbox": 0.1 * loss_bbox,
                "joint": loss_joint,
                "total": loss_2d + loss_1d + 0.1 * loss_bbox + loss_joint
            }
        else:
            loss_dict = None

        fused_poses = torch.cat([fused_poses, proposal_centers[:, :, 3:5].reshape(batch_size,\
                                 -1, 1, 2).repeat(1, 1, self.num_joints, 1)], dim=3)
        return fused_poses, poses, proposal_centers.detach(), loss_dict, input_heatmaps


def get(cfg, is_train=True):
    if cfg.BACKBONE:
        backbone = eval(cfg.BACKBONE + '.get')(cfg, is_train=is_train)
    else:
        backbone = None
    model = VoxelPoseNet(backbone, cfg)
    return model