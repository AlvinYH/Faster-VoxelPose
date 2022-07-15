# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cnns_2d import P2PNet
from models.weight_net import WeightNet
from models.project_individual import ProjectLayer

class SoftArgmaxLayer(nn.Module):
    def __init__(self, cfg):
        super(SoftArgmaxLayer, self).__init__()
        self.beta = cfg.NETWORK.BETA

    def forward(self, x, grids):
        batch_size = x.size(1)
        channel = x.size(2)
        x = x.reshape(3, batch_size, channel, -1, 1)
        x = F.softmax(self.beta * x, dim=3)
        grids = grids.reshape(3, 1, 1, -1, 2) 
        x = torch.mul(x, grids)
        x = torch.sum(x, dim=3)
        return x

class JointLocalizationNet(nn.Module):
    def __init__(self, cfg):
        super(JointLocalizationNet, self).__init__()
        self.voxels_per_axis = cfg.INDIVIDUAL_SPEC.VOXELS_PER_AXIS
        self.whole_space_center = torch.tensor(cfg.CAPTURE_SPEC.SPACE_CENTER)
        self.conv_net = P2PNet(cfg.DATASET.NUM_JOINTS, cfg.DATASET.NUM_JOINTS)
        self.weight_net = WeightNet(cfg)
        self.project_layer = ProjectLayer(cfg)
        self.soft_argmax_layer = SoftArgmaxLayer(cfg)

    def fuse_pose_preds(self, pose_preds, weights):
        weights = torch.chunk(weights, 3)
        xy_weight, xz_weight, yz_weight = weights[0], weights[1], weights[2]
        xy_pred, xz_pred, yz_pred = pose_preds[0], pose_preds[1], pose_preds[2]

        # normalize
        x_weight = torch.cat([xy_weight, xz_weight], dim=2)
        y_weight = torch.cat([xy_weight, yz_weight], dim=2)
        z_weight = torch.cat([xz_weight, yz_weight], dim=2)
        x_weight = x_weight / torch.sum(x_weight, dim=2).unsqueeze(2)
        y_weight = y_weight / torch.sum(y_weight, dim=2).unsqueeze(2)
        z_weight = z_weight / torch.sum(z_weight, dim=2).unsqueeze(2)
        
        x_pred = x_weight[:, :, :1] * xy_pred[:, :, :1] + x_weight[:, :, 1:] * xz_pred[:, :, :1]
        y_pred = y_weight[:, :, :1] * xy_pred[:, :, 1:] + y_weight[:, :, 1:] * yz_pred[:, :, :1]
        z_pred = z_weight[:, :, :1] * xz_pred[:, :, 1:] + z_weight[:, :, 1:] * yz_pred[:, :, 1:]
        
        pred = torch.cat([x_pred, y_pred, z_pred], dim=2)
        return pred

    def forward(self, meta, heatmaps, proposal_centers, mask):
        device = heatmaps[0].device
        batch_size = proposal_centers.shape[0]
        max_proposals = proposal_centers.shape[1]
        num_joints = heatmaps[0].shape[1]
        all_fused_pose_preds = torch.zeros((batch_size, max_proposals, num_joints, 3), device=device)
        all_pose_preds = torch.zeros((3, batch_size, max_proposals, num_joints, 2), device=device)

        for i in range(batch_size):
            if torch.sum(mask[i]) == 0:
                continue
            
            # construct person-specific feature cubes
            cubes, grids = self.project_layer(heatmaps, i, meta, self.voxels_per_axis, proposal_centers[i, mask[i]])
            
            # project to orthogonal planes and extract joint features
            input = torch.cat([torch.max(cubes, dim=4)[0], torch.max(cubes, dim=3)[0], 
                               torch.max(cubes, dim=2)[0]])
            joint_features = torch.stack(torch.chunk(self.conv_net(input), 3), dim=0)
            
            pose_preds = self.soft_argmax_layer(joint_features, grids)

            # add offset relative to the space center
            if self.whole_space_center.device != device:
                self.whole_space_center = self.whole_space_center.to(device)
            offset = (proposal_centers[i, mask[i], :3] - self.whole_space_center).unsqueeze(1)

            pose_preds[0] += offset[:, :, :2]
            pose_preds[1] += offset[:, :, ::2]
            pose_preds[2] += offset[:, :, 1:]

            # compute fusion weight and obtain final prediction
            weights = self.weight_net(joint_features)
            fused_pose_preds = self.fuse_pose_preds(pose_preds, weights)
            all_fused_pose_preds[i, mask[i]] = fused_pose_preds
            all_pose_preds[:, i, mask[i]] = pose_preds
            
        return all_fused_pose_preds, all_pose_preds