# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class Basic2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super(Basic2DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.block(x)


class Res2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Res2DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_planes)
        )

        if in_planes == out_planes: 
            self.skip_con = nn.Sequential() 
        else:
            self.skip_con = nn.Sequential(  # adjust the dimension
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_planes)
            )
    
    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)  # skip connection
        return F.relu(res + skip, True)

class WeightNet(nn.Module):
    def __init__(self, cfg):
        super(WeightNet, self).__init__()
        self.voxels_per_axis = cfg.INDIVIDUAL_SPEC.VOXELS_PER_AXIS
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.num_channel_joint_feat = cfg.NETWORK.NUM_CHANNEL_JOINT_FEAT
        self.num_channel_joint_hidden = cfg.NETWORK.NUM_CHANNEL_JOINT_HIDDEN
        self.heatmap_feature_net = nn.Sequential(
            nn.Conv2d(1, self.num_channel_joint_feat, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_channel_joint_feat),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Sequential(
            nn.Linear(self.num_channel_joint_feat, self.num_channel_joint_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_channel_joint_hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.flatten(x, 0, 1)
        batch_size = x.shape[0]
        num_joints = self.num_joints
        x = x.view(batch_size * num_joints, 1, self.voxels_per_axis[0], self.voxels_per_axis[1])
        
        x = self.heatmap_feature_net(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(batch_size * num_joints, -1)
        x = self.output(x)
        x = x.view(batch_size, num_joints, 1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)