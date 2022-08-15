# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.cameras as cameras
from utils.transforms import affine_transform_pts_cuda as do_transform


class ProjectLayer(nn.Module):
    def __init__(self, cfg):
        super(ProjectLayer, self).__init__()
        self.image_size = cfg.NETWORK.IMAGE_SIZE
        self.heatmap_size = cfg.NETWORK.HEATMAP_SIZE
        self.ori_image_width = cfg.DATASET.ORI_IMAGE_WIDTH
        self.ori_image_height = cfg.DATASET.ORI_IMAGE_HEIGHT

        self.space_size = cfg.CAPTURE_SPEC.SPACE_SIZE
        self.space_center = cfg.CAPTURE_SPEC.SPACE_CENTER
        self.voxels_per_axis = cfg.CAPTURE_SPEC.VOXELS_PER_AXIS

        self.grid = None
        self.sample_grid = {}

    def compute_grid(self, boxSize, boxCenter, nBins, device=None):
        if isinstance(boxSize, int) or isinstance(boxSize, float):
            boxSize = [boxSize, boxSize, boxSize]
        if isinstance(nBins, int):
            nBins = [nBins, nBins, nBins]

        grid1Dx = torch.linspace(-boxSize[0] / 2, boxSize[0] / 2, nBins[0], device=device)
        grid1Dy = torch.linspace(-boxSize[1] / 2, boxSize[1] / 2, nBins[1], device=device)
        grid1Dz = torch.linspace(-boxSize[2] / 2, boxSize[2] / 2, nBins[2], device=device)
        gridx, gridy, gridz = torch.meshgrid(
            grid1Dx + boxCenter[0],
            grid1Dy + boxCenter[1],
            grid1Dz + boxCenter[2],
        )
        gridx = gridx.contiguous().view(-1, 1)
        gridy = gridy.contiguous().view(-1, 1)
        gridz = gridz.contiguous().view(-1, 1)
        grid = torch.cat([gridx, gridy, gridz], dim=1)
        self.grid = grid
        return

    def project_grid(self, meta, i, grid, w, h, c, nbins, device):
        trans = meta[c]['trans'][i].float()
        cam = {}
        for k, v in meta[c]['camera'].items():
            cam[k] = v[i]

        xy = cameras.project_pose(grid, cam)
        xy = torch.clamp(xy, -1.0, max(self.ori_image_width, self.ori_image_height))
        xy = do_transform(xy, trans)
        xy = xy * torch.tensor(
            [w, h], dtype=torch.float, device=device) / torch.tensor(
            self.image_size, dtype=torch.float, device=device)

        sample_grid = xy / torch.tensor(
            [w - 1, h - 1], dtype=torch.float,
            device=device) * 2.0 - 1.0
        sample_grid = torch.clamp(sample_grid.view(1, 1, nbins, 2), -1.1, 1.1)
        return sample_grid
        
    def compute_voxel_features(self, heatmaps, meta):
        device = heatmaps[0].device
        batch_size = heatmaps[0].shape[0]
        num_joints = heatmaps[0].shape[1]
        nbins = self.voxels_per_axis[0] * self.voxels_per_axis[1] * self.voxels_per_axis[2]
        n = len(heatmaps)
        cubes = torch.zeros(batch_size, num_joints, 1, nbins, device=device)
        w, h = self.heatmap_size
        
        for i in range(batch_size):
            curr_seq = meta[0]['seq'][i]
            if curr_seq not in self.sample_grid:
                print("=> Save the sampling grid in HDN for sequence", curr_seq)
                sample_grids = torch.zeros(n, 1, nbins, 2, device=device)
                for c in range(n):
                    sample_grid = self.project_grid(meta, i, self.grid, w, h, c, nbins, device)
                    sample_grids[c] = sample_grid.squeeze(0)
                self.sample_grid[curr_seq] = sample_grids

            shared_sample_grid = self.sample_grid[curr_seq]
            cubes[i] = torch.mean(F.grid_sample(heatmaps[:, i], shared_sample_grid, align_corners=True), dim=0).squeeze(0)
            
        cubes = cubes.clamp(0.0, 1.0)
        cubes = cubes.view(batch_size, num_joints, self.voxels_per_axis[0], self.voxels_per_axis[1], self.voxels_per_axis[2]) 
        return cubes

    def forward(self, heatmaps, meta):
        if self.grid is None:
            self.compute_grid(self.space_size, self.space_center, self.voxels_per_axis, device=heatmaps[0].device)

        cubes = self.compute_voxel_features(heatmaps, meta)
        return cubes