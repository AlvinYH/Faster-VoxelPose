# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import utils.cameras as cameras
from utils.transforms import affine_transform_pts_cuda as do_transform


class ProjectLayer(nn.Module):
    def __init__(self, cfg):
        super(ProjectLayer, self).__init__()
        self.image_size = cfg.NETWORK.IMAGE_SIZE
        self.heatmap_size = cfg.NETWORK.HEATMAP_SIZE
        self.ori_image_width = cfg.DATASET.ORI_IMAGE_WIDTH
        self.ori_image_height = cfg.DATASET.ORI_IMAGE_HEIGHT
        self.device = torch.device(int(cfg.GPUS.split(',')[0]))

        # constants for back-projection
        self.whole_space_center = torch.tensor(cfg.CAPTURE_SPEC.SPACE_CENTER, device=self.device)
        self.whole_space_size = torch.tensor(cfg.CAPTURE_SPEC.SPACE_SIZE, device=self.device)
        self.ind_space_size = torch.tensor(cfg.INDIVIDUAL_SPEC.SPACE_SIZE, device=self.device)
        self.voxels_per_axis = torch.tensor(cfg.INDIVIDUAL_SPEC.VOXELS_PER_AXIS, device=self.device, dtype=torch.int32)
        self.fine_voxels_per_axis = (self.whole_space_size / self.ind_space_size * (self.voxels_per_axis - 1)).int() + 1

        self.scale = (self.fine_voxels_per_axis.float() - 1) / (torch.tensor(cfg.CAPTURE_SPEC.VOXELS_PER_AXIS, device=self.device) - 1)
        self.bias = - self.ind_space_size / 2.0 / self.whole_space_size * (self.fine_voxels_per_axis - 1)

        self.save_grid() 
        self.sample_grid = {}

    def compute_grid(self, boxSize, boxCenter, nBins, device=None):
        if isinstance(boxSize, int) or isinstance(boxSize, float):
            boxSize = [boxSize, boxSize, boxSize]
        if isinstance(nBins, int):
            nBins = [nBins, nBins, nBins]

        grid1Dx = torch.linspace(-boxSize[0] / 2, boxSize[0] / 2, int(nBins[0]), device=device)
        grid1Dy = torch.linspace(-boxSize[1] / 2, boxSize[1] / 2, int(nBins[1]), device=device)
        grid1Dz = torch.linspace(-boxSize[2] / 2, boxSize[2] / 2, int(nBins[2]), device=device)
        gridx, gridy, gridz = torch.meshgrid(
            grid1Dx + boxCenter[0],
            grid1Dy + boxCenter[1],
            grid1Dz + boxCenter[2],
        )
        gridx = gridx.contiguous().view(-1, 1)
        gridy = gridy.contiguous().view(-1, 1)
        gridz = gridz.contiguous().view(-1, 1)
        grid = torch.cat([gridx, gridy, gridz], dim=1)
        return grid

    def save_grid(self):
        print("=> Save the 3D grid for feature sampling")
        grid = self.compute_grid(self.ind_space_size, self.whole_space_center, self.voxels_per_axis, device=self.device)
        grid = grid.view(self.voxels_per_axis[0], self.voxels_per_axis[1], self.voxels_per_axis[2], 3)
        self.center_grid = torch.stack([grid[:, :, 0, :2].reshape(-1, 2), grid[:, 0, :, ::2].reshape(-1, 2), \
                                        grid[0, :, :, 1:].reshape(-1, 2)])
        self.fine_grid = self.compute_grid(self.whole_space_size, self.whole_space_center, self.fine_voxels_per_axis, device=self.device)

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
        
    '''
    only compute the projected 2D finer grid once for each sequence
    '''
    def compute_sample_grid(self, heatmaps, meta, i, voxels_per_axis, seq):
        device = heatmaps[0].device
        nbins = voxels_per_axis[0] * voxels_per_axis[1] * voxels_per_axis[2]
        n = len(heatmaps)
        w, h = self.heatmap_size
        # compute the sample grid
        sample_grids = torch.zeros(n, 1, 1, nbins, 2, device=device)
        for c in range(n):
            sample_grid = self.project_grid(meta, i, self.fine_grid, w, h, c, nbins, device)
            sample_grids[c] = sample_grid
        self.sample_grid[seq] = sample_grids.view(n, voxels_per_axis[0], voxels_per_axis[1], voxels_per_axis[2], 2)

    def compute_voxel_features(self, heatmaps, index, meta, proposal_centers):
        device = heatmaps[0].device
        num_people = proposal_centers.shape[0]
        num_joints = heatmaps[0].shape[1]
        n = len(heatmaps)
        cubes = torch.zeros(num_people, num_joints, self.voxels_per_axis[0], self.voxels_per_axis[1], self.voxels_per_axis[2], device=device)

        curr_seq = meta[0]['seq'][index]
        if curr_seq not in self.sample_grid:
            print("=> Save the sampling grid in JLN for sequence", curr_seq)
            self.compute_sample_grid(heatmaps, meta, index, self.fine_voxels_per_axis, curr_seq)

        # compute the index of the top left point in the fine-grained volume
        # proposal centers: [batch_size, 7]
        centers_tl = torch.round(proposal_centers[:, :3].float() * self.scale + self.bias).int()
        offset = centers_tl.float() / (self.fine_voxels_per_axis - 1) * self.whole_space_size - self.whole_space_size / 2.0 + self.ind_space_size / 2.0
        
        # mask the feature volume outside the bounding box
        mask = ((1 - proposal_centers[:, 5:7]) / 2 * (self.voxels_per_axis[0:2] - 1)).int()
        mask[mask < 0] = 0
        # the vertical length of the bounding box is kept fixed as 2000mm
        mask = torch.cat([mask, torch.zeros((num_people, 1), device=device, dtype=torch.int32)], dim=1)

        # compute the valid range to filter the outsider
        start = torch.where(centers_tl + mask >= 0, centers_tl + mask, torch.zeros_like(centers_tl))
        end = torch.where(centers_tl + self.voxels_per_axis - mask <= self.fine_voxels_per_axis, centers_tl + self.voxels_per_axis - mask, self.fine_voxels_per_axis)

        # construct the feature volume
        for i in range(num_people):
            if torch.sum(start[i] >= end[i]) > 0:
                continue
            sample_grid = self.sample_grid[curr_seq]
            sample_grid = sample_grid[:, start[i, 0]:end[i, 0], start[i, 1]:end[i, 1], start[i, 2]:end[i, 2]].reshape(n, 1, -1, 2)

            accu_cubes = torch.mean(F.grid_sample(heatmaps[:, index], sample_grid, align_corners=True), dim=0).view(num_joints, end[i, 0]-start[i, 0], end[i, 1]-start[i, 1], end[i, 2]-start[i, 2])
            cubes[i, :, start[i, 0]-centers_tl[i, 0]:end[i, 0]-centers_tl[i, 0], start[i, 1]-centers_tl[i, 1]:end[i, 1]-centers_tl[i, 1], start[i, 2]-centers_tl[i, 2]:end[i, 2]-centers_tl[i, 2]] = accu_cubes
            del sample_grid
        cubes = cubes.clamp(0.0, 1.0)

        return cubes, offset

    def forward(self, heatmaps, index, meta, proposal_centers):
        cubes, offset = self.compute_voxel_features(heatmaps, index, meta, proposal_centers)
        return cubes, offset