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

        self.whole_space_center = np.array(cfg.CAPTURE_SPEC.SPACE_CENTER)
        self.whole_space_size = np.array(cfg.CAPTURE_SPEC.SPACE_SIZE)
        self.space_size = np.array(cfg.INDIVIDUAL_SPEC.SPACE_SIZE)
        self.voxels_per_axis = np.array(cfg.INDIVIDUAL_SPEC.VOXELS_PER_AXIS)
        self.fine_voxels_per_axis = (self.whole_space_size / self.space_size * (self.voxels_per_axis - 1)).astype(int) + 1
                              
        self.sample_grid = {}
        self.grids = None

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
        
    def compute_sample_grid(self, heatmaps, meta, i, cube_size, key):
        device = heatmaps[0].device
        nbins = cube_size[0] * cube_size[1] * cube_size[2]
        n = len(heatmaps)
        w, h = self.heatmap_size
        # compute the sample grid
        grid = self.compute_grid(self.whole_space_size, self.whole_space_center, cube_size, device=device)
        sample_grids = torch.zeros(n, 1, 1, nbins, 2, device=device)
        for c in range(n):
            sample_grid = self.project_grid(meta, i, grid, w, h, c, nbins, device)
            sample_grids[c] = sample_grid
        self.sample_grid[key] = sample_grids.view(n, cube_size[0], cube_size[1], cube_size[2], 2)

    def compute_voxel_features(self, heatmaps, index, meta, voxels_per_axis, proposal_centers):
        device = heatmaps[0].device
        num_people = proposal_centers.shape[0]
        num_joints = heatmaps[0].shape[1]
        n = len(heatmaps)
        cubes = torch.zeros(num_people, num_joints, voxels_per_axis[0], voxels_per_axis[1], voxels_per_axis[2], device=device)

        # only need to compute a small grid located at the space center
        if self.grids is None:
            print("compute the grid only once")
            grid = self.compute_grid(self.space_size, self.whole_space_center, self.voxels_per_axis, device=device)
            grid = grid.view(self.voxels_per_axis[0], self.voxels_per_axis[1], self.voxels_per_axis[2], 3)
            self.grids = torch.stack([grid[:, :, 0, :2].reshape(-1, 2), grid[:, 0, :, ::2].reshape(-1, 2), \
                                      grid[0, :, :, 1:].reshape(-1, 2)])

        curr_key = meta[0]['key'][index]
        if curr_key not in self.sample_grid:
            print("add new key", curr_key)
            self.compute_sample_grid(heatmaps, meta, index, self.fine_voxels_per_axis, curr_key)

        # mask the feature volume outside the bounding box
        proposal_centers_np = proposal_centers.cpu().numpy()
        mask_x = ((1 - proposal_centers_np[:, 5]) / 2 * (self.voxels_per_axis[0] - 1)).astype(int)
        mask_y = ((1 - proposal_centers_np[:, 6]) / 2 * (self.voxels_per_axis[1] - 1)).astype(int)
        mask_x[mask_x < 0] = 0
        mask_y[mask_y < 0] = 0

        # compute the index of the top left point in the fine-grained cube
        centers_tl = ((proposal_centers_np[:, 0:3] + self.whole_space_size / 2.0 - self.whole_space_center - 
                       self.space_size / 2.0) / self.whole_space_size * (self.fine_voxels_per_axis - 1))
        centers_tl = centers_tl.round().astype(int)
        
        # compute the valid range (filter the outsider)
        x, y, z = centers_tl[:, 0], centers_tl[:, 1], centers_tl[:, 2]
        x_start, y_start, z_start = np.maximum(-x, 0) + mask_x, np.maximum(-y, 0) + mask_y, np.maximum(-z, 0)    
        x_end = np.minimum(int(self.fine_voxels_per_axis[0]) - x, int(self.voxels_per_axis[0])) - mask_x
        y_end = np.minimum(int(self.fine_voxels_per_axis[1]) - y, int(self.voxels_per_axis[1])) - mask_y
        z_end = np.minimum(int(self.fine_voxels_per_axis[2]) - z, int(self.voxels_per_axis[2]))

        # construct the feature volume
        for i in range(num_people):
            if x_start[i] >= x_end[i] or y_start[i] >= y_end[i]:
                continue
            sample_grid = self.sample_grid[curr_key]
            sample_grid = sample_grid[:, x_start[i]+x[i]:x_end[i]+x[i], y_start[i]+y[i]:y_end[i]+y[i], z_start[i]+z[i]:z_end[i]+z[i]].reshape(n, 1, -1, 2)

            accu_cubes = torch.mean(F.grid_sample(heatmaps[:, index], sample_grid, align_corners=True), dim=0).view(num_joints, x_end[i]-x_start[i], y_end[i]-y_start[i], z_end[i]-z_start[i])
            cubes[i, :, x_start[i]:x_end[i], y_start[i]:y_end[i], z_start[i]:z_end[i]] = accu_cubes
                
            del sample_grid
        cubes = cubes.clamp(0.0, 1.0)

        return cubes

    def forward(self, heatmaps, index, meta, voxels_per_axis, proposal_centers):
        cubes = self.compute_voxel_features(heatmaps, index, meta, voxels_per_axis, proposal_centers)
        return cubes, self.grids