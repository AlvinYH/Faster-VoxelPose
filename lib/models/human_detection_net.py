# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn

from models.cnns_2d import CenterNet
from models.cnns_1d import C2CNet
from models.project_whole import ProjectLayer
from core.proposal import nms2D

class ProposalLayer(nn.Module):
    def __init__(self, cfg):
        super(ProposalLayer, self).__init__()
        self.max_people = cfg.CAPTURE_SPEC.MAX_PEOPLE
        self.min_score = cfg.CAPTURE_SPEC.MIN_SCORE
        self.device = torch.device(cfg.DEVICE)

        # constants for getting real coordinates
        self.scale = (torch.tensor(cfg.CAPTURE_SPEC.SPACE_SIZE) / (torch.tensor(cfg.CAPTURE_SPEC.VOXELS_PER_AXIS) - 1)).to(self.device) 
        self.bias = (torch.tensor(cfg.CAPTURE_SPEC.SPACE_CENTER) - torch.tensor(cfg.CAPTURE_SPEC.SPACE_SIZE) / 2.0).to(self.device)

    def filter_proposal(self, topk_index, bbox_preds, gt_3d, gt_bbox, num_person):
        batch_size = topk_index.shape[0]
        proposal2gt = torch.zeros(batch_size, self.max_people, device=topk_index.device)
        
        for i in range(batch_size):
            proposals = topk_index[i].reshape(self.max_people, 1, -1)
            gt = gt_3d[i, :num_person[i]].reshape(1, num_person[i], -1)
            dist = torch.sqrt(torch.sum((proposals - gt)**2, dim=-1))
            min_dist, min_gt = torch.min(dist, dim=-1)
            proposal2gt[i] = min_gt
            proposal2gt[i][min_dist > 500.0] = -1.0
            for k in range(self.max_people):
                threshold = 0.1
                if proposal2gt[i, k] < 0:
                    continue
                if torch.sum(bbox_preds[i, k] < gt_bbox[i, proposal2gt[i, k].long()] - threshold):
                    bbox_preds[i, k] = gt_bbox[i, proposal2gt[i, k].long()]
        return proposal2gt

    def forward(self, topk_index, topk_confs, match_bbox_preds, meta):
        device = topk_index.device
        batch_size = topk_index.shape[0]

        # convert into real coordinates for filtering
        topk_index = topk_index.float() * self.scale + self.bias

        proposal_centers = torch.zeros(batch_size, self.max_people, 7, device=device)
        proposal_centers[:, :, 0:3] = topk_index
        proposal_centers[:, :, 4] = topk_confs

        # match to gt for training
        if self.training and ('roots_3d' in meta and 'num_person' in meta):
            gt_3d = meta['roots_3d'].float().to(self.device)
            gt_bbox = meta['bbox'].float().to(self.device)
            num_person = meta['num_person']
            proposal2gt = self.filter_proposal(topk_index, match_bbox_preds, gt_3d, gt_bbox, num_person)
            proposal_centers[:, :, 3] = proposal2gt
        else:
            proposal_centers[:, :, 3] = (topk_confs > self.min_score).float() - 1.0  # if ground-truths are not available.
        proposal_centers[: ,:, 5:7] = match_bbox_preds
        return proposal_centers

class HumanDetectionNet(nn.Module):
    def __init__(self, cfg):
        super(HumanDetectionNet, self).__init__()
        self.max_people = cfg.CAPTURE_SPEC.MAX_PEOPLE
        self.project_layer = ProjectLayer(cfg)
        self.center_net = CenterNet(cfg.DATASET.NUM_JOINTS, 1)
        self.c2c_net = C2CNet(cfg.DATASET.NUM_JOINTS, 1)
        self.proposal_layer = ProposalLayer(cfg)
        
    def forward(self, heatmaps, meta, cameras, resize_transform):
        batch_size = heatmaps.shape[0]
        num_joints = heatmaps.shape[2]
        
        # construct feature cubes
        feature_cubes = self.project_layer(heatmaps, meta, cameras, resize_transform)                                         
        
        # generate 2d proposals
        proposal_heatmaps_2d, bbox_preds = self.center_net(feature_cubes)
        topk_2d_confs, topk_2d_index, topk_2d_flatten_index = nms2D(proposal_heatmaps_2d.detach(), self.max_people) 
        
        # extract the matched bbox predictions
        bbox_preds = torch.flatten(bbox_preds, 2, 3).permute(0, 2, 1)
        match_bbox_preds = torch.gather(bbox_preds, dim=1, index=topk_2d_flatten_index.unsqueeze(2).repeat(1, 1, 2))
        
        # extract the matched 1d features and feed them into 1d CNN
        feature_1d = torch.gather(torch.flatten(feature_cubes, 2, 3).permute(0, 2, 1, 3), dim=1,\
                                  index=topk_2d_flatten_index.view(batch_size, -1, 1, 1).repeat(1, 1, num_joints, feature_cubes.shape[4]))
        proposal_heatmaps_1d = self.c2c_net(torch.flatten(feature_1d, 0, 1)).view(batch_size, self.max_people, -1)
        topk_1d_confs, topk_1d_index = proposal_heatmaps_1d.detach().topk(1)

        # assemble predictions
        topk_index = torch.cat([topk_2d_index, topk_1d_index], dim=2)
        
        # confidence score: product of 2d value and 1d value
        topk_confs = topk_2d_confs * topk_1d_confs.squeeze(2)
        proposal_centers = self.proposal_layer(topk_index, topk_confs, match_bbox_preds, meta)

        return proposal_heatmaps_2d, proposal_heatmaps_1d, proposal_centers, bbox_preds