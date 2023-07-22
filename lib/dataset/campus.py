# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
import json_tricks as json
import pickle
import scipy.io as scio
import logging
import copy
import os
import cv2
from collections import OrderedDict

from dataset.JointsDataset import JointsDataset

logger = logging.getLogger(__name__)

campus_joints_def = {
    'Right-Ankle': 0,
    'Right-Knee': 1,
    'Right-Hip': 2,
    'Left-Hip': 3,
    'Left-Knee': 4,
    'Left-Ankle': 5,
    'Right-Wrist': 6,
    'Right-Elbow': 7,
    'Right-Shoulder': 8,
    'Left-Shoulder': 9,
    'Left-Elbow': 10,
    'Left-Wrist': 11,
    'Bottom-Head': 12,
    'Top-Head': 13
}

campus_bones_def = [
    [13, 12],  # head
    [12, 9], [9, 10], [10, 11],  # left arm
    [12, 8], [8, 7], [7, 6],  # right arm
    [9, 3], [8, 2],  # trunk
    [3, 4], [4, 5],  # left leg
    [2, 1], [1, 0],  # right leg
]

class Campus(JointsDataset):
    def __init__(self, cfg, is_train=True, transform=None):
        super().__init__(cfg, is_train, transform)
        self.has_evaluate_function = True
        self.frame_range = list(range(350, 471)) + list(range(650, 751))
        self.pred_pose2d = self._get_pred_pose2d()
        self.num_joints = len(campus_joints_def)
        self.cameras = self._get_cam()
        self._get_db()

    def _get_pred_pose2d(self):
        file = os.path.join(self.dataset_dir, 'pred_campus_maskrcnn_hrnet_coco.pkl')
        with open(file, "rb") as pfile:
            logging.info("=> load {}".format(file))
            pred_2d = pickle.load(pfile)

        return pred_2d

    def _get_db(self):
        datafile = os.path.join(self.dataset_dir, 'actorsGT.mat')
        actor_3d = scio.loadmat(datafile)['actor3D']
        actor_3d = np.array(np.array(actor_3d.tolist()).tolist(), dtype=object).squeeze()
        num_person = len(actor_3d)

        for i in self.frame_range:
            all_image_path, all_preds, all_poses_3d, all_poses_3d_vis = [], [], [], []
            
            for person in range(num_person):
                pose3d = actor_3d[person][i] * 1000.0
                if len(pose3d[0]) > 0:
                    all_poses_3d.append(pose3d)
                    all_poses_3d_vis.append(np.ones(self.num_joints))
            
            missing_image = False
            for k in range(self.num_views):
                image_path = osp.join(self.dataset_dir, "Camera{}".format(k), "campus4-c{}-{:05d}.png".format(k, i))
                if not osp.exists(image_path):
                    logger.info("Image not found: {}. Skipped.".format(image_path))
                    missing_image = True
                    break
                all_image_path.append(image_path)

                # save all 2D pred results
                pred_index = '{}_{}'.format(k, i)
                preds = self.pred_pose2d[pred_index]
                preds = [np.array(p["pred"]) for p in preds]
                all_preds.append(preds)

            if missing_image:
                continue
            
            self.db.append({
                'seq': 'campus',
                'all_image_path': all_image_path,
                'pred_pose2d': all_preds,
                'joints_3d': all_poses_3d,
                'joints_3d_vis': all_poses_3d_vis
            })

        super()._rebuild_db()
        logger.info("=> {} images from {} views loaded".format(len(self.db), self.num_views))
        return

    def _get_cam(self):
        cam_file = osp.join(self.dataset_dir, 'calibration_campus.json')
        with open(cam_file) as cfile:
            cameras = json.load(cfile)

        for id, cam in cameras.items():
            for k, v in cam.items():
                cameras[id][k] = np.array(v)
        
        cameras_int_key = {}
        for id, cam in cameras.items():
            cameras_int_key[int(id)] = cam

        our_cameras = dict()
        our_cameras['campus'] = cameras_int_key
        return our_cameras

    def __getitem__(self, idx):
        input, target, meta, input_heatmap = super().__getitem__(idx)
        return input, target, meta, input_heatmap

    def __len__(self):
        return len(self.db)

    def evaluate(self, preds, recall_threshold=500):
        datafile = os.path.join(self.dataset_dir, 'actorsGT.mat')
        actor_3d = scio.loadmat(datafile)['actor3D']
        actor_3d = np.array(np.array(actor_3d.tolist()).tolist(), dtype=object).squeeze()
        num_person = len(actor_3d)
        total_gt = 0
        match_gt = 0

        limbs = [[0, 1], [1, 2], [3, 4], [4, 5], [
                  6, 7], [7, 8], [9, 10], [10, 11], [12, 13]]
        correct_parts = np.zeros(num_person)
        total_parts = np.zeros(num_person)
        alpha = 0.5
        bone_correct_parts = np.zeros((num_person, 10))

        for i, fi in enumerate(self.frame_range):
            pred_coco = preds[i].detach().cpu().numpy()
            pred_coco = pred_coco[pred_coco[:, 0, 3] >= 0, :, :3]
            if(len(pred_coco) == 0):
                continue

            pred = np.stack([self.coco2campus3D(p)
                             for p in copy.deepcopy(pred_coco[:, :, :3])])

            for person in range(num_person):
                gt = actor_3d[person][fi] * 1000.0
                if len(gt[0]) == 0:
                    continue

                mpjpes = np.mean(np.sqrt(np.sum((gt[np.newaxis] - pred) ** 2, axis=-1)), axis=-1)
                min_n = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                if min_mpjpe < recall_threshold:
                    match_gt += 1
                total_gt += 1

                for j, k in enumerate(limbs):
                    total_parts[person] += 1
                    error_s = np.linalg.norm(pred[min_n, k[0], 0:3] - gt[k[0]])
                    error_e = np.linalg.norm(pred[min_n, k[1], 0:3] - gt[k[1]])
                    limb_length = np.linalg.norm(gt[k[0]] - gt[k[1]])
                    if (error_s + error_e) / 2.0 <= alpha * limb_length:
                        correct_parts[person] += 1
                        bone_correct_parts[person, j] += 1
                pred_hip = (pred[min_n, 2, 0:3] + pred[min_n, 3, 0:3]) / 2.0
                gt_hip = (gt[2] + gt[3]) / 2.0
                total_parts[person] += 1
                error_s = np.linalg.norm(pred_hip - gt_hip)
                error_e = np.linalg.norm(pred[min_n, 12, 0:3] - gt[12])
                limb_length = np.linalg.norm(gt_hip - gt[12])
                if (error_s + error_e) / 2.0 <= alpha * limb_length:
                    correct_parts[person] += 1
                    bone_correct_parts[person, 9] += 1

        actor_pcp = correct_parts / (total_parts + 1e-8)
        avg_pcp = np.mean(actor_pcp[:3])

        bone_group = OrderedDict(
            [('Head', [8]), ('Torso', [9]), ('Upper arms', [5, 6]),
             ('Lower arms', [4, 7]), ('Upper legs', [1, 2]), ('Lower legs', [0, 3])])
        bone_person_pcp = OrderedDict()
        for k, v in bone_group.items():
            bone_person_pcp[k] = np.sum(
                bone_correct_parts[:, v], axis=-1) / (total_parts / 10 * len(v) + 1e-8)
        
        recall = match_gt / (total_gt + 1e-8)
        msg = '     | Actor 1 | Actor 2 | Actor 3 | Average | \n' \
              ' PCP |  {pcp_1:.2f}  |  {pcp_2:.2f}  |  {pcp_3:.2f}  |  {pcp_avg:.2f}  |\t Recall@500mm: {recall:.4f}'.format(
                pcp_1=actor_pcp[0]*100, pcp_2=actor_pcp[1]*100, pcp_3=actor_pcp[2]*100, pcp_avg=avg_pcp*100, recall=recall)
        metric = np.mean(avg_pcp)

        return metric, msg

    @staticmethod
    def coco2campus3D(coco_pose):
        """
        transform coco order(our method output) 3d pose to shelf dataset order with interpolation
        :param coco_pose: np.array with shape 17x3
        :return: 3D pose in campus order with shape 14x3
        """
        campus_pose = np.zeros((14, 3))
        coco2campus = np.array([16, 14, 12, 11, 13, 15, 10, 8, 6, 5, 7, 9])
        campus_pose[0: 12] += coco_pose[coco2campus]

        mid_sho = (coco_pose[5] + coco_pose[6]) / 2  # L and R shoulder
        head_center = (coco_pose[3] + coco_pose[4]) / 2  # middle of two ear

        head_bottom = (mid_sho + head_center) / 2  # nose and head center
        head_top = head_bottom + (head_center - head_bottom) * 2
        campus_pose[12] += head_bottom
        campus_pose[13] += head_top

        return campus_pose
