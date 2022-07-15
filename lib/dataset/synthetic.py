# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import pickle
import logging
import random
import json_tricks as json

from utils.transforms import rotate_points
from utils.cameras_cpu import project_pose
from dataset.JointsDataset import JointsDataset

logger = logging.getLogger(__name__)


class Synthetic(JointsDataset):
    def __init__(self, cfg, is_train=True, add_noise_to_heatmap=False, transform=None):
        super().__init__(cfg, is_train, True, transform)

        self.has_evaluate_function = False
        self.num_of_data = 100

        self.distort = cfg.DISTORT_IMAGE
        self.max_people = cfg.CAPTURE_SPEC.MAX_PEOPLE

        self.camera_file = cfg.SYNTHETIC.CAMERA_FILE
        self.pose_file = cfg.SYNTHETIC.POSE_FILE
        self.max_synthetic_people = cfg.SYNTHETIC.MAX_PEOPLE

        self.ori_image_width = cfg.DATASET.ORI_IMAGE_WIDTH
        self.ori_image_height = cfg.DATASET.ORI_IMAGE_HEIGHT

        self.space_x_min = cfg.CAPTURE_SPEC.SPACE_CENTER[0] - \
            cfg.CAPTURE_SPEC.SPACE_SIZE[0] / 2.0
        self.space_x_max = cfg.CAPTURE_SPEC.SPACE_CENTER[0] + \
            cfg.CAPTURE_SPEC.SPACE_SIZE[0] / 2.0
        self.space_y_min = cfg.CAPTURE_SPEC.SPACE_CENTER[1] - \
            cfg.CAPTURE_SPEC.SPACE_SIZE[1] / 2.0
        self.space_y_max = cfg.CAPTURE_SPEC.SPACE_CENTER[1] + \
            cfg.CAPTURE_SPEC.SPACE_SIZE[1] / 2.0
        self.cameras = self._get_cam()
        self.poses = self._get_pose()

        self._get_db()

    def _get_pose(self):
        pose_db_file = os.path.join(self.dataset_root, self.pose_file)
        poses = pickle.load(open(pose_db_file, "rb"))
        return poses

    def _get_cam(self):
        cam_file = osp.join(self.dataset_root, self.camera_file)
        extension = osp.splitext(cam_file)[1]

        if extension == '.json':
            with open(cam_file) as cfile:
                cameras = json.load(cfile)

        if extension == '.pkl':
            with open(cam_file, 'rb') as cfile:
                cameras = pickle.load(cfile)

        for id, cam in cameras.items():
            for k, v in cam.items():
                cameras[id][k] = np.array(v)

        cameras_int_key = {}
        for id, cam in cameras.items():
            cameras_int_key[int(id)] = cam

        return cameras_int_key

    def _get_db(self):
        db = []
        for i in range(self.num_of_data):
            bbox_list = []
            center_list = []
            nposes = np.random.choice(range(self.max_synthetic_people)) + 1
            select_poses = np.random.choice(self.poses, nposes)
            joints_3d = np.array([p['pose'] for p in select_poses])
            joints_3d_vis = np.array([p['vis'][:, -1] for p in select_poses])

            for n in range(nposes):
                points = joints_3d[n][:, :2].copy()
                if isinstance(self.root_id, int):
                    center = points[self.root_id]
                elif isinstance(self.root_id, list):
                    center = np.mean([points[j] for j in self.root_id], axis=0)
                
                rotation = np.random.uniform(-180, 180)

                loop = 0
                while loop < 100:
                    human_center = self.get_random_human_center(center_list)
                    human_xy = rotate_points(
                        points, center, rotation) - center + human_center

                    if self.isvalid(
                            human_center,
                            self.calc_bbox(human_xy, joints_3d_vis[n]),
                            bbox_list):
                        break
                    loop += 1

                if loop >= 100:
                    nposes = n
                    joints_3d = joints_3d[:n]
                    joints_3d_vis = joints_3d_vis[:n]
                    break
                else:
                    center_list.append(human_center)
                    bbox_list.append(self.calc_bbox(human_xy, joints_3d_vis[n]))
                    joints_3d[n][:, :2] = human_xy

            for k, cam in self.cameras.items():
                self.db.append({
                    'key': '0',
                    'camera': cam,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis
                })

        super()._rebuild_db()
        logger.info("=> {} synthetic images from {} views loaded".format(len(self.db), self.num_views))
        return db

    def __getitem__(self, idx):
        input, target, meta, input_heatmap = [], [], [], []
        for k in range(self.num_views):
            i, t, m, ih = super().__getitem__(self.num_views * idx + k)
            input.append(i)
            target.append(t)
            meta.append(m)
            input_heatmap.append(ih)
        return input, target, meta, input_heatmap

    def __len__(self):
        return self.num_of_data

    def evaluate(self):
        pass

    def get_random_human_center(self, center_list):
        if len(center_list) == 0 or random.random() < 0.7:
            new_center = np.array(
                [np.random.uniform(self.space_x_min, self.space_x_max), np.random.uniform(self.space_y_min, self.space_y_max)])
        else:
            xy = center_list[np.random.choice(range(len(center_list)))]
            new_center = xy + np.random.normal(500, 50, 2) * np.random.choice([1, -1], 2)

        return new_center

    def isvalid(self, new_center, bbox, bbox_list):
        # check the bbox is inside the capture space
        if bbox[0] < self.space_x_min or bbox[1] < self.space_y_min or bbox[2] > self.space_x_max or bbox[3] > self.space_y_max:
            return False

        new_center_us = new_center.reshape(1, -1)
        vis = 0
        for k, cam in self.cameras.items():
            width = self.ori_image_width
            height = self.ori_image_height
            loc_2d = project_pose(
                np.hstack((new_center_us, [[1000.0]])), cam, distort=self.distort)
            if 10 < loc_2d[0, 0] < width - 10 and 10 < loc_2d[0, 1] < height - 10:
                vis += 1

        if len(bbox_list) == 0:
            return vis >= 2

        bbox_list = np.array(bbox_list)
        x0 = np.maximum(bbox[0], bbox_list[:, 0])
        y0 = np.maximum(bbox[1], bbox_list[:, 1])
        x1 = np.minimum(bbox[2], bbox_list[:, 2])
        y1 = np.minimum(bbox[3], bbox_list[:, 3])

        intersection = np.maximum(0, (x1 - x0) * (y1 - y0))
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        area_list = (bbox_list[:, 2] - bbox_list[:, 0]) * \
            (bbox_list[:, 3] - bbox_list[:, 1])
        iou_list = intersection / (area + area_list - intersection)

        return vis >= 2 and np.max(iou_list) < 0.01

    @staticmethod
    def calc_bbox(pose, pose_vis):
        index = pose_vis > 0
        bbox = [np.min(pose[index, 0]), np.min(pose[index, 1]),
                np.max(pose[index, 0]), np.max(pose[index, 1])]

        return np.array(bbox)
