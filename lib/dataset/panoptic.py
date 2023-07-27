# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os.path as osp
import numpy as np
import json_tricks as json
import pickle
import logging
import cv2
import copy

from dataset.JointsDataset import JointsDataset

logger = logging.getLogger(__name__)

TRAIN_LIST = [
    '160422_ultimatum1',
    '160224_haggling1',
    '160226_haggling1',
    '161202_haggling1',
    '160906_ian1',
    '160906_ian2',
    '160906_ian3',
    '160906_band1',
    '160906_band2'
    # '160906_band3',
]
VAL_LIST = [
    '160906_pizza1',
    '160422_haggling1',
    '160906_ian5',
    '160906_band4',
]

panoptic_joints_def = {
    'neck': 0,
    'nose': 1,
    'mid-hip': 2,
    'l-shoulder': 3,
    'l-elbow': 4,
    'l-wrist': 5,
    'l-hip': 6,
    'l-knee': 7,
    'l-ankle': 8,
    'r-shoulder': 9,
    'r-elbow': 10,
    'r-wrist': 11,
    'r-hip': 12,
    'r-knee': 13,
    'r-ankle': 14,
}

panoptic_bones_def = [
    [0, 1], [0, 2],  # trunk
    [0, 3], [3, 4], [4, 5],  # left arm
    [0, 9], [9, 10], [10, 11],  # right arm
    [2, 6], [6, 7], [7, 8],  # left leg
    [2, 12], [12, 13], [13, 14],  # right leg
]


class Panoptic(JointsDataset):
    def __init__(self, cfg, is_train=True, transform=None):
        super().__init__(cfg, is_train, transform)
        
        self.num_joints = len(panoptic_joints_def)
        self.num_views = cfg.DATASET.CAMERA_NUM
        self.root_id = cfg.DATASET.ROOT_JOINT_ID

        self.has_evaluate_function = True
        self.transform = transform
        self.cam_list = [(0, 3), (0, 6), (0, 12), (0, 13), (0, 23)][:self.num_views]

        if is_train:
            self.image_set = 'train'
            self.sequence_list = TRAIN_LIST
            self._interval = 3
        else:
            self.image_set = 'validation'
            self.sequence_list = VAL_LIST
            self._interval = 12

        self.cameras = self._get_cam()
        self.db_file = '{}_meta.pkl'.format(self.image_set, self.num_views)
        self.db_file = osp.join(self.dataset_dir, self.db_file)

        if osp.exists(self.db_file):
            info = pickle.load(open(self.db_file, 'rb'))
            assert info['sequence_list'] == self.sequence_list
            assert info['interval'] == self._interval
            self.db = info['db']
        else:
            self._get_db()
            info = {
                'sequence_list': self.sequence_list,
                'interval': self._interval,
                'db': self.db
            }
            pickle.dump(info, open(self.db_file, 'wb'))
        self.db_size = len(self.db)
    
    def _get_db(self):
        for seq in self.sequence_list:
            curr_anno = osp.join(self.dataset_dir, seq, 'hdPose3d_stage1_coco19')
            anno_files = sorted(glob.iglob('{:s}/*.json'.format(curr_anno)))

            # save all image paths and 3d gt joints
            for i, anno_file in enumerate(anno_files):
                if i % self._interval == 0:
                    with open(anno_file, "r") as f:
                        bodies = json.load(f)["bodies"]
                    if len(bodies) == 0:
                        continue

                    all_image_path = []
                    missing_image = False
                    for k in range(self.num_views):
                        suffix = osp.basename(anno_file).replace("body3DScene", "")
                        prefix = "{:02d}_{:02d}".format(self.cam_list[k][0], self.cam_list[k][1])
                        image_path = osp.join(self.dataset_dir, seq, "hdImgs", prefix, prefix + suffix)
                        image_path = image_path.replace("json", "jpg")
                        if not osp.exists(image_path):
                            logger.info("Image not found: {}. Skipped.".format(image_path))
                            missing_image = True
                            break
                        all_image_path.append(image_path)

                    if missing_image:
                        continue

                    all_poses_3d = []
                    all_poses_3d_vis = []
                    for body in bodies:
                        pose3d = np.array(body['joints19']).reshape((-1, 4))
                        pose3d = pose3d[:self.num_joints]

                        joints_vis = pose3d[:, -1]
                        joints_vis = np.maximum(joints_vis, 0.0)
                        # ignore the joints with visibility less than 0.1
                        if joints_vis[self.root_id] <= 0.1:
                            continue

                        # coordinate transformation
                        M = np.array([[1.0, 0.0, 0.0],
                                      [0.0, 0.0, -1.0],
                                      [0.0, 1.0, 0.0]])
                        pose3d[:, 0:3] = pose3d[:, 0:3].dot(M)

                        all_poses_3d.append(pose3d[:, 0:3] * 10.0)
                        all_poses_3d_vis.append(joints_vis)

                    if len(all_poses_3d) > 0:
                        self.db.append({
                            'seq': seq,
                            'all_image_path': all_image_path,
                            'joints_3d': all_poses_3d,
                            'joints_3d_vis': all_poses_3d_vis,
                        })
            
        super()._rebuild_db()
        logger.info("=> {} images from {} views loaded".format(len(self.db), self.num_views))
        return

    def _get_cam(self):
        cameras = dict()
        M = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [0.0, 1.0, 0.0]])

        for seq in self.sequence_list:
            cameras[seq] = []

            cam_file = osp.join(self.dataset_dir, seq, "calibration_{:s}.json".format(seq))
            with open(cam_file, "r") as f:
                calib = json.load(f)

            for cam in calib["cameras"]:
                if (cam['panel'], cam['node']) in self.cam_list:
                    sel_cam = {}
                    sel_cam['K'] = np.array(cam['K'])
                    sel_cam['distCoef'] = np.array(cam['distCoef'])
                    sel_cam['R'] = np.array(cam['R']).dot(M)
                    sel_cam['t'] = np.array(cam['t']).reshape((3, 1))
                    cameras[seq].append(sel_cam)
            
            # convert the format of camera parameters
            for k, v in enumerate(cameras[seq]):
                our_cam = dict()
                our_cam['R'] = v['R']
                our_cam['T'] = -np.dot(v['R'].T, v['t']) * 10.0  # the order to handle rotation and translation is reversed
                our_cam['fx'] = v['K'][0, 0]
                our_cam['fy'] = v['K'][1, 1]
                our_cam['cx'] = v['K'][0, 2]
                our_cam['cy'] = v['K'][1, 2]
                our_cam['k'] = v['distCoef'][[0, 1, 4]].reshape(3, 1)
                our_cam['p'] = v['distCoef'][[2, 3]].reshape(2, 1)
                cameras[seq][k] = our_cam
        return cameras

    def __getitem__(self, idx):
        input, target, meta, input_heatmap = super().__getitem__(idx)
        return input, target, meta, input_heatmap

    def __len__(self):
        return self.db_size

    def evaluate(self, preds):
        eval_list = []
        gt_num = self.db_size
        assert len(preds) == gt_num, 'number mismatch'

        total_gt = 0
        for i in range(len(preds)):
            db_rec = copy.deepcopy(self.db[i])
            num_person = db_rec['meta']['num_person']
            joints_3d = db_rec['meta']['joints_3d'][:num_person]
            joints_3d_vis = db_rec['meta']['joints_3d_vis'][:num_person]

            if num_person == 0:
                continue

            pred = preds[i].detach().cpu().numpy()
            pred = pred[pred[:, 0, 3] >= 0]
            for pose in pred:
                mpjpes = []
                for (gt, gt_vis) in zip(joints_3d, joints_3d_vis):
                    vis = gt_vis > 0.1
                    mpjpe = np.mean(np.sqrt(np.sum((pose[vis, 0:3] - gt[vis]) ** 2, axis=-1)))
                    mpjpes.append(mpjpe)
                min_gt = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                score = pose[0, 4]
                eval_list.append({
                    "mpjpe": float(min_mpjpe),
                    "score": float(score),
                    "gt_id": int(total_gt + min_gt)
                })
            total_gt += len(joints_3d)

        mpjpe_threshold = np.arange(25, 155, 25)
        aps = []
        recs = []
        for t in mpjpe_threshold:
            ap, rec = self._eval_list_to_ap(eval_list, total_gt, t)
            aps.append(ap)
            recs.append(rec)

        mpjpe = self._eval_list_to_mpjpe(eval_list)
        recall = self._eval_list_to_recall(eval_list, total_gt)
        msg = 'Evaluation results on Panoptic dataset:\nap@25: {aps_25:.4f}\tap@50: {aps_50:.4f}\tap@75: {aps_75:.4f}\t' \
              'ap@100: {aps_100:.4f}\tap@125: {aps_125:.4f}\tap@150: {aps_150:.4f}\t' \
              'recall@500mm: {recall:.4f}\tmpjpe@500mm: {mpjpe:.3f}'.format(
                aps_25=aps[0], aps_50=aps[1], aps_75=aps[2], aps_100=aps[3],
                aps_125=aps[4], aps_150=aps[5], recall=recall, mpjpe=mpjpe
              )
        metric = np.mean(aps)

        return metric, msg

    @staticmethod
    def _eval_list_to_ap(eval_list, total_gt, threshold):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        total_num = len(eval_list)

        tp = np.zeros(total_num)
        fp = np.zeros(total_num)
        gt_det = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                tp[i] = 1
                gt_det.append(item["gt_id"])
            else:
                fp[i] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / (total_gt + 1e-5)
        precise = tp / (tp + fp + 1e-5)
        for n in range(total_num - 2, -1, -1):
            precise[n] = max(precise[n], precise[n + 1])

        precise = np.concatenate(([0], precise, [0]))
        recall = np.concatenate(([0], recall, [1]))
        index = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])

        return ap, recall[-2]

    @staticmethod
    def _eval_list_to_mpjpe(eval_list, threshold=500):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        gt_det = []

        mpjpes = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                mpjpes.append(item["mpjpe"])
                gt_det.append(item["gt_id"])

        return np.mean(mpjpes) if len(mpjpes) > 0 else np.inf

    @staticmethod
    def _eval_list_to_recall(eval_list, total_gt, threshold=500):
        gt_ids = [e["gt_id"] for e in eval_list if e["mpjpe"] < threshold]

        return len(np.unique(gt_ids)) / total_gt