# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import numpy as np
import torchvision
import cv2
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from utils.cameras import project_pose
from utils.transforms import affine_transform_pts_cuda as do_transform

# coco17
LIMBS17 = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [11, 13], [13, 15],
        [6, 12], [12, 14], [14, 16], [5, 6], [11, 12]]

# shelf / campus
LIMBS14 = [[0, 1], [1, 2], [3, 4], [4, 5], [2, 3], [6, 7], [7, 8], [9, 10],
          [10, 11], [2, 8], [3, 9], [8, 12], [9, 12], [12, 13]]

# panoptic
LIMBS15 = [[0, 1], [0, 2], [0, 3], [3, 4], [4, 5], [0, 9], [9, 10],
         [10, 11], [2, 6], [2, 12], [6, 7], [7, 8], [12, 13], [13, 14]]

# colors for visualization
colors = ['b', 'g', 'c', 'y', 'm', 'orange', 'pink', 'royalblue', 'lightgreen', 'gold']
idx_list = {'xy':[0, 1], 'xz':[0, 2], 'yz':[1, 2]}


def train_vis_all(config, meta, cameras, resize_transform, inputs, input_heatmaps, fused_poses, plane_poses, proposal_centers, prefix):
    if '2d_planes' in config.TRAIN.VIS_TYPE:
        save_2d_planes(config, meta, fused_poses, plane_poses, proposal_centers, prefix)
    
    if 'image_with_poses' in config.TRAIN.VIS_TYPE:
        save_image_with_poses(config, inputs, fused_poses, meta, cameras, resize_transform, prefix)
    
    if 'heatmaps' in config.TRAIN.VIS_TYPE:
        if config.DATASET.TRAIN_HEATMAP_SRC != 'image':
            raise ValueError("cannot visualize heatmaps")
        save_heatmaps(inputs, input_heatmaps, prefix)


def test_vis_all(config, meta, cameras, resize_transform, inputs, input_heatmaps, fused_poses, plane_poses, proposal_centers, prefix):
    if '2d_planes' in config.TEST.VIS_TYPE:
        save_2d_planes(config, meta, fused_poses, plane_poses, proposal_centers, prefix)
    
    if 'image_with_poses' in config.TEST.VIS_TYPE:
        save_image_with_poses(config, inputs, fused_poses, meta, cameras, resize_transform, prefix)
    
    if 'heatmaps' in config.TEST.VIS_TYPE:
        if config.DATASET.TEST_HEATMAP_SRC == 'pred':
            raise ValueError("cannot visualize heatmaps only given 2D predictions")
        save_heatmaps(inputs, input_heatmaps, prefix)


def vis_3d_poses(ax, num_person, poses, poses_vis=None):
    for n in range(num_person):
        pose = poses[n]
        if poses_vis is not None:
            pose_vis = poses_vis[n]

        for k in eval("LIMBS{}".format(len(pose))):
            x = [float(pose[k[0], 0]), float(pose[k[1], 0])]
            y = [float(pose[k[0], 1]), float(pose[k[1], 1])]
            z = [float(pose[k[0], 2]), float(pose[k[1], 2])]

            # 3d visualization for GT
            if poses_vis is not None:
                if pose_vis[k[0]] > 0.1 and pose_vis[k[1]] > 0.1:
                    ax.plot(x, y, z, c='r', lw=1.5, marker='o', 
                            markerfacecolor='w', markersize=2, markeredgewidth=1)
                else:
                    ax.plot(x, y, z, c='r', ls='--', lw=1.5, marker='o', 
                            markerfacecolor='w', markersize=2, markeredgewidth=1)
            
            # 3d visualization for prediction
            else:
                ax.plot(x, y, z, c=colors[int(n % 10)], lw=1.5, marker='o', 
                        markerfacecolor='w', markersize=2, markeredgewidth=1)
                

def vis_2d_poses(ax, num_person, poses, poses_vis=None, plane_type='xy'):
    if poses_vis is not None:  # gt visualization
        idx = idx_list[plane_type]
    else:
        idx = [0, 1]

    for n in range(num_person):
        pose = poses[n]
        if poses_vis is not None:
            pose_vis = poses_vis[n]

        for k in eval("LIMBS{}".format(len(pose))):
            x = [float(pose[k[0], idx[0]]), float(pose[k[1], idx[0]])]
            y = [float(pose[k[0], idx[1]]), float(pose[k[1], idx[1]])]

            # 2d visualization for GT
            if poses_vis is not None:
                if pose_vis[k[0]] > 0.1 and pose_vis[k[1]] > 0.1:
                    ax.plot(x, y, c='r', lw=1.5, marker='o', 
                            markerfacecolor='w', markersize=2, markeredgewidth=1)
                else:
                    ax.plot(x, y, c='r', ls='--', lw=1.5, marker='o', 
                            markerfacecolor='w', markersize=2, markeredgewidth=1)
            
            # 2d visualization for prediction
            else:
                ax.plot(x, y, c=colors[int(n % 10)], lw=1.5, marker='o', 
                        markerfacecolor='w', markersize=2, markeredgewidth=1)
    

def vis_2d_bbox(ax, config, proposal_centers, plane_type='xy'):
    idx = idx_list[plane_type]
    individual_space_size = config.INDIVIDUAL_SPEC.SPACE_SIZE
    
    for i in range(len(proposal_centers)):
        if proposal_centers[i, 3] < 0:
            continue
        
        top_left, bbox_size = [], []

        top_left.append(proposal_centers[i, 0] - proposal_centers[i, 5] * individual_space_size[0] / 2)
        top_left.append(proposal_centers[i, 1] - proposal_centers[i, 6] * individual_space_size[1] / 2)
        top_left.append(proposal_centers[i, 2] - individual_space_size[2] / 2)

        bbox_size.append(proposal_centers[i, 5] * individual_space_size[0])
        bbox_size.append(proposal_centers[i, 6] * individual_space_size[1])
        bbox_size.append(individual_space_size[2])
        
        top_left_x, top_left_y = top_left[idx[0]], top_left[idx[1]]
        width, height = bbox_size[idx[0]], bbox_size[idx[1]]
        
        rect = plt.Rectangle((top_left_x, top_left_y), width, height, fill=False, edgecolor = 'red',linewidth=1)
        ax.add_patch(rect)


def save_2d_planes(config, meta, fused_poses, plane_poses, proposal_centers, prefix):
    basename = os.path.basename(prefix)
    dirname = os.path.dirname(prefix)
    dirname1 = os.path.join(dirname, '2d_planes')

    if not os.path.exists(dirname1):
        os.makedirs(dirname1)
    prefix = os.path.join(dirname1, basename)
    file_name = "{}.png".format(prefix)

    batch_size = fused_poses.shape[0]
    xplot = 4 
    yplot = batch_size

    width = 4.0 * xplot
    height = 4.0 * yplot
    fig = plt.figure(0, figsize=(width, height))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
                        top=0.95, wspace=0.2, hspace=0.15)
    
    proposal_centers = proposal_centers.detach().cpu().numpy()
    mask = (proposal_centers[:, :, 3] >= 0)
    
    for i in range(batch_size):
        # 1. 3d projection
        ax = plt.subplot(yplot, xplot, 4*i+1, projection='3d')

        if 'joints_3d' in meta:
            num_person = meta['num_person'][i]
            joints_3d = meta['joints_3d'][i]
            joints_3d_vis = meta['joints_3d_vis'][i]
            vis_3d_poses(ax, num_person, joints_3d, poses_vis=joints_3d_vis)

        vis_3d_poses(ax, len(fused_poses[i, mask[i]]), fused_poses[i, mask[i]])
      
        if i == 0:
            plt.title('3d pose', fontdict={'weight':'normal','size': 15})
        

        # 2. xy_projection
        ax = plt.subplot(yplot, xplot, 4*i+2)
        
        if 'joints_3d' in meta:
            vis_2d_poses(ax, num_person, joints_3d, poses_vis=joints_3d_vis, plane_type='xy')

        vis_2d_poses(ax, len(plane_poses[0][i, mask[i]]), plane_poses[0][i, mask[i]])
        vis_2d_bbox(ax, config, proposal_centers[i], plane_type='xy')  

        if i == 0:
            plt.title('xy projection',fontdict={'weight':'normal','size': 15})


        # 3. xz_projection
        ax = plt.subplot(yplot, xplot, 4*i+3)

        if 'joints_3d' in meta:
            vis_2d_poses(ax, num_person, joints_3d, poses_vis=joints_3d_vis, plane_type='xz')

        vis_2d_poses(ax, len(plane_poses[1][i, mask[i]]), plane_poses[1][i, mask[i]])
        vis_2d_bbox(ax, config, proposal_centers[i], plane_type='xz')

        if i == 0:
            plt.title('xz projection',fontdict={'weight':'normal','size': 15})
        

        # 4. yz_projection
        ax = plt.subplot(yplot, xplot, 4*i+4)
        if 'joints_3d' in meta:
            vis_2d_poses(ax, num_person, joints_3d, poses_vis=joints_3d_vis, plane_type='yz')

        vis_2d_poses(ax, len(plane_poses[2][i, mask[i]]), plane_poses[2][i, mask[i]])
        vis_2d_bbox(ax, config, proposal_centers[i], plane_type='yz')

        if i == 0:
            plt.title('yz projection',fontdict={'weight':'normal','size': 15})
        
    plt.savefig(file_name)
    plt.close(0)


def save_image_with_poses(config, images, poses, meta, cameras, resize_transform, prefix):
    basename = os.path.basename(prefix)
    dirname = os.path.join(os.path.dirname(prefix), 'image_with_poses')
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    prefix = os.path.join(dirname, basename)
 
    batch_size, num_views, _, height, width = images.shape
    max_people = poses.shape[1]
    num_joints = poses.shape[2]

    for c in range(num_views):
        file_name = prefix + '_view_{}.jpg'.format(c + 1)
        batch_image = images[:, c].flip(1)
        grid = torchvision.utils.make_grid(batch_image, 1, padding=0, normalize=True)
        ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2,0).cpu().numpy().copy()
        limbs = eval("LIMBS{}".format(num_joints))

        for i in range(batch_size):
            curr_seq = meta['seq'][i]

            for n in range(max_people):
                if poses[i, n, 0, 4] < config.CAPTURE_SPEC.MIN_SCORE:
                    continue

                color = np.flip(np.array(matplotlib.colors.to_rgb(colors[int(n % 10)]))) * 255
                pose_2d = project_pose(poses[i, n, :, :3], cameras[curr_seq][c])
                pose_2d = do_transform(pose_2d, resize_transform)
                
                for j in range(num_joints):
                    if is_valid_coord(pose_2d[j], width, height):
                        xc = pose_2d[j][0]
                        yc = i * height + pose_2d[j][1]
                        cv2.circle(ndarr, (int(xc), int(yc)), 8, color, -1)
                    
                for limb in limbs:
                    parent = pose_2d[limb[0]]
                    if not is_valid_coord(parent, width, height):
                        continue
                    child = pose_2d[limb[1]]
                    if not is_valid_coord(child, width, height):
                        continue

                    px = parent[0]
                    py = i * height + parent[1]
                    cx = child[0]
                    cy = i * height + child[1]
                    cv2.line(ndarr, (int(px), int(py)), (int(cx), int(cy)), color, 4)
                    
        cv2.imwrite(file_name, ndarr)


def save_heatmaps(batch_images, batch_heatmaps, prefix):
    basename = os.path.basename(prefix)
    dirname = os.path.join(os.path.dirname(prefix), 'heatmaps')
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    prefix = os.path.join(dirname, basename)

    batch_size, num_views, _, image_height, image_width = batch_images.shape
    num_joints = batch_heatmaps.shape[2]

    for c in range(num_views):
        file_name = prefix + '_view_{}.jpg'.format(c + 1)

        batch_image = batch_images[:, c].clone()
        min = float(batch_image.min())
        max = float(batch_image.max())
        batch_image.add_(-min).div_(max - min + 1e-5)
    
        batch_image = batch_image.flip(1)
        grid_image = np.zeros((batch_size * image_height, (num_joints + 1) * image_width, 3), dtype=np.uint8)

        for i in range(batch_size):
            image = batch_image[i].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            heatmaps = batch_heatmaps[i, c].mul(255).clamp(0, 255).byte().cpu().numpy()
            height_begin = image_height * i
            height_end = image_height * (i + 1)
            for j in range(num_joints):
                resized_heatmap = cv2.resize(heatmaps[j], (int(image_width), int(image_height)))
                colored_heatmap = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)
                masked_image = colored_heatmap * 0.7 + image * 0.3

                width_begin = image_width * (j + 1)
                width_end = image_width * (j + 2)
                grid_image[height_begin:height_end, width_begin:width_end, :] = masked_image

            grid_image[height_begin:height_end, :image_width, :] = image
        cv2.imwrite(file_name, grid_image)


def is_valid_coord(joint, width, height):
    valid_x = joint[0] >= 0 and joint[0] < width
    valid_y = joint[1] >= 0 and joint[1] < height
    return valid_x and valid_y