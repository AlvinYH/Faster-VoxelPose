# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import division
import torch
import numpy as np


def unfold_camera_param(camera, device):
    R = torch.as_tensor(camera['R'], dtype=torch.float, device=device)
    T = torch.as_tensor(camera['T'], dtype=torch.float, device=device)
    f = torch.tensor(np.array([camera['fx'], camera['fy']]), dtype=torch.float, device=device).reshape(-1, 1)
    c = torch.tensor(np.array([camera['cx'], camera['cy']]), dtype=torch.float, device=device).reshape(-1, 1)
    k = torch.as_tensor(camera['k'], dtype=torch.float, device=device)
    p = torch.as_tensor(camera['p'], dtype=torch.float, device=device)
    return R, T, f, c, k, p

def unfold_camera_param_cpu(camera):
    R = camera['R']
    T = camera['T']
    f = np.array([camera['fx'], camera['fy']]).reshape(-1, 1)
    c = np.array([camera['cx'], camera['cy']]).reshape(-1, 1)
    k = camera['k']
    p = camera['p']
    return R, T, f, c, k, p


def project_point(x, R, T, f, c, k, p):
    """
    Args
        x: Nx3 points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        f: (scalar) Camera focal length
        c: 2x1 Camera center
        k: 3x1 Camera radial distortion coefficients
        p: 2x1 Camera tangential distortion coefficients
    Returns
        ypixel.T: Nx2 points in pixel space
    """
    xcam = torch.mm(R, x.T - T)
    y = xcam[:2] / (xcam[2] + 1e-5)

    # add camera distortion
    r = torch.sum(y ** 2, dim=0)
    d = 1 + k[0] * r + k[1] * r * r + k[2] * r * r * r
    u = y[0, :] * d + 2 * p[0] * y[0, :] * y[1, :] + p[1] * (r + 2 * y[0, :] * y[0, :])
    v = y[1, :] * d + 2 * p[1] * y[0, :] * y[1, :] + p[0] * (r + 2 * y[1, :] * y[1, :])
    y[0, :] = u
    y[1, :] = v

    ypixel = f * y + c

    return ypixel.T

def project_point_cpu(x, R, T, f, c, k, p):
    """
    Args
        x: Nx3 points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        f: (scalar) Camera focal length
        c: 2x1 Camera center
        k: 3x1 Camera radial distortion coefficients
        p: 2x1 Camera tangential distortion coefficients
    Returns
        ypixel.T: Nx2 points in pixel space
    """
    xcam = np.matmul(R, x.T - T)
    y = xcam[:2] / (xcam[2] + 1e-5)

    # add camera distortion
    r = np.sum(y ** 2, axis=0)
    d = 1 + k[0] * r + k[1] * r * r + k[2] * r * r * r
    u = y[0, :] * d + 2 * p[0] * y[0, :] * y[1, :] + p[1] * (r + 2 * y[0, :] * y[0, :])
    v = y[1, :] * d + 2 * p[1] * y[0, :] * y[1, :] + p[0] * (r + 2 * y[1, :] * y[1, :])
    y[0, :] = u
    y[1, :] = v

    ypixel = f * y + c

    return ypixel.T


def project_pose(x, camera):
    R, T, f, c, k, p = unfold_camera_param(camera, x.device)
    return project_point(x, R, T, f, c, k, p)

def project_pose_cpu(x, camera):
    R, T, f, c, k, p = unfold_camera_param_cpu(camera)
    return project_point_cpu(x, R, T, f, c, k, p)