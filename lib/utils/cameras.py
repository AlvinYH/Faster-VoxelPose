# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import division
import torch
import numpy as np


def unfold_camera_param(camera, device=None):
    R = torch.as_tensor(camera['R'], dtype=torch.float, device=device)
    T = torch.as_tensor(camera['T'], dtype=torch.float, device=device)
    fx = torch.as_tensor(camera['fx'], dtype=torch.float, device=device)
    fy = torch.as_tensor(camera['fy'], dtype=torch.float, device=device)
    f = torch.tensor([fx, fy], dtype=torch.float, device=device).reshape(2, 1)
    c = torch.as_tensor(
        [[camera['cx']], [camera['cy']]],
        dtype=torch.float,
        device=device)
    k = torch.as_tensor(camera['k'], dtype=torch.float, device=device)
    p = torch.as_tensor(camera['p'], dtype=torch.float, device=device)
    return R, T, f, c, k, p


def project_pose(x, camera):
    R, T, f, c, k, p = unfold_camera_param(camera, device=x.device)
    return project_point(x, R, T, f, c, k, p)


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


def world_to_camera_frame(x, R, T):
    """
    Args
        x: Nx3 3d points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
    Returns
        xcam: Nx3 3d points in camera coordinates
    """

    R = torch.as_tensor(R, device=x.device)
    T = torch.as_tensor(T, device=x.device)
    xcam = torch.mm(R, torch.t(x) - T)
    return torch.t(xcam)


def camera_to_world_frame(x, R, T):
    """
    Args
        x: Nx3 points in camera coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
    Returns
        xcam: Nx3 points in world coordinates
    """

    R = torch.as_tensor(R, device=x.device)
    T = torch.as_tensor(T, device=x.device)
    xcam = torch.mm(torch.t(R), torch.t(x))
    xcam = xcam + T  # rotate and translate
    return torch.t(xcam)
