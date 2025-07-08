import numpy as np
import torch
import matplotlib.pyplot as plt
def box_minus(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Implements box-minus operator (quaternion difference)
    https://docs.leggedrobotics.com/kindr/cheatsheet_latest.pdf

    Args:
        q1 (torch.Tensor): quaternion
        q2 (torch.Tensor): quaternion

    Returns:
        torch.Tensor: q2 box-minus q1
    """
    eps: float = 1e-9
    quat = quat_mul(quat_conjugate(q2), q1)
    mask = quat[:, 0] < 0
    quat[mask] = -quat[mask]
    quat_w = quat[:, 0]
    quat_vec = quat[:, 1:]
    quat_vec_norm = torch.norm(quat_vec, dim=1)
    return (
        2.0
        * torch.atan2(quat_vec_norm, quat_w).unsqueeze(-1)
        * (quat_vec / quat_vec_norm.clamp(min=eps, max=None).unsqueeze(-1))
    )

def box_plus(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Implements box-plus operator (addition of 3D rotation vector to quaternion)
    https://docs.leggedrobotics.com/kindr/cheatsheet_latest.pdf
    Args:
        q (torch.Tensor): quaternion
        v (torch.Tensor): 3D-vector (rotation vector)
    Returns:
        torch.Tensor: q box-plus v
    """
    eps: float = 1e-9
    v_norm = torch.norm(v, dim=1)
    v_quat_w = torch.cos(v_norm * 0.5).unsqueeze(-1)
    v_quat_vec = torch.sin(v_norm * 0.5).unsqueeze(-1) * (v / v_norm.clamp(min=eps, max=None).unsqueeze(-1))
    v_quat = torch.cat([v_quat_w, v_quat_vec], dim=1)
    return quat_mul(quat_conjugate(q), v_quat)
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from scipy.spatial.transform import Rotation

@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)


@torch.jit.script
def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

def euler_angles_to_quats(
    euler_angles: torch.Tensor, degrees: bool = False, extrinsic: bool = True, device=None
) -> torch.Tensor:
    """Vectorized version of converting euler angles to quaternion (scalar first)

    Args:
        euler_angles (typing.Union[np.ndarray, torch.Tensor]): euler angles with shape (N, 3)
        degrees (bool, optional): True if degrees, False if radians. Defaults to False.
        extrinsic (bool, optional): True if the euler angles follows the extrinsic angles
                   convention (equivalent to ZYX ordering but returned in the reverse) and False if it follows
                   the intrinsic angles conventions (equivalent to XYZ ordering).
                   Defaults to True.

    Returns:
        typing.Union[np.ndarray, torch.Tensor]: quaternions representation of the angles (N, 4) - scalar first.
    """
    if extrinsic:
        order = "xyz"
    else:
        order = "XYZ"
    # TODO: implement a torch version
    rot = Rotation.from_euler(order, euler_angles.cpu().numpy(), degrees=degrees)
    result = rot.as_quat()
    if len(result.shape) == 1:
        result = result[[3, 0, 1, 2]]
    else:
        result = result[:, [3, 0, 1, 2]]
    result = torch.from_numpy(np.asarray(result, dtype=np.float32)).float().to(device)
    return result


def rot_matrices_to_quats(rotation_matrices: torch.Tensor, device=None) -> torch.Tensor:
    """Vectorized version of converting rotation matrices to quaternions

    Args:
        rotation_matrices (torch.Tensor): N Rotation matrices with shape (N, 3, 3) or (3, 3)

    Returns:
        torch.Tensor: quaternion representation of the rotation matrices (N, 4) or (4,) - scalar first
    """
    rot = Rotation.from_matrix(rotation_matrices.cpu().numpy())
    result = rot.as_quat()
    if len(result.shape) == 1:
        result = result[[3, 0, 1, 2]]
    else:
        result = result[:, [3, 0, 1, 2]]
    result = torch.from_numpy(np.asarray(result, dtype=np.float32)).float().to(device)
    return result


def rad2deg(radian_value: torch.Tensor, device=None) -> torch.Tensor:
    """_summary_

    Args:
        radian_value (torch.Tensor): _description_
        device (_type_, optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    return torch.rad2deg(radian_value).float().to(device)


def deg2rad(degree_value: float, device=None) -> torch.Tensor:
    """_summary_

    Args:
        degree_value (torch.Tensor): _description_
        device (_type_, optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    return torch.deg2rad(degree_value).float().to(device)


@torch.jit.script
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([w, x, y, z], dim=-1).view(shape)

    return quat


@torch.jit.script
def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((a[:, 0:1], -a[:, 1:]), dim=-1).view(shape)


@torch.jit.script
def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, 1:]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


@torch.jit.script
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


@torch.jit.script
def quat_unit(a):
    return normalize(a)


@torch.jit.script
def quat_from_angle_axis(angle, axis):
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return quat_unit(torch.cat([w, xyz], dim=-1))


@torch.jit.script
def quat_axis(q, axis=0):
    # type: (Tensor, int) -> Tensor
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


@torch.jit.script
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


@torch.jit.script
def get_basis_vector(q, v):
    return quat_rotate(q, v)


@torch.jit.script
def quats_to_rot_matrices(quats):
    squeeze_flag = False
    if quats.dim() == 1:
        squeeze_flag = True
        quats = torch.unsqueeze(quats, 0)
    nq = torch.linalg.vecdot(quats, quats, dim=1)
    singularities = nq < 1e-10
    result = torch.zeros(quats.shape[0], 3, 3, device=quats.device)
    result[singularities] = torch.eye(3, device=quats.device).reshape((1, 3, 3)).repeat(sum(singularities), 1, 1)
    non_singular = quats[torch.logical_not(singularities)] * torch.sqrt(2.0 / nq).reshape((-1, 1)).repeat(1, 4)
    non_singular = torch.einsum("bi,bj->bij", non_singular, non_singular)
    result[torch.logical_not(singularities), 0, 0] = 1.0 - non_singular[:, 2, 2] - non_singular[:, 3, 3]
    result[torch.logical_not(singularities), 0, 1] = non_singular[:, 1, 2] - non_singular[:, 3, 0]
    result[torch.logical_not(singularities), 0, 2] = non_singular[:, 1, 3] + non_singular[:, 2, 0]
    result[torch.logical_not(singularities), 1, 0] = non_singular[:, 1, 2] + non_singular[:, 3, 0]
    result[torch.logical_not(singularities), 1, 1] = 1.0 - non_singular[:, 1, 1] - non_singular[:, 3, 3]
    result[torch.logical_not(singularities), 1, 2] = non_singular[:, 2, 3] - non_singular[:, 1, 0]
    result[torch.logical_not(singularities), 2, 0] = non_singular[:, 1, 3] - non_singular[:, 2, 0]
    result[torch.logical_not(singularities), 2, 1] = non_singular[:, 2, 3] + non_singular[:, 1, 0]
    result[torch.logical_not(singularities), 2, 2] = 1.0 - non_singular[:, 1, 1] - non_singular[:, 2, 2]
    if squeeze_flag:
        result = torch.squeeze(result)
    return result


@torch.jit.script
def matrices_to_euler_angles(mat, extrinsic: bool = True):
    _POLE_LIMIT = 1.0 - 1e-6
    if extrinsic:
        north_pole = mat[:, 2, 0] > _POLE_LIMIT
        south_pole = mat[:, 2, 0] < -_POLE_LIMIT
        result = torch.zeros(mat.shape[0], 3, device=mat.device)
        result[north_pole, 0] = 0.0
        result[north_pole, 1] = -np.pi / 2
        result[north_pole, 2] = torch.arctan2(mat[north_pole, 0, 1], mat[north_pole, 0, 2])
        result[south_pole, 0] = 0.0
        result[south_pole, 1] = np.pi / 2
        result[south_pole, 2] = torch.arctan2(mat[south_pole, 0, 1], mat[south_pole, 0, 2])
        result[torch.logical_not(torch.logical_or(south_pole, north_pole)), 0] = torch.arctan2(
            mat[torch.logical_not(torch.logical_or(south_pole, north_pole)), 2, 1],
            mat[torch.logical_not(torch.logical_or(south_pole, north_pole)), 2, 2],
        )
        result[torch.logical_not(torch.logical_or(south_pole, north_pole)), 1] = -torch.arcsin(
            mat[torch.logical_not(torch.logical_or(south_pole, north_pole)), 2, 0]
        )
        result[torch.logical_not(torch.logical_or(south_pole, north_pole)), 2] = torch.arctan2(
            mat[torch.logical_not(torch.logical_or(south_pole, north_pole)), 1, 0],
            mat[torch.logical_not(torch.logical_or(south_pole, north_pole)), 0, 0],
        )
    else:
        north_pole = mat[:, 2, 0] > _POLE_LIMIT
        south_pole = mat[:, 2, 0] < -_POLE_LIMIT
        result = torch.zeros(mat.shape[0], 3, device=mat.device)
        result[north_pole, 0] = torch.arctan2(mat[north_pole, 1, 0], mat[north_pole, 1, 1])
        result[north_pole, 1] = np.pi / 2
        result[north_pole, 2] = 0.0
        result[south_pole, 0] = torch.arctan2(mat[south_pole, 1, 0], mat[south_pole, 1, 1])
        result[south_pole, 1] = -np.pi / 2
        result[south_pole, 2] = 0.0
        result[torch.logical_not(torch.logical_or(south_pole, north_pole)), 0] = -torch.arctan2(
            mat[torch.logical_not(torch.logical_or(south_pole, north_pole)), 1, 2],
            mat[torch.logical_not(torch.logical_or(south_pole, north_pole)), 2, 2],
        )
        result[torch.logical_not(torch.logical_or(south_pole, north_pole)), 1] = torch.arcsin(
            mat[torch.logical_not(torch.logical_or(south_pole, north_pole)), 0, 2]
        )
        result[torch.logical_not(torch.logical_or(south_pole, north_pole)), 2] = -torch.arctan2(
            mat[torch.logical_not(torch.logical_or(south_pole, north_pole)), 0, 1],
            mat[torch.logical_not(torch.logical_or(south_pole, north_pole)), 0, 0],
        )
    return result


@torch.jit.script
def get_euler_xyz(q, extrinsic: bool = True):
    if extrinsic:
        qw, qx, qy, qz = 0, 1, 2, 3
        # roll (x-axis rotation)
        sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
        cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)
        sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
        pitch = torch.where(torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

        # yaw (z-axis rotation)
        siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
        cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)
    else:
        result = matrices_to_euler_angles(quats_to_rot_matrices(q), extrinsic=False)
        return result[:, 0], result[:, 1], result[:, 2]


@torch.jit.script
def quat_from_euler_xyz(roll, pitch, yaw, extrinsic: bool = True):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    if extrinsic:
        qw = cy * cr * cp + sy * sr * sp
        qx = cy * sr * cp - sy * cr * sp
        qy = cy * cr * sp + sy * sr * cp
        qz = sy * cr * cp - cy * sr * sp
    else:
        qw = -sr * sp * sy + cr * cp * cy
        qx = sr * cp * cy + sp * sy * cr
        qy = -sr * sy * cp + sp * cr * cy
        qz = sr * sp * cy + sy * cr * cp

    return torch.stack([qw, qx, qy, qz], dim=-1)


@torch.jit.script
def quat_diff_rad(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Get the difference in radians between two quaternions.

    Args:
        a: first quaternion, shape (N, 4)
        b: second quaternion, shape (N, 4)
    Returns:
        Difference in radians, shape (N,)
    """
    b_conj = quat_conjugate(b)
    mul = quat_mul(a, b_conj)
    # 2 * torch.acos(torch.abs(mul[:, -1]))
    return 2.0 * torch.asin(torch.clamp(torch.norm(mul[:, 1:], p=2, dim=-1), max=1.0))


# NB: do not make this function jit, since it is passed around as an argument.
def normalise_quat_in_pose(pose):
    """Takes a pose and normalises the quaternion portion of it.

    Args:
        pose: shape N, 7
    Returns:
        Pose with normalised quat. Shape N, 7
    """
    pos = pose[:, 0:3]
    quat = pose[:, 3:7]
    quat /= torch.norm(quat, dim=-1, p=2).reshape(-1, 1)
    return torch.cat([pos, quat], dim=-1)


@torch.jit.script
def compute_heading_and_up(torso_rotation, inv_start_rot, to_target, vec0, vec1, up_idx):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    num_envs = torso_rotation.shape[0]
    target_dirs = normalize(to_target)

    torso_quat = quat_mul(torso_rotation, inv_start_rot)
    up_vec = get_basis_vector(torso_quat, vec1).view(num_envs, 3)
    heading_vec = get_basis_vector(torso_quat, vec0).view(num_envs, 3)
    up_proj = up_vec[:, up_idx]
    heading_proj = torch.bmm(heading_vec.view(num_envs, 1, 3), target_dirs.view(num_envs, 3, 1)).view(num_envs)

    return torso_quat, up_proj, heading_proj, up_vec, heading_vec


@torch.jit.script
def compute_rot(torso_quat, velocity, ang_velocity, targets, torso_positions, extrinsic: bool = True):
    vel_loc = quat_rotate_inverse(torso_quat, velocity)
    angvel_loc = quat_rotate_inverse(torso_quat, ang_velocity)

    roll, pitch, yaw = get_euler_xyz(torso_quat, extrinsic=extrinsic)

    walk_target_angle = torch.atan2(targets[:, 2] - torso_positions[:, 2], targets[:, 0] - torso_positions[:, 0])
    angle_to_target = walk_target_angle - yaw

    return vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target


def xyzw2wxyz(q):
    return torch.roll(q, 1, -1)


def wxyz2xyzw(q):
    return torch.roll(q, -1, -1)

@torch.jit.script
def get_yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    """Extract the yaw component of a quaternion.

    Args:
        quat: The orientation in (w, x, y, z). Shape is (..., 4)

    Returns:
        A quaternion with only yaw component.
    """
    shape = quat.shape
    quat_yaw = quat.clone().view(-1, 4)
    qw = quat_yaw[:, 0]
    qx = quat_yaw[:, 1]
    qy = quat_yaw[:, 2]
    qz = quat_yaw[:, 3]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw[:] = 0.0
    quat_yaw[:, 3] = torch.sin(yaw / 2)
    quat_yaw[:, 0] = torch.cos(yaw / 2)
    quat_yaw = normalize(quat_yaw)
    return quat_yaw.view(shape)