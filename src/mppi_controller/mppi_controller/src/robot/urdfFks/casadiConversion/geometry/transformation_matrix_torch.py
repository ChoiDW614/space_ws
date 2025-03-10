import torch
from torch import sin, cos, tensor, eye

def rotation_matrix_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = cos(roll)
    sr = sin(roll)
    cp = cos(pitch)
    sp = sin(pitch)
    cy = cos(yaw)
    sy = sin(yaw)

    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr
    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr
    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr

    R = tensor([
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22],
    ])
    return R

def make_transform_matrix(xyz, rpy):
    T = eye(4)
    R = rotation_matrix_rpy(rpy[0], rpy[1], rpy[2])
    T[:3, :3] = R
    T[0, 3] = xyz[0]
    T[1, 3] = xyz[1]
    T[2, 3] = xyz[2]
    return T

def prismatic_transform(xyz, rpy, axis, q_val):
    T_origin = make_transform_matrix(xyz, rpy)

    axis = tensor(axis)
    if torch.linalg.norm(axis) < 1e-12:
        axis = tensor([1.0, 0.0, 0.0])
    else:
        axis = axis / torch.linalg.norm(axis)

    T_slide = eye(4)
    T_slide[0:3, 3] = axis * q_val

    return T_origin @ T_slide

def revolute_transform(xyz, rpy, axis, q_val):
    T_origin = make_transform_matrix(xyz, rpy)

    axis = tensor(axis)
    if torch.linalg.norm(axis) < 1e-12:
        axis = tensor([1.0, 0.0, 0.0])
    else:
        axis = axis / torch.linalg.norm(axis)

    c = cos(q_val)
    s = sin(q_val)
    vx, vy, vz = axis

    R_axis = tensor([
        [c + vx*vx*(1-c),   vx*vy*(1-c) - vz*s, vx*vz*(1-c) + vy*s],
        [vy*vx*(1-c) + vz*s, c + vy*vy*(1-c),   vy*vz*(1-c) - vx*s],
        [vz*vx*(1-c) - vy*s, vz*vy*(1-c) + vx*s, c + vz*vz*(1-c)]
    ])

    T_rot = eye(4)
    T_rot[:3, :3] = R_axis

    return T_origin @ T_rot
