import numpy as np

def rotation_matrix_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr
    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr
    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr

    R = np.array([
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22],
    ])
    return R

def make_transform_matrix(xyz, rpy):
    T = np.eye(4)
    R = rotation_matrix_rpy(rpy[0], rpy[1], rpy[2])
    T[:3, :3] = R
    T[0, 3] = xyz[0]
    T[1, 3] = xyz[1]
    T[2, 3] = xyz[2]
    return T

def prismatic_transform(xyz, rpy, axis, q_val):
    T_origin = make_transform_matrix(xyz, rpy)

    axis = np.asarray(axis) 
    if np.linalg.norm(axis) < 1e-12:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = axis / np.linalg.norm(axis)

    T_slide = np.eye(4)
    T_slide[0:3, 3] = axis * q_val

    return T_origin @ T_slide

def revolute_transform(xyz, rpy, axis, q_val):
    T_origin = make_transform_matrix(xyz, rpy)

    axis = np.asarray(axis)
    if np.linalg.norm(axis) < 1e-12:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = axis / np.linalg.norm(axis)

    c = np.cos(q_val)
    s = np.sin(q_val)
    vx, vy, vz = axis

    R_axis = np.array([
        [c + vx*vx*(1-c),   vx*vy*(1-c) - vz*s, vx*vz*(1-c) + vy*s],
        [vy*vx*(1-c) + vz*s, c + vy*vy*(1-c),   vy*vz*(1-c) - vx*s],
        [vz*vx*(1-c) - vy*s, vz*vy*(1-c) + vx*s, c + vz*vz*(1-c)]
    ])

    T_rot = np.eye(4)
    T_rot[:3, :3] = R_axis

    return T_origin @ T_rot
