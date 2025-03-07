# transformation_matrix_numpy.py
import numpy as np

def rotation_matrix_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Roll-Pitch-Yaw (ZYX 혹은 원하는 convention)에 따라 3x3 회전행렬을 반환.
    (아래 예시는 roll->pitch->yaw 순으로 곱한다고 가정한 ZYX 컨벤션 예시)
    """
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # ZYX 형태로 회전행렬 구성
    # 만약 X->Y->Z 등 다른 convention을 쓰고 싶다면 원하는 순서로 곱셈 변경 가능
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
    ], dtype=float)
    return R

def make_transform_matrix(xyz, rpy):
    """
    xyz: (x, y, z) 위치
    rpy: (roll, pitch, yaw)
    를 받아 4x4 동차변환행렬을 구성.
    """
    T = np.eye(4)
    R = rotation_matrix_rpy(rpy[0], rpy[1], rpy[2])
    T[:3, :3] = R
    T[0, 3] = xyz[0]
    T[1, 3] = xyz[1]
    T[2, 3] = xyz[2]
    return T

def prismatic_transform(xyz, rpy, axis, q_val):
    """
    Prismatic(슬라이더) 조인트의 변환행렬 (4x4).
    xyz, rpy로 기본 좌표계를 만든 뒤,
    axis 방향으로 q_val만큼 슬라이드.
    """
    # 먼저 조인트의 origin 변환행렬
    T_origin = make_transform_matrix(xyz, rpy)

    # axis 방향으로 q_val만큼 이동하는 변환행렬
    # axis는 normalize 해두는 게 안전
    axis = np.asarray(axis, dtype=float)
    if np.linalg.norm(axis) < 1e-12:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = axis / np.linalg.norm(axis)

    T_slide = np.eye(4)
    T_slide[0:3, 3] = axis * q_val

    # 최종 변환행렬 = origin * slide
    return T_origin @ T_slide

def revolute_transform(xyz, rpy, axis, q_val):
    """
    Revolute(회전) 조인트의 변환행렬 (4x4).
    xyz, rpy로 기본 좌표계를 만든 뒤,
    axis를 중심으로 q_val 라디안만큼 회전.
    """
    # 먼저 조인트의 origin 변환행렬
    T_origin = make_transform_matrix(xyz, rpy)

    axis = np.asarray(axis, dtype=float)
    if np.linalg.norm(axis) < 1e-12:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = axis / np.linalg.norm(axis)

    c = np.cos(q_val)
    s = np.sin(q_val)
    vx, vy, vz = axis

    # 로드리게스 회전공식
    R_axis = np.array([
        [c + vx*vx*(1-c),   vx*vy*(1-c) - vz*s, vx*vz*(1-c) + vy*s],
        [vy*vx*(1-c) + vz*s, c + vy*vy*(1-c),   vy*vz*(1-c) - vx*s],
        [vz*vx*(1-c) - vy*s, vz*vy*(1-c) + vx*s, c + vz*vz*(1-c)]
    ], dtype=float)

    T_rot = np.eye(4)
    T_rot[:3, :3] = R_axis

    return T_origin @ T_rot
