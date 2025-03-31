import numpy as np

def normalize(q):
    return q / np.linalg.norm(q)

def quat_mult(q, p):
    qx, qy, qz, qw = q
    px, py, pz, pw = p
    return np.array([
        qw*pw - qx*px - qy*py - qz*pz,
        qw*px + qx*pw + qy*pz - qz*py,
        qw*py - qx*pz + qy*pw + qz*px,
        qw*pz + qx*py - qy*px + qz*pw
    ])

def quat_from_omega(omega, dt):
    angle = np.linalg.norm(omega) * dt
    if angle < 1e-6:
        return np.array([0, 0, 0, 1])
    axis = omega / np.linalg.norm(omega)
    half_angle = angle / 2.0
    return normalize(np.array([*(np.sin(half_angle) * axis), np.cos(half_angle)]))

