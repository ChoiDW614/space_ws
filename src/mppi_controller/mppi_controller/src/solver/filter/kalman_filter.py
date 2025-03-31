import numpy as np
import torch

from filterpy.kalman import ExtendedKalmanFilter
from mppi_controller.src.solver.filter.quaternion import *
from mppi_controller.src.utils.pose import Pose


class KalmanFilter(object):
    """
    x = [px, py, pz, q0, q1, q2, q3]
    u = [dx, dy, dz, dr, dp, dyaw]
    """
    def __init__(self):
        self.n_x = 7
        self.n_z = 7
        self.n_u = 6
        self.dt = 0.1
        self.ekf = ExtendedKalmanFilter(dim_x=self.n_x, dim_z=self.n_z)

        self.ekf.x = np.array([0,0,0,0,0,0,1])  

        self.ekf.P = np.eye(self.n_x) * 1e-2
        self.ekf.Q = np.eye(self.n_x) * 1e-4
        self.ekf.R = np.eye(self.n_z) * 1e-1

        self.ekf.F = np.eye(self.n_x)

    def fx(self, x, u, dt):
        pose = x[0:3]
        q = x[3:7]
        v = u[0:3]
        # omega = u[3:6]

        # Update the state
        pos_next = pose + v * dt
        # delta_q = quat_from_omega(omega, dt)
        delta_q = u[3:7] * dt
        quat_next = quat_mult(q, delta_q)
        quat_next = normalize(quat_next)
        
        return np.hstack((pos_next, quat_next)).reshape(7, 1)

    def Hx(self, x):
        return x
    
    def FJacobian(self, x, u, dt):
        F = np.eye(7)
        # omega = u[3:6]
        # delta_q = quat_from_omega(omega, dt)
        delta_q = u[3:7] * dt

        L = np.array([
            [delta_q[3], -delta_q[0], -delta_q[1], -delta_q[2]],
            [delta_q[0],  delta_q[3],  delta_q[2], -delta_q[1]],
            [delta_q[1], -delta_q[2],  delta_q[3],  delta_q[0]],
            [delta_q[2],  delta_q[1], -delta_q[0],  delta_q[3]]
        ])
        F[3:7, 3:7] = L
        return F
    

    def HJacobian(self, x, dt):
        n = len(x)
        return np.eye(n)
    

    def update(self, z, u, dt = None):
        if dt is None:
            dt = self.dt

        if isinstance(z, torch.Tensor):
            z = z.detach().cpu().numpy()
        elif isinstance(z, Pose):
            z = np.concatenate([z.pose.numpy(), z.orientation.numpy()])
        
        if isinstance(u, torch.Tensor):
            u = u.detach().cpu().numpy()
        elif isinstance(u, Pose):
            u = np.concatenate([u.pose.numpy(), u.orientation.numpy()])

        self.predict_update(z, u, dt)
    

    def predict_update(self, z, u, dt):
        self.ekf.F = np.eye(self.n_x)
        self.ekf.predict_update(z=z, HJacobian=self.HJacobian, Hx=self.Hx, args=(dt,), hx_args=(), u=u)

        q = self.ekf.x[3:7].flatten()
        q = normalize(q)
        self.ekf.x[3:7] = q.reshape(4,)


    def predict_x(self, n_step: int):
        x_pred, P_pred = self.predict_multi_step(self.ekf.x, self.ekf.P, n_step, self.dt,
                                                 self.ekf.Q, self.FJacobian, self.fx)
        return x_pred, P_pred


    def predict_multi_step(x, P, n_steps, dt, Q, FJacobian, fx_func, u = 0):
        x_pred = x.copy()
        P_pred = P.copy()
        for _ in range(n_steps):
            F = FJacobian(x_pred.flatten(), u, dt)
            x_pred = fx_func(x_pred, u, dt)
            P_pred = F @ P_pred @ F.T + Q

            q = x_pred[3:7].flatten()
            q = normalize(q)
            x_pred[3:7] = q.reshape(4, 1)
        return x_pred, P_pred


    def set_init_pose(self, pose: Pose):
        self.ekf.x[0:3] = pose.pose.numpy()
        self.ekf.x[3:7] = pose.orientation.numpy()
        return
    
