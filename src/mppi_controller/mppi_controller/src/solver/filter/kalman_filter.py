import numpy as np
import torch

from filterpy.kalman import ExtendedKalmanFilter
from mppi_controller.src.utils.pose import Pose


class SatellitePoseKalmanFilter(ExtendedKalmanFilter):
    """
    x = [px, dx, py, pz, r, p, y]
    u = [dx, dy, dz, dr, dp, dy]
    """
    def __init__(self, dim_x=6, dim_z=6):
        self.n_x = dim_x
        self.n_z = dim_z
        self.n_u = 6
        self.dt = 0.1
        self.n_step = 1
        super().__init__(dim_x=self.n_x, dim_z=self.n_z)

        self.x = np.array([0,0,0,0,0,0], dtype=np.float32)

        self.B = np.eye(self.n_x, dtype=np.float32) * self.dt
        self.P = np.eye(self.n_x, dtype=np.float32) * 1e-2
        self.Q = np.eye(self.n_x, dtype=np.float32) * 1e-4
        self.R = np.eye(self.n_z, dtype=np.float32) * 1e-1
        self.F = np.eye(self.n_x, dtype=np.float32)

        self.u_prev = np.zeros(self.n_u, dtype=np.float32)
        self.future_x = np.zeros((self.n_step, self.n_x), dtype=np.float32)
        self.future_P = np.zeros((self.n_step, self.n_x, self.n_x), dtype=np.float32)


    def fx(self, x, u, dt):
        pos = x[0:3]
        rot = x[3:6]
        dpos = u[0:3]
        drot = u[3:6]

        pos_next = pos + dpos * dt
        rot_next = rot + drot * dt
        
        return np.hstack((pos_next, rot_next)).flatten()

    def Hx(self, x):
        return x
    
    def FJacobian(self, x, u, dt):
        F = np.eye(len(x))
        return F
    

    def HJacobian(self, x, dt):
        n = len(x)
        return np.eye(n)
    

    def predict_and_update(self, z: Pose, u_p, u_r, dt = None):
        if dt is not None:
            self.dt = dt

        z = np.concatenate([z.np_pose, z.np_rpy])
        u = torch.cat([u_p, u_r], dim=0).numpy()

        self.u_prev = u.copy()

        self.B = np.eye(self.n_x, dtype=np.float32) * self.dt
        self.F = self.FJacobian(self.x, u, self.dt)
        self.predict_update(z=z, HJacobian=self.HJacobian, Hx=self.Hx, args=(dt,), hx_args=(), u=u)
        return


    def predict_multi_step(self, n_step: int):
        if self.n_step != n_step:
            self.n_step = n_step
            self.future_x = np.zeros((self.n_step, self.n_x), dtype=np.float32)
            self.future_P = np.zeros((self.n_step, self.n_x, self.n_x), dtype=np.float32)

        x_pred = self.x.copy()
        P_pred = self.P.copy()
        self.B = np.eye(self.n_x, dtype=np.float32) * self.dt
        
        for i in range(n_step):
            F = self.FJacobian(x_pred.flatten(), self.u_prev, self.dt)
            
            x_pred = self.fx(x_pred.flatten(), self.u_prev, self.dt)
            P_pred = np.dot(F, P_pred).dot(F.T) + self.Q

            self.future_x[i,:] = x_pred.copy()
            self.future_P[i,:,:] = P_pred.copy()
        return self.future_x, self.future_P


    def set_init_pose(self, pose: Pose):
        self.x[0:3] = pose.np_pose
        self.x[3:6] = pose.np_rpy
        return
    
    @property
    def pose(self):
        return torch.from_numpy(self.x[0:3])
    
    @property
    def rpy(self):
        return torch.from_numpy(self.x[3:6])
    
    @property
    def np_pose(self):
        return self.x[0:3].flatten().copy()
    
    @property
    def np_rpy(self):
        return self.x[3:6].flatten().copy()
    