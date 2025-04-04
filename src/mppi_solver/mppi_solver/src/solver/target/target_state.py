import numpy as np
import torch
from collections import deque
from rclpy.logging import get_logger
from builtin_interfaces.msg import Time as MSG_Time

from mppi_solver.src.solver.filter.kalman_filter import SatellitePoseKalmanFilter

from mppi_solver.src.utils.pose import Pose
from mppi_solver.src.utils.time import Time


class DockingInterface(object):
    def __init__(self, init_pose: Pose, predict_step: int):
        self.logger = get_logger("EKF")

        self.interface_pose = Pose()
        self.interface_pose_prev = Pose()
        self.interface_vel_pose = torch.tensor([0.0, 0.0, 0.0])
        self.interface_vel_rpy = torch.tensor([0.0, 0.0, 0.0])

        self.interface_time = Time()
        self.interface_time_prev = Time()

        self.ekf = SatellitePoseKalmanFilter(dim_x=6, dim_z=6)
        self.ekf.set_init_pose(init_pose)
        self.predict_interface_pose = None
        self.predict_interface_cov = None

        self.n_predict = predict_step
        self.dt = 0.0
        self.eps = 1e-9

        # self.x_history = deque(maxlen=self.n_predict)


    def update_velocity(self):
        self.dt = self.interface_time.time - self.interface_time_prev.time
        self.interface_vel_pose = (self.interface_pose.pose - self.interface_pose_prev.pose) / self.dt
        self.interface_vel_rpy = self.compute_angular_rate(self.interface_pose.rpy, self.interface_pose_prev.rpy)
        return
    

    def compute_angular_rate(self, euler1, euler2):
        diff = euler1 - euler2
        diff = (diff + torch.pi) % (2 * torch.pi) - torch.pi
        angular_rate = diff / self.dt
        return angular_rate


    def ekf_update(self):
        self.dt = self.interface_time.time - self.interface_time_prev.time
        self.ekf.predict_and_update(self.interface_pose, self.interface_vel_pose, self.interface_vel_rpy, self.dt)
        self.predict_interface_pose, self.predict_interface_cov = self.ekf.predict_multi_step(self.n_predict)

        # Check predict and actual state
        # self.x_history.append(np.hstack([self.interface_pose.np_pose, self.interface_pose.np_rpy]))
        # if len(self.x_history) == self.x_history.maxlen:
        #     predicted_state = self.predict_interface_pose[-1, :]
        #     actual_state = self.x_history.pop()
        #     error = actual_state - predicted_state
        #     self.logger.info('act ' + str(actual_state))
        #     self.logger.info('prd ' + str(predicted_state))
        #     self.logger.info('err ' + str(error))

    @property
    def pose(self):
        return self.interface_pose
    
    @pose.setter
    def pose(self, pose: Pose):
        self.interface_pose = pose.copy()

    @property
    def pose_prev(self):
        return self.interface_pose_prev
    
    @pose_prev.setter
    def pose_prev(self, pose: Pose):
        self.interface_pose_prev = pose.copy()

    @property
    def vel_pose(self):
        return self.interface_vel_pose
    
    @vel_pose.setter
    def vel_pose(self, vel: torch.Tensor):
        self.interface_vel_pose = vel

    @property
    def vel_rpy(self):
        return self.interface_vel_rpy
    
    @vel_rpy.setter
    def vel_rpy(self, vel: torch.Tensor):
        self.interface_vel_rpy = vel

    @property
    def predict_pose(self):
        return self.predict_interface_pose

    @property
    def time(self):
        return self.interface_time.time
    
    @time.setter
    def time(self, time: Time):
        if isinstance(time, Time):
            self.interface_time = time
        elif isinstance(time, MSG_Time):
            self.interface_time.time = time
        else:
            self.interface_time.time = time

    @property
    def time_prev(self):
        return self.interface_time_prev.time
    
    @time_prev.setter
    def time_prev(self, time):
        if isinstance(time, Time):
            self.interface_time_prev = time
        elif isinstance(time, MSG_Time):
            self.interface_time_prev.time = time
        else:
            self.interface_time_prev.time = time

