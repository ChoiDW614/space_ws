import os
import math
import yaml
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

import pinocchio as pin

from rclpy.logging import get_logger
from ament_index_python.packages import get_package_share_directory

from mppi_controller.src.solver.sampling.gaussian_noise import GaussianSample
from mppi_controller.src.robot.urdfFks.urdfFk import URDFForwardKinematics
from mppi_controller.src.utils.pose import Pose, pose_diff, pos_diff
from mppi_controller.src.utils.rotation_conversions import matrix_to_quaternion, euler_angles_to_matrix, matrix_to_euler_angles


class MPPI():
    def __init__(self):
        self.logger = get_logger("MPPI")

        # torch env
        os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info('Device: ' + self.device.type)
        torch.set_default_dtype(torch.float32)

        # Sampling parameters
        self.n_action = 7
        self.n_samples = 1024
        self.n_horizen = 32
        self.dt = 0.01

        # Sampling class
        self.sample_gen = GaussianSample(self.n_horizen, self.n_action, device= self.device)

        # Manipulator states
        self._q = torch.zeros(self.n_action, device=self.device)
        self._qdot = torch.zeros(self.n_action, device=self.device)
        self._qddot = torch.zeros(self.n_action, device=self.device)

        self.ee_pose = Pose()
        self.eefTraj = torch.zeros((self.n_samples, self.n_horizen, 4, 4), device=self.device)

        self.qTraj = torch.zeros((self.n_samples, self.n_horizen, self.n_action), device=self.device)
        self.vTraj = torch.zeros((self.n_samples, self.n_horizen, self.n_action), device=self.device)

        # Action
        self.u_prev = torch.zeros((self.n_horizen, self.n_action), device=self.device)
        self.v_prev = torch.zeros((self.n_horizen, self.n_action), device=self.device)

        # base control states
        self.base_pose = Pose()

        # Target states
        self.target_pose = Pose()
        self.target_pose.pose = torch.tensor([1.0, 1.0, 1.0])
        self.target_pose.orientation = torch.tensor([1.0, 0.0, 1.0, 1.0])

        # cost weight parameters
        self._tracking_pose_weight = 1.0
        self._tracking_orientation_weight = 0.1
        self._gamma = 0.95
        self._lambda = 500.0

        # Import URDF for forward kinematics
        package_name = "mppi_controller"
        urdf_file_path = os.path.join(get_package_share_directory(package_name), "models", "canadarm", "Canadarm2_w_iss.urdf")

        self.fk_canadarm = URDFForwardKinematics(urdf_file_path, root_link='Base_SSRMS', end_links = 'EE_SSRMS')

        mount_tf = torch.eye(4, device=self.device)
        mount_tf[0:3, 0:3] = euler_angles_to_matrix(torch.tensor([3.1416, 0.0, 0.0]), 'XYZ')
        mount_tf[0:3, 3] = torch.tensor([0.0, 0.0, 3.6])

        self.fk_canadarm.set_mount_transformation(mount_tf)
        self.fk_canadarm.set_samples_and_timesteps(self.n_samples, self.n_horizen)


    def compute_control_input(self):
        self.ee_pose.from_matrix(self.fk_canadarm.forward_kinematics_cpu(self._q, 'EE_SSRMS', self.base_pose.tf_matrix()))
        pose_err = pos_diff(self.ee_pose, self.target_pose)
        self.logger.info("pose err: " + str(round(pose_err.detach().item(), 3)))
        if pose_err < 0.01:
            self.logger.info("target reached!")
            return self.u_prev

        # sampling
        samples = self.sample_gen.get_action(n_sample=self.n_samples, mu=self.u_prev, seed=time.time_ns())

        traj_samples = self.fk_canadarm.forward_kinematics(samples, 'EE_SSRMS', self.base_pose.tf_matrix(self.device))

        tracking_cost = self.tracking_cost(traj_samples)
        terminal_cost = self.terminal_cost()
        u = self.update_control_input(samples, tracking_cost, terminal_cost)
        
        return u

    
    def tracking_cost(self, sample):
        ee_sample_pose = sample[:,:,0:3,3]
        ee_sample_orientation = sample[:,:,0:3,0:3]

        diff_pose = ee_sample_pose - self.target_pose.pose.to(device=self.device)
        diff_orientation = matrix_to_euler_angles(ee_sample_orientation, "ZYX") - self.target_pose.rpy().to(device=self.device)

        cost_pose = torch.sum(torch.pow(diff_pose, 2), dim=2)
        cost_orientation = torch.sum(torch.abs(diff_orientation), dim=2)

        tracking_cost = self._tracking_pose_weight * cost_pose + self._tracking_orientation_weight * cost_orientation

        gamma = self._gamma ** torch.arange(self.n_horizen, device=self.device)
        tracking_cost = tracking_cost * gamma
        return tracking_cost
    

    def terminal_cost(self):
        terminal_cost = torch.zeros([self.n_samples, self.n_action])
        terminal_cost = (self._gamma ** self.n_horizen) * torch.zeros([self.n_samples], device=self.device)
        return terminal_cost

    
    def update_control_input(self, samples, tracking_cost, terminal_cost):
        final_cost = torch.sum(tracking_cost, dim=1)
        final_cost += terminal_cost

        final_cost -= torch.min(final_cost)

        weight = torch.softmax(-final_cost / self._lambda, dim=0)

        u = torch.sum(
            weight.view(self.n_samples, 1, 1) * samples, dim=0
        )
        self.u_prev = u.clone()
        self.u = u.clone()

        # self.mu = torch.sum(
        #     weight.view(self.n_samples, 1, 1) * samples, dim=0
        # )

        # diff = samples - self.mu.unsqueeze(0)
        # update_sigma = torch.einsum(
        #     'n,nhr,nhc->hrc',
        #     weight,
        #     diff,
        #     diff
        # )
        # self.sigma = (1 - self.alpha_sigma) * self.sigma + self.alpha_sigma * update_sigma
        # self.logger.info("sigma sum: " + str(round(torch.sum(self.sigma).detach().item(), 3)))
        
        return u[0,:]







        

    def set_init_joint(self, init_joint_states):
        init_joint_states = init_joint_states.to(self.device)
    
        self._q = init_joint_states[:, 0]
        self._qdot = init_joint_states[:, 1]
        self._qddot = init_joint_states[:, 2]

    def set_ee_pose(self, pos, ori):
        self.ee_pose.pose = pos
        self.ee_pose.orientation = ori

    def set_base_pose(self, pos, ori):
        self.base_pose.pose = pos
        self.base_pose.orientation = ori

    def set_target_pose(self, pos, ori):
        self.target_pose.pose = pos
        self.target_pose.orientation = ori
    
    def set_target_pose(self, pose: Pose):
        self.target_pose.pose = pose.pose
        self.target_pose.orientation = pose.orientation
