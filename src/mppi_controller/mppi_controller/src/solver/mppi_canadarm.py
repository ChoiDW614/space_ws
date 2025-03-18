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
        torch.manual_seed(time.time_ns())
        torch.set_default_dtype(torch.float32)

        self.num_joint = 7

        # Sampling parameters
        self.num_samples = 1024
        self.num_timestep = 256

        # self.num_samples = 5
        # self.num_timestep = 3

        self.reset_mu = torch.zeros(self.num_timestep, self.num_joint, device=self.device)
        self.reset_sigma = torch.stack([torch.eye(self.num_joint, device=self.device) for _ in range(self.num_timestep)], dim=0)

        self.mu = self.reset_mu.clone()
        self.sigma = self.reset_sigma.clone()

        self.alpha_mu = 0.1
        self.alpha_sigma = 0.1

        # Manipulator states
        self.ee_pose = Pose()
        self._init_q = torch.zeros(self.num_joint, device=self.device)
        self._init_qdot = torch.zeros(self.num_joint, device=self.device)
        self._init_qddot = torch.zeros(self.num_joint, device=self.device)
        self.u_prev = torch.zeros(self.num_joint, device=self.device)

        # base control states
        self.base_pose = Pose()

        # Target states
        self.target_pose = Pose()
        self.target_pose.pose = torch.tensor([1.0, 1.0, 1.0])
        self.target_pose.orientation = torch.tensor([1.0, 0.0, 1.0, 1.0])

        # cost weight parameters
        self._tracking_pose_weight = 1.0
        self._tracking_orientation_weight = 5.0
        self._gamma = 0.95
        self._lambda = 10.0

        # Import URDF for forward kinematics
        package_name = "mppi_controller"
        urdf_file_path = os.path.join(get_package_share_directory(package_name), "models", "canadarm", "Canadarm2_w_iss.urdf")

        self.fk_canadarm = URDFForwardKinematics(urdf_file_path, root_link='Base_SSRMS', end_links = 'EE_SSRMS')

        mount_tf = torch.eye(4, device=self.device)
        mount_tf[0:3, 0:3] = euler_angles_to_matrix(torch.tensor([3.1416, 0.0, 0.0]), 'XYZ')
        mount_tf[0:3, 3] = torch.tensor([0.0, 0.0, 3.6])

        self.fk_canadarm.set_mount_transformation(mount_tf)
        self.fk_canadarm.set_samples_and_timesteps(self.num_samples, self.num_timestep)


    def compute_control_input(self):
        self.ee_pose.from_matrix(self.fk_canadarm.forward_kinematics_cpu(self._init_q, 'EE_SSRMS', self.base_pose.tf_matrix()))
        pose_err = pos_diff(self.ee_pose, self.target_pose)
        self.logger.info("pose err: " + str(round(pose_err.detach().item(), 3)))
        if pose_err < 0.01:
            self.logger.info("target reached!")
            return self.u_prev

        # sampling
        noise = self.sampling_state()

        traj_samples = self.fk_canadarm.forward_kinematics(noise, 'EE_SSRMS', self.base_pose.tf_matrix(self.device))

        tracking_cost = self.tracking_cost(traj_samples)
        terminal_cost = self.terminal_cost()
        u = self.update_control_input(noise, tracking_cost, terminal_cost)
        
        # save previous control input
        self.u_prev = u
        return u[0]


    def sampling_state(self):
        try:
            random_generator = torch.distributions.MultivariateNormal(loc=self.mu, covariance_matrix=self.sigma)
        except ValueError:
            self.logger.warn("Covariance_matrix not satisfy the constraint PositiveDefinite")
            self.sigma = self.reset_sigma.clone()
            random_generator = torch.distributions.MultivariateNormal(loc=self.mu, covariance_matrix=self.sigma)

        noise = random_generator.sample((self.num_samples,)) # shape (sample, timestep, joint)
        return noise
    

    def tracking_cost(self, sample):
        ee_sample_pose = sample[:,:,0:3,3]
        ee_sample_orientation = sample[:,:,0:3,0:3]

        diff_pose = ee_sample_pose - self.target_pose.pose.to(device=self.device)
        # diff_orientation = matrix_to_quaternion(ee_sample_orientation) - self.target_pose.orientation.to(device=self.device)

        cost_pose = torch.sum(torch.pow(diff_pose, 2), dim=2)
        # cost_orientation = torch.sum(torch.abs(diff_orientation), dim=2)

        # tracking_cost = self._tracking_pose_weight * cost_pose + self._tracking_orientation_weight * cost_orientation
        tracking_cost = self._tracking_pose_weight * cost_pose

        gamma = self._gamma ** torch.arange(self.num_timestep, device=self.device)
        tracking_cost = tracking_cost * gamma
        return tracking_cost
    

    def terminal_cost(self):
        terminal_cost = torch.zeros([self.num_samples, self.num_joint])
        terminal_cost = (self._gamma ** self.num_timestep) * torch.zeros([self.num_samples], device=self.device)
        return terminal_cost

    
    def update_control_input(self, noise, tracking_cost, terminal_cost):
        final_cost = torch.sum(tracking_cost, dim=1)
        final_cost += terminal_cost
        final_cost -= torch.min(final_cost)

        weight = torch.softmax(-final_cost / self._lambda, dim=0)

        self.mu = torch.sum(
            weight.view(self.num_samples, 1, 1) * noise, dim=0
        )

        diff = noise - self.mu.unsqueeze(0)
        update_sigma = torch.einsum(
            'n,nhr,nhc->hrc',
            weight,
            diff,
            diff
        )
        self.sigma = (1 - self.alpha_sigma) * self.sigma + self.alpha_sigma * update_sigma

        self.logger.info("sigma sum: " + str(round(torch.sum(self.sigma).detach().item(), 3)))
        
        return self.mu







        

    def set_init_joint(self, init_joint_states):
        init_joint_states = init_joint_states.to(self.device)
    
        self._init_q = init_joint_states[:, 0]
        self._init_qdot = init_joint_states[:, 1]
        self._init_qddot = init_joint_states[:, 2]

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

    def reset_sampling_weight(self):
        self.mu = self.reset_mu.clone()
        self.sigma = self.reset_sigma.clone()
    