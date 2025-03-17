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
from mppi_controller.src.utils.pose import Pose, pose_diff
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

        self.reset_mu = torch.zeros(self.num_samples, self.num_joint, device=self.device)
        self.reset_sigma = torch.stack([torch.eye(self.num_joint, device=self.device) for _ in range(self.num_samples)], dim=0)

        self.mu = self.reset_mu.clone()
        self.sigma = self.reset_sigma.clone()

        # Manipulator states
        self.ee_pose = Pose()
        self._init_q = torch.zeros(7, device=self.device)
        self._init_qdot = torch.zeros(7, device=self.device)
        self._init_qddot = torch.zeros(7, device=self.device)
        self.u_prev = torch.zeros(self.num_joint)

        # base control states
        self.base_pose = Pose()

        # Target states
        self.target_pose = Pose()
        self.target_pose.pose = torch.tensor([1.0, 1.0, 1.0])
        self.target_pose.orientation = torch.tensor([1.0, 0.0, 1.0, 1.0])

        # cost weight parameters
        self.tracking_pose_weight = 1.0
        self.tracking_orientation_weight = 5.0


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
        ee_pose = self.fk_canadarm.forward_kinematics_cpu(self._init_q, 'EE_SSRMS', self.base_pose.tf_matrix())
        self.ee_pose.from_matrix(ee_pose)
        pose_err = pose_diff(self.ee_pose, self.target_pose)
        if pose_err < 0.01:
            self.logger.info("target reached!")
            return

        sample = self.sampling_state()

        traj_samples = self.fk_canadarm.forward_kinematics(sample, 'EE_SSRMS', self.base_pose.tf_matrix(self.device))

        cost = self.tracking_cost(traj_samples)
        u = self.update_control_input(sample, cost)
        
        return u


    def sampling_state(self):
        try:
            random_generator = torch.distributions.MultivariateNormal(loc=self.mu, covariance_matrix=self.sigma)
        except ValueError:
            self.logger.warn("Covariance_matrix not satisfy the constraint PositiveDefinite")
            self.sigma = self.reset_sigma.clone()
            random_generator = torch.distributions.MultivariateNormal(loc=self.mu, covariance_matrix=self.sigma)

        noise = random_generator.sample((self.num_timestep,)).permute(1, 0, 2) # shape (sample, timestep, joint)
        sample = noise + self._init_q.expand(self.num_samples, self.num_timestep, self.num_joint).clone()
        return sample
    

    def tracking_cost(self, sample):
        ee_sample_pose = sample[:,:,0:3,3]
        ee_sample_orientation = sample[:,:,0:3,0:3]

        diff_pose = ee_sample_pose - self.target_pose.pose.to(device=self.device)
        diff_orientation = matrix_to_quaternion(ee_sample_orientation) - self.target_pose.orientation.to(device=self.device)

        cost_pose = torch.sum(torch.pow(diff_pose, 2), dim=2)
        cost_orientation = torch.sum(torch.abs(diff_orientation), dim=2)

        tracking_cost = self.tracking_pose_weight * cost_pose + self.tracking_orientation_weight * cost_orientation

        # terminal cost
        # tracking_cost[:,-1] += self.tracking_pose_weight * cost_pose + self.tracking_orientation_weight * cost_orientation

        return tracking_cost

    
    def update_control_input(self, sample, cost):
        



        u_input = torch.zeros([self.num_timestep, self.num_joint])

        # for t in range(self.num_timestep):
        #     for k in range(self.num_samples):
        #         u_input[t, k] += cost[:k] * sample[k, t]

        # self.u_prev += u_input

        return self.u_prev









        

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

    def reset_sampling_weight(self):
        self.mu = self.reset_mu.clone()
        self.sigma = self.reset_sigma.clone()
    