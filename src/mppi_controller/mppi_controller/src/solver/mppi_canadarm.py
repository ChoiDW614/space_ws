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

        # Target states
        self.target_pose = Pose()

        # Import URDF for forward kinematics
        package_name = "mppi_controller"
        urdf_file_path = os.path.join(get_package_share_directory(package_name), "models", "canadarm", "Canadarm2.urdf")

        self.fk_canadarm = URDFForwardKinematics(urdf_file_path, root_link='Base_SSRMS', end_links = 'EE_SSRMS')
        # 어느게 정확한지 모르겠음
        # tf_Base_SSRMS = torch.tensor([[1.0,        0.0,       0.0, 1.0],
        #                               [0.0,       -1.0, 0.0015927, 0.0],
        #                               [0.0, -0.0015927,      -1.0, 1.5],
        #                               [0.0,        0.0,       0.0, 1.0]])
        tf_Base_SSRMS = torch.tensor([[1.0,      0.0,       0.0, 1.0],
                                      [0.0,     -1.0, -7.24e-06, 0.0],
                                      [0.0, 7.24e-06,      -1.0, 1.5],
                                      [0.0,      0.0,       0.0, 1.0]], device=self.device)
        # tf_Base_SSRMS = torch.tensor([[1.0,  0.0,  0.0, 1.0],
        #                               [0.0, -1.0,  0.0, 0.0],
        #                               [0.0,  0.0, -1.0, 1.5],
        #                               [0.0,  0.0,  0.0, 1.0]])
        self.fk_canadarm.set_mount_transformation(tf_Base_SSRMS)
        self.fk_canadarm.set_samples_and_timesteps(self.num_samples, self.num_timestep)


    def compute_control(self):
        pose_err = pose_diff(self.ee_pose, self.target_pose)
        if pose_err < 0.01:
            return

        sample = self.sampling_state()

        trajSamples = self.fk_canadarm.forward_kinematics(sample, 'EE_SSRMS')
        
        return


    def sampling_state(self):
        try:
            random_generator = torch.distributions.MultivariateNormal(loc=self.mu, covariance_matrix=self.sigma)
        except ValueError:
            self.logger.warn("Covariance_matrix not satisfy the constraint PositiveDefinite")
            self.sigma = self.reset_sigma.clone()
            random_generator = torch.distributions.MultivariateNormal(loc=self.mu, covariance_matrix=self.sigma)

        noise = random_generator.sample((self.num_timestep,)).permute(1, 0, 2) # shape (sample, timestep, joint)
        sample = noise + self._init_q.unsqueeze(0).unsqueeze(0).repeat(self.num_samples, self.num_timestep, 1)
        return sample
    











        

    def set_init_joint(self, init_joint_states):
        init_joint_states = init_joint_states.to(self.device)
    
        self._init_q = init_joint_states[:, 0]
        self._init_qdot = init_joint_states[:, 1]
        self._init_qddot = init_joint_states[:, 2]

    def set_ee_pose(self, pos, ori):
        self.ee_pose.pose = pos
        self.ee_pose.orientation = ori

    def set_target_pose(self, pos, ori):
        self.target_pose.pose = pos
        self.target_pose.orientation = ori

    def reset_sampling_weight(self):
        self.mu = self.reset_mu.clone()
        self.sigma = self.reset_sigma.clone()
    