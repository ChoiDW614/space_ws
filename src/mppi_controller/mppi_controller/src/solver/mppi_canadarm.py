import os
import math
import yaml
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
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

        # Manipulator states
        self._q = torch.zeros(self.n_action, device=self.device)
        self._qdot = torch.zeros(self.n_action, device=self.device)
        self._qddot = torch.zeros(self.n_action, device=self.device)

        self.ee_pose = Pose()
        self.eefTraj = torch.zeros((self.n_samples, self.n_horizen, 4, 4), device=self.device)

        # Action
        self.u_prev = torch.zeros((self.n_horizen, self.n_action), device=self.device)
        self.v_prev = torch.zeros((self.n_horizen, self.n_action), device=self.device)

        # Buffer
        self.buffer_size = 10
        self.weight_buffer = torch.zeros((self.buffer_size, self.n_samples), device=self.device)
        self.action_buffer = torch.zeros((self.buffer_size, self.n_samples, self.n_horizen, self.n_action), device=self.device)

        # Sampling class
        self.sample_gen = GaussianSample(self.n_horizen, self.n_action, self.buffer_size, device= self.device)

        # base control states
        self.base_pose = Pose()

        # Target states
        self.target_pose = Pose()
        self.target_pose.pose = torch.tensor([0.0, 0.0, 0.0])
        self.target_pose.orientation = torch.tensor([1.0, 0.0, 0.0, 1.0])

        # cost weight parameters
        self._tracking_pose_weight = 1.0
        self._tracking_orientation_weight = 0.1

        self._terminal_pose_weight = 10.0
        self._terminal_orientation_weight = 1.0
        
        self._gamma = 0.98
        self._lambda = 100.0

        # Import URDF for forward kinematics
        package_name = "mppi_controller"
        urdf_file_path = os.path.join(get_package_share_directory(package_name), "models", "canadarm", "Canadarm2_w_iss.urdf")

        self.fk_canadarm = URDFForwardKinematics(urdf_file_path, root_link='Base_SSRMS', end_links = 'EE_SSRMS')

        mount_tf = torch.eye(4, device=self.device)
        mount_tf[0:3, 0:3] = euler_angles_to_matrix(torch.tensor([3.1416, 0.0, 0.0]), 'XYZ')
        mount_tf[0:3, 3] = torch.tensor([0.0, 0.0, 3.6])

        self.fk_canadarm.set_mount_transformation(mount_tf)
        self.fk_canadarm.set_samples_and_timesteps(self.n_samples, self.n_horizen)

        # Log
        self.cnt = 0
        self.cost_log = SummaryWriter('runs/experiment_1')


    def compute_control_input(self):
        pose_err = self.prev_forward_kinematics()
        self.log(pose_err)
        if pose_err < 0.01:
            self.logger.info("target reached!")
            return self.u_prev

        samples = self.sample_gen.get_action(n_sample=self.n_samples, q=self._q, seed=time.time_ns())

        self.eefTraj = self.fk_canadarm.forward_kinematics(samples, 'EE_SSRMS', self.base_pose.tf_matrix(self.device))

        tracking_cost = self.tracking_cost()
        terminal_cost = self.terminal_cost()
        u = self.update_control_input(samples, tracking_cost, terminal_cost)
        
        return u
    

    def prev_forward_kinematics(self):
        self.ee_pose.from_matrix(self.fk_canadarm.forward_kinematics_cpu(self._q, 'EE_SSRMS', self.base_pose.tf_matrix()))
        pose_err = pos_diff(self.ee_pose, self.target_pose)

        # self.logger.info("pose: " + str(self.ee_pose.pose))
        self.logger.info("pose err: " + str(round(pose_err.detach().item(), 3)))
        return pose_err

    
    def tracking_cost(self):
        ee_sample_pose = self.eefTraj[:,:,0:3,3]
        ee_sample_orientation = self.eefTraj[:,:,0:3,0:3]

        diff_pose = ee_sample_pose - self.target_pose.pose.to(device=self.device)
        diff_orientation = matrix_to_euler_angles(ee_sample_orientation, "ZYX") - self.target_pose.rpy().to(device=self.device)

        cost_pose = torch.sum(torch.pow(diff_pose, 2), dim=2)
        cost_orientation = torch.sum(torch.abs(diff_orientation), dim=2)

        # tracking_cost = self._tracking_pose_weight * cost_pose + self._tracking_orientation_weight * cost_orientation
        tracking_cost = self._tracking_pose_weight * cost_pose

        gamma = self._gamma ** torch.arange(self.n_horizen, device=self.device)
        tracking_cost = tracking_cost * gamma
        return tracking_cost
    

    def terminal_cost(self):
        ee_terminal_pose = self.eefTraj[:,-1,0:3,3]
        ee_terminal_orientation = self.eefTraj[:,-1,0:3,0:3]

        diff_pose = ee_terminal_pose - self.target_pose.pose.to(device=self.device)
        diff_orientation = matrix_to_euler_angles(ee_terminal_orientation, "ZYX") - self.target_pose.rpy().to(device=self.device)

        cost_pose = torch.sum(torch.pow(diff_pose, 2), dim=1)
        cost_orientation = torch.sum(torch.abs(diff_orientation), dim=1)

        terminal_cost = self._terminal_pose_weight * cost_pose + self._terminal_orientation_weight * cost_orientation
        terminal_cost = self._terminal_pose_weight * cost_pose

        terminal_cost = (self._gamma ** self.n_horizen) * terminal_cost
        return terminal_cost

    
    def update_control_input(self, samples, tracking_cost, terminal_cost):
        final_cost = torch.sum(tracking_cost, dim=1)
        final_cost += terminal_cost

        final_cost -= torch.min(final_cost)

        weight = torch.softmax(-final_cost / self._lambda, dim=0)

        u = torch.sum(weight.view(self.n_samples, 1, 1) * samples, dim=0)

        self.u_prev = u.clone()
        self.u = u.clone()

        self.roll_buffer(weight, samples)
        self.sample_gen.update_distribution(weight=self.weight_buffer, action=self.action_buffer)

        return u[0,:]
    


    def log(self, cost_data):
        self.cost_log.add_scalar('cost', cost_data)
        self.cnt += 1


    def roll_buffer(self, weight, action):
        self.weight_buffer = torch.roll(self.weight_buffer, shifts=1, dims=0)
        self.action_buffer = torch.roll(self.action_buffer, shifts=1, dims=0)

        self.weight_buffer[0,:] = weight
        self.action_buffer[0,:,:] = action
        return


    def set_init_joint(self, init_joint_states):
        init_joint_states = init_joint_states.to(self.device)
    
        self._q = init_joint_states[:, 0]
        self._qdot = init_joint_states[:, 1]
        self._qddot = init_joint_states[:, 2]
        return

    def set_ee_pose(self, pos, ori):
        self.ee_pose.pose = pos
        self.ee_pose.orientation = ori
        return

    def set_base_pose(self, pos, ori):
        self.base_pose.pose = pos
        self.base_pose.orientation = ori
        return

    def set_target_pose(self, pos, ori):
        self.target_pose.pose = pos
        self.target_pose.orientation = ori
        return
    
    def set_target_pose(self, pose: Pose):
        self.target_pose.pose = pose.pose
        self.target_pose.orientation = pose.orientation
        return
