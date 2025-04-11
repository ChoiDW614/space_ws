import os
import math
import yaml
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.multivariate_normal import MultivariateNormal

from rclpy.logging import get_logger
from ament_index_python.packages import get_package_share_directory

from mppi_solver.src.solver.sampling.gaussian_noise import GaussianSample
from mppi_solver.src.solver.cost.pose_cost import PoseCost

from mppi_solver.src.robot.urdfFks.urdfFk import URDFForwardKinematics
from mppi_solver.src.robot.urdfFks.transformation_matrix import transformation_matrix_from_xyzrpy_cpu

from mppi_solver.src.utils.pose import Pose, pose_diff, pos_diff
from mppi_solver.src.utils.rotation_conversions import euler_angles_to_matrix, matrix_to_euler_angles


class MPPI():
    def __init__(self, isBaseMoving):
        self.logger = get_logger("MPPI")

        # torch env
        os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info('Device: ' + self.device.type)
        torch.set_default_dtype(torch.float32)

        # Sampling parameters
        self.isBaseMoving = isBaseMoving
        if self.isBaseMoving:
            self.n_action = 13
            self.n_manipulator_dof = 7
            self.n_mobile_dof = 6
            self.n_samples = 1024
            self.n_horizen = 64
            self.dt = 0.01
        else:
            self.n_action = 7
            self.n_manipulator_dof = 7
            self.n_mobile_dof = 0
            self.n_samples = 1024
            self.n_horizen = 64
            self.dt = 0.01

        # Manipulator states
        self._q = torch.zeros(self.n_action, device=self.device)
        self._qdot = torch.zeros(self.n_action, device=self.device)
        self._qddot = torch.zeros(self.n_action, device=self.device)

        self.ee_pose = Pose()
        self.eefTraj = torch.zeros((self.n_samples, self.n_horizen, 4, 4), device=self.device)

        # Action
        self.u = torch.zeros((self.n_horizen, self.n_action), device=self.device)
        self.u_prev = torch.zeros_like(self._qddot, device=self.device)
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
        self.target_pose.orientation = torch.tensor([0.0, 0.0, 0.0, 1.0])
        self.predict_target_pose = torch.zeros((self.n_horizen, 6))
        
        self.pose_cost = PoseCost(self.n_horizen, self.device)
        self._lambda = 0.01

        # Import URDF for forward kinematics
        package_name = "mppi_solver"
        if self.isBaseMoving:
            urdf_file_path = os.path.join(get_package_share_directory(package_name), "models", "canadarm", "floating_canadarm.urdf")
        else:
            urdf_file_path = os.path.join(get_package_share_directory(package_name), "models", "canadarm", "Canadarm2_w_iss.urdf")

        self.fk_canadarm = URDFForwardKinematics(urdf_file_path, root_link='Base_SSRMS', end_links = 'EE_SSRMS')

        mount_tf = torch.eye(4, device=self.device)
        mount_tf[0:3, 0:3] = euler_angles_to_matrix(torch.tensor([3.1416, 0.0, 0.0]), 'XYZ')
        mount_tf[0:3, 3] = torch.tensor([0.0, 0.0, 3.6])

        self.fk_canadarm.set_mount_transformation(mount_tf)
        self.fk_canadarm.set_samples_and_timesteps(self.n_samples, self.n_horizen, self.n_mobile_dof)

        # Log
        # self.cnt = 0
        # log_root = 'src/mppi_solver/mppi_solver/runs'
        # if not os.path.exists(log_root):
        #     os.makedirs(log_root)

        # log_path = os.path.join(log_root, datetime.now().strftime("%Y%m%d-%H%M%S"))
        # self.cost_log = SummaryWriter(log_path)


    def compute_control_input(self):
        pose_err = self.prev_forward_kinematics()

        if pose_err < 0.5:
            self.sample_gen.sigma = torch.eye(self.n_action, device=self.device)
            self.sample_gen.sigma[:4, :4] *= 0.5
            self.sample_gen.sigma[4:, 4:] *= 0.5
            self.pose_cost._tracking_pose_weight = 3.0
            self.pose_cost._tracking_orientation_weight = 0.7         
        else :
            self.sample_gen.sigma = torch.eye(self.n_action, device=self.device)
            self.sample_gen.sigma[:4, :4] *= 2.0
            self.sample_gen.sigma[4:, 4:] *= 2.0
            self.pose_cost._tracking_pose_weight = 10.0
            self.pose_cost._tracking_orientation_weight = 0.3

        if pose_err < 0.01:
            self.logger.info("target reached!")
            return self.u_prev, self._q, self._qdot

        # samples = self.sample_gen.get_action(n_sample=self.n_samples, q=self._q, seed=time.time_ns())
        noise = self.sample_gen.simple_sampling(n_sample=self.n_samples, seed=time.time_ns())
        uSamples = self.u + noise
        qSamples = self.getSampleJoint(uSamples)

        self.eefTraj = self.fk_canadarm.forward_kinematics(qSamples, 'EE_SSRMS', 'Base_SSRMS', self.base_pose.tf_matrix(self.device), base_movement=self.isBaseMoving)

        tracking_cost = self.pose_cost.tracking_cost(self.eefTraj, self.target_pose)
        # tracking_cost = self.pose_cost.predict_tracking_cost(self.eefTraj, self.predict_target_pose)
        terminal_cost = self.pose_cost.terminal_cost(self.eefTraj, self.target_pose)
        u = self.update_control_input(noise, tracking_cost, terminal_cost)
        
        return u
    

    def prev_forward_kinematics(self):
        tf_base = transformation_matrix_from_xyzrpy_cpu(q=self._q)

        self.ee_pose.from_matrix(self.fk_canadarm.forward_kinematics_cpu(self._q[self.n_mobile_dof:], 'EE_SSRMS', 'Base_SSRMS', self.base_pose.tf_matrix(), base_movement=False))
        pose_err = pos_diff(self.ee_pose, self.target_pose)

        # self.logger.info("pose2: " + str(self.ee_pose.pose))
        # self.logger.info("pose err: " + str(round(pose_err.detach().item(), 3)))
        return pose_err

    
    def update_control_input(self, samples, tracking_cost, terminal_cost):
        final_cost = torch.zeros((self.n_samples), device=self.device)
        final_cost += torch.sum(tracking_cost, dim=1)

        # final_cost += torch.sum(filter_cost, dim=1)
        # final_cost += terminal_cost
        rho = final_cost.min()
        scaledS = (-1.0 / self._lambda) * (final_cost - rho)
        eta = torch.exp(scaledS).sum()
        weight = torch.exp(scaledS)/eta
        # final_cost -= torch.min(final_cost)
        # weight = torch.softmax(-final_cost / self._lambda, dim=0)

        w_epsilon = torch.sum(weight.view(self.n_samples, 1, 1) * samples, dim=0)
        w_epsilon = self.moving_average_filter(xx = w_epsilon, window_size= 10)
        u = self.u_prev.clone() + w_epsilon

        self.u_prev = u[0,:].clone()
        self.u = u.clone()

        # self.roll_buffer(weight, samples)
        # self.sample_gen.update_distribution(weight=self.weight_buffer, action=self.action_buffer)
        # self.u_prev *= 1.0

        # TEST
        v = self._qdot.clone() + self.u_prev * self.dt
        q = self._q.clone() + self._qdot.clone() * self.dt + self.u_prev * self.dt **2 * 0.5

        return self.u_prev ,q, v
        
    
    def getSampleJoint(self, samples):
        # samples: (n_sample, n_horizon, n_action)
        n_sample, n_horizon, n_action = samples.shape

        # 초기 속도와 위치 확장
        qdot0 = self._qdot.unsqueeze(0).unsqueeze(0).expand(n_sample, 1, n_action)  # (n_sample, 1, n_action)
        q0 = self._q.unsqueeze(0).unsqueeze(0).expand(n_sample, 1, n_action)        # (n_sample, 1, n_action)

        # 누적 속도 계산: v[i] = v[i-1] + a[i] * dt
        v = torch.cumsum(samples * self.dt, dim=1) + qdot0  # (n_sample, n_horizon, n_action)

        # 이전 속도: [v0, v0+..., ..., v_{N-1}]
        v_prev = torch.cat([qdot0, v[:, :-1, :]], dim=1)  # (n_sample, n_horizon, n_action)

        # 누적 위치 계산: q[i] = q[i-1] + v[i-1] * dt + 0.5 * a[i] * dt^2
        dq = v_prev * self.dt + 0.5 * samples * self.dt**2
        q = torch.cumsum(dq, dim=1) + q0

        return q


    def log(self, cost_data):
        self.cost_log.add_scalar('cost', cost_data)
        self.cnt += 1


    def roll_buffer(self, weight, action):
        self.weight_buffer = torch.roll(self.weight_buffer, shifts=1, dims=0)
        self.action_buffer = torch.roll(self.action_buffer, shifts=1, dims=0)

        self.weight_buffer[0,:] = weight
        self.action_buffer[0,:,:] = action
        return


    def set_joint(self, joint_states):
        joint_states = joint_states.to(self.device)
    
        self._q = joint_states[:, 0]
        self._qdot = joint_states[:, 1]
        self._qddot = joint_states[:, 2]
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

    def set_predict_target_pose(self, pose: np.ndarray):
        if isinstance(pose, torch.Tensor):
            self.predict_target_pose = pose.to(self.device)
        else:
            self.predict_target_pose = torch.from_numpy(pose).to(self.device)
        return
    
    def moving_average_filter(self, xx: torch.Tensor, window_size: int) -> torch.Tensor:
        """
        Apply moving average filter for smoothing input sequence.

        Args:
            xx (torch.Tensor): Input tensor of shape (N, dim), where N is the sequence length.
            window_size (int): Size of the moving average window.

        Returns:
            torch.Tensor: Smoothed tensor of the same shape as xx.
        """
        N, dim = xx.shape  # N: sequence length, dim: number of dimensions

        # Reshape xx to (batch_size=1, channels=dim, sequence_length=N)
        xx = xx.t().unsqueeze(0)  # Shape: (1, dim, N)

        # Create the filter weights for convolution
        b = torch.ones((dim, 1, window_size), device=xx.device) / window_size  # Shape: (dim, 1, window_size)

        # Adjust padding to ensure output length matches input length
        padding_left = window_size // 2
        padding_right = window_size - padding_left - 1

        xx_padded = torch.nn.functional.pad(xx, (padding_left, padding_right), mode='reflect')  # Shape: (1, dim, N + padding_left + padding_right)

        # Perform convolution using groups to apply the filter independently to each dimension
        xx_mean = torch.nn.functional.conv1d(xx_padded, b, groups=dim)  # Shape: (1, dim, N)

        # Reshape back to (N, dim)
        xx_mean = xx_mean.squeeze(0).t()  # Shape: (N, dim)

        # Edge correction to compensate for the convolution effect at the boundaries
        n_conv = (window_size + 1) // 2  # Equivalent to math.ceil(window_size / 2)

        # Correct the first element
        factor0 = window_size / n_conv
        xx_mean[0, :] *= factor0

        if n_conv > 1:
            # Indices for the rest of the elements to correct
            i_range = torch.arange(1, n_conv, device=xx.device)  # [1, 2, ..., n_conv - 1]

            # Factors for the beginning of the sequence
            factor_start = window_size / (i_range + n_conv)  # Shape: (n_conv - 1,)
            xx_mean[1:n_conv, :] *= factor_start.unsqueeze(1)  # Apply factors to xx_mean[1:n_conv, :]

            # Factors for the end of the sequence
            denom_end = i_range + n_conv - (window_size % 2)
            factor_end = window_size / denom_end  # Shape: (n_conv - 1,)
            xx_mean[-n_conv+1:, :] *= factor_end.flip(0).unsqueeze(1)  # Apply factors to xx_mean[-(n_conv-1):, :]

        return xx_mean
