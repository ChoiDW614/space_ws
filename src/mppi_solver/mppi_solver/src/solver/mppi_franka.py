import os
import math
import yaml
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
from torch.distributions.multivariate_normal import MultivariateNormal

from rclpy.logging import get_logger
from ament_index_python.packages import get_package_share_directory

from mppi_solver.src.solver.sampling.gaussian_noise import GaussianSample
from mppi_solver.src.robot.urdfFks.urdfFk import URDFForwardKinematics
from mppi_solver.src.utils.pose import Pose, pose_diff, pos_diff
from mppi_solver.src.utils.rotation_conversions import euler_angles_to_matrix, matrix_to_euler_angles

from mppi_solver.src.robot.urdfFks.transformation_matrix import transformation_matrix_from_xyzrpy_cpu


class FrankaKinematics:
    def __init__(self,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu',  # specify device

                 ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.DH_params = torch.tensor(
            [
                [0, 0.333, 0.0],
                [0.0, 0.0, -np.pi/2],
                [0.0, 0.316, np.pi/2],
                [0.0825,0.0, np.pi/2],
                [-0.0825, 0.384, -np.pi/2],
                [0.0, 0.0, np.pi/2],
                [0.088, 0.0, np.pi/2]
            ],
            dtype=torch.float32,
            device=self.device
        )
        # self.DH_params = np.array(
        #     [
        #         [0, 0.333, 0.0],
        #         [0.0, 0.0, -np.pi/2],
        #         [0.0, 0.316, np.pi/2],
        #         [0.0825,0.0, np.pi/2],
        #         [-0.0825, 0.384, -np.pi/2],
        #         [0.0, 0.0, np.pi/2],
        #         [0.088, 0.0, np.pi/2]
        #     ]
        # )

    def computeMatrix(self, params, theta):       
        # params : N by 3 
        # theta : N
        a, d, alpha = (params[:,0], params[:,1], params[:,2])




        mat = torch.stack([
            torch.stack([torch.cos(theta), -torch.sin(theta), torch.zeros_like(theta), a], dim=-1),
            torch.stack([
                torch.sin(theta) * torch.cos(alpha), torch.cos(theta) * torch.cos(alpha), -torch.sin(alpha), -d * torch.sin(alpha)
            ], dim=-1),
            torch.stack([
                torch.sin(theta) * torch.sin(alpha), torch.cos(theta) * torch.sin(alpha), torch.cos(alpha), d * torch.cos(alpha)
            ], dim=-1),
            torch.stack([torch.zeros_like(theta), torch.zeros_like(theta), torch.zeros_like(theta), torch.ones_like(theta)], dim=-1)
        ], dim=-2)  # 최종 shape: (batch, 4, 4)

        # print("Mat Size :", mat.shape)

        # mat = torch.stack([
        #         torch.stack([torch.cos(theta), -torch.sin(theta), torch.zeros_like(theta), a_expanded], dim=-1),
        #         torch.stack([torch.sin(theta) * torch.cos(alpha_expanded), torch.cos(theta) * torch.cos(alpha_expanded), -torch.sin(alpha_expanded), -d_expanded * torch.sin(alpha_expanded)], dim=-1),
        #         torch.stack([torch.sin(theta) * torch.sin(alpha_expanded), torch.cos(theta) * torch.sin(alpha_expanded), torch.cos(alpha_expanded), d_expanded * torch.cos(alpha_expanded)], dim=-1),
        #         torch.stack([torch.zeros_like(theta), torch.zeros_like(theta), torch.zeros_like(theta), torch.ones_like(theta)], dim=-1)
        #         ], dim=-2)
    
        return mat
    

    # def computeMatrix2(self, params, theta):       
    #     # params : N by 3 
    #     # theta : N
    #     print("Theta ",theta.shape)
    #     print("params ",params.shape)
        
    #     a, d, alpha = (params[0], params[1], params[2])




    #     # mat = torch.stack([
    #     #     torch.stack([torch.cos(theta), -torch.sin(theta), torch.zeros_like(theta), a], dim=-1),
    #     #     torch.stack([
    #     #         torch.sin(theta) * torch.cos(alpha), torch.cos(theta) * torch.cos(alpha), -torch.sin(alpha), -d * torch.sin(alpha)
    #     #     ], dim=-1),
    #     #     torch.stack([
    #     #         torch.sin(theta) * torch.sin(alpha), torch.cos(theta) * torch.sin(alpha), torch.cos(alpha), d * torch.cos(alpha)
    #     #     ], dim=-1),
    #     #     torch.stack([torch.zeros_like(theta), torch.zeros_like(theta), torch.zeros_like(theta), torch.ones_like(theta)], dim=-1)
    #     # ], dim=-2)  # 최종 shape: (batch, 4, 4)

    #     # print("Mat Size :", mat.shape)

    #     mat = torch.stack([
    #             torch.stack([torch.cos(theta), -torch.sin(theta), torch.zeros_like(theta), a], dim=-1),
    #             torch.stack([torch.sin(theta) * torch.cos(alpha), torch.cos(theta) * torch.cos(alpha), -torch.sin(alpha), -d * torch.sin(alpha)], dim=-1),
    #             torch.stack([torch.sin(theta) * torch.sin(alpha), torch.cos(theta) * torch.sin(alpha), torch.cos(alpha), d * torch.cos(alpha)], dim=-1),
    #             torch.stack([torch.zeros_like(theta), torch.zeros_like(theta), torch.zeros_like(theta), torch.ones_like(theta)], dim=-1)
    #             ], dim=-2)
    
    #     return mat


    def computeEEFKinematics(self, q):
        sample_num = q.shape[0]

        mat = torch.eye(4).unsqueeze(0).expand(sample_num,-1,-1)

        joint_num = 7

        p = torch.zeros((sample_num, joint_num,3), device=self.device)
        z = torch.zeros((sample_num, joint_num,3), device=self.device)
        t = torch.zeros((sample_num, 6,3), device=self.device)
        J = torch.zeros((sample_num, 6,7), device=self.device)


        for i in range(joint_num):
            params = self.DH_params[i,:]
            params_expanded = params.unsqueeze(0).expand(sample_num, -1)

            theta = q[:,i]
            if i==0:
                # p[:,0,:] = torch.tensor([0.0, 0.0, params[1]],dtype=torch.float32, device=self.device).unsqueeze(0).expand(sample_num,-1)
                # t[:,0,:] = torch.tensor([0,0,1],dtype=torch.float32, device=self.device).unsqueeze(0).expand(sample_num,-1)
                mat = self.computeMatrix(params_expanded, theta)

            else:
                mat =  torch.bmm(mat, self.computeMatrix(params_expanded, theta))
                
                # print(mat.shape)
                # p[:, i, :] = torch.stack([mat[:, 0, -1], mat[:, 1, -1], mat[:, 2, -1]], dim=-1)
                # t[:, i, :] = torch.stack([mat[:, 0, 2], mat[:, 1, 2], mat[:, 2, 2]], dim=-1)       

        # for i in range(0,t.shape[0]):
        #     t[i,0] = p[-1,0] - p[i,0]
        #     t[i,1] = p[-1,1] - p[i,1]
        #     t[i,2] = p[-1,2] - p[i,2]                

        #     J[:3,i] = torch.tensor([z[i,1] * t[i,2] - z[i,2]* t[i,1],
        #                         z[i,2] * t[i,0] - z[i,0]* t[i,2],
        #                         z[i,0] * t[i,1] - z[i,1]* t[i,0],], 
        #                         dtype=torch.float32,
        #                         device=self.device)

        #     J[3:, i] = torch.tensor([t[i,0], t[i,1], t[i,2]],dtype=torch.float32, device=self.device)            

        return mat, J
    


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
        self.n_manipulator_dof = 7
        self.n_mobile_dof = 0
        self.n_samples = 256
        self.n_horizen = 64
        self.dt = 0.01

        # Manipulator states
        self._q = torch.zeros(self.n_action, device=self.device)
        self._q = torch.tensor([0.0, -0.78, 0.0, -2.35, 0.0, 1.57, 0.0], device = self.device)
        self._qdot = torch.zeros(self.n_action, device=self.device)
        self._qddot = torch.zeros(self.n_action, device=self.device)

        self.ee_pose = Pose()
        self.eefTraj = torch.zeros((self.n_samples, self.n_horizen, 4, 4), device=self.device)

        # Action
        self.u = torch.zeros((self.n_horizen, self.n_action), device=self.device)
        self.u_prev = self._qddot.clone() # qddot Sampling
        self.v_prev = torch.zeros((self.n_horizen, self.n_action), device=self.device)

        # Buffer
        self.buffer_size = 10
        self.weight_buffer = torch.zeros((self.buffer_size, self.n_samples), device=self.device)
        self.action_buffer = torch.zeros((self.buffer_size, self.n_samples, self.n_horizen, self.n_action), device=self.device)

        # Sampling class
        self.sample_gen = GaussianSample(self.n_horizen, self.n_action, self.buffer_size, device= self.device)

        # base control states
        self.base_pose = Pose()
        posWorld = torch.tensor([0.0,0.0,0.0])
        oriWorld = torch.tensor([0.0, 0.0, 0.0, 1.0])
        self.base_pose.pose = posWorld
        self.base_pose.orientation = oriWorld


        # Target states
        self.target_pose = Pose()
        self.target_pose.pose = torch.tensor([0.1, 0.3, 0.5])
        self.target_pose.orientation = torch.tensor([1.0, 0.0, 0.0, 0.0])

        # cost weight parameters
        self._tracking_pose_weight = 10.0
        self._tracking_orientation_weight = 0.25

        self._terminal_pose_weight = 10.0
        self._terminal_orientation_weight = 1.0
        
        self._gamma = 0.9
        self._lambda = 0.1

        self.sigma = torch.eye(self.n_action, device=self.device)
        self.sigma[:4, :4] *= 4.0
        self.sigma[4:, 4:] *= 3.0


        self.kinematics = FrankaKinematics(self.device)


        # Import URDF for forward kinematics
        package_name = "mppi_solver"
        urdf_file_path = os.path.join(get_package_share_directory(package_name), "models", "franka", "franka.urdf")

        self.fk_franka = URDFForwardKinematics(urdf_file_path, root_link='panda_link0', end_links = 'panda_link7')

        # mount_tf = torch.eye(4, device=self.device)
        # mount_tf[0:3, 0:3] = euler_angles_to_matrix(torch.tensor([3.1416, 0.0, 0.0]), 'XYZ')
        # mount_tf[0:3, 3] = torch.tensor([0.0, 0.0, 3.6])

        # self.fk_canadarm.set_mount_transformation(mount_tf)
        self.fk_franka.set_mount_transformation_deivce(self.device)
        self.fk_franka.set_samples_and_timesteps(self.n_samples, self.n_horizen, self.n_mobile_dof)

        # Log
        self.cnt = 0
        log_root = 'src/mppi_solver/mppi_solver/runs'
        if not os.path.exists(log_root):
            os.makedirs(log_root)

        log_path = os.path.join(log_root, datetime.now().strftime("%Y%m%d-%H%M%S"))
        # self.cost_log = SummaryWriter(log_path)


    # TEST 2025.04.03 #
    def sampling(self):
        # Size : [N, T, DoF]
        torch.manual_seed(time.time_ns())
        gaussian = torch.randn(self.n_samples, self.n_horizen, self.n_action, device= self.device) 
        self.sigma_matrix = self.sigma.expand(self.n_samples, self.n_horizen, -1, -1)
        noise = torch.matmul(gaussian.unsqueeze(-2), self.sigma_matrix).squeeze(-2)

        return noise


    # TEST END #

    def compute_control_input(self):
        pose_err = self.prev_forward_kinematics()

        if pose_err < 0.01:
            self.logger.info("target reached!")
            return self.u_prev, self._q, self._qdot

        # 일단 Tracking 만 넣는다 가정하면
        noise = self.sampling()
        uSamples = self.u + noise
        qSamples = self.getSampleJoint(uSamples)

        # self.computeTraj(q0, v0, noise)
        self.eefTraj = self.fk_franka.forward_kinematics(qSamples, 'panda_link7', 'panda_link0', self.base_pose.tf_matrix(self.device), base_movement=False)

        tracking_cost = self.tracking_cost()
        terminal_cost = self.terminal_cost()
        u, q, v = self.update_control_input(noise, tracking_cost, terminal_cost)

        return u, q ,v
    

    def prev_forward_kinematics(self):
        tf_base = transformation_matrix_from_xyzrpy_cpu(q=self._q)

        self.ee_pose.from_matrix(self.fk_franka.forward_kinematics_cpu(self._q[self.n_mobile_dof:], 'panda_link7', 'panda_link0', self.base_pose.tf_matrix(), base_movement=False))

        pose_err = pos_diff(self.ee_pose, self.target_pose)

        # self.logger.info("pose2: " + str(self.ee_pose.pose))
        # self.logger.info("pose err: " + str(round(pose_err.detach().item(), 3)))
        return pose_err

    
    def tracking_cost(self):
        ee_sample_pose = self.eefTraj[:,:,0:3,3]
        ee_sample_orientation = self.eefTraj[:,:,0:3,0:3]

        


        diff_pose = ee_sample_pose - self.target_pose.pose.to(device=self.device)
        # print("Diff Pose :",diff_pose)
        
        diff_orientation = matrix_to_euler_angles(ee_sample_orientation, "ZYX") - self.target_pose.rpy.to(device=self.device)

        cost_pose = torch.sum(torch.pow(diff_pose, 2), dim=2)
        cost_orientation = torch.sum(torch.abs(diff_orientation), dim=2)

        tracking_cost = self._tracking_pose_weight * cost_pose + self._tracking_orientation_weight * cost_orientation
        # tracking_cost = self._tracking_pose_weight * cost_pose

        gamma = self._gamma ** torch.arange(self.n_horizen, device=self.device)
        tracking_cost = tracking_cost * gamma
        return tracking_cost
    
    def filtering_cost(self, samples):
        diff_joint = samples[:,:,:] - self._q
        diff_joint = torch.norm(diff_joint, dim=2) * 5.0
        gamma = self._gamma ** torch.arange(self.n_horizen, device=self.device)

        filter_cost = diff_joint * gamma

        return filter_cost


    def terminal_cost(self):
        ee_terminal_pose = self.eefTraj[:,-1,0:3,3]
        ee_terminal_orientation = self.eefTraj[:,-1,0:3,0:3]

        diff_pose = ee_terminal_pose - self.target_pose.pose.to(device=self.device)
        diff_orientation = matrix_to_euler_angles(ee_terminal_orientation, "ZYX") - self.target_pose.rpy.to(device=self.device)

        cost_pose = torch.sum(torch.pow(diff_pose, 2), dim=1)
        cost_orientation = torch.sum(torch.abs(diff_orientation), dim=1)

        terminal_cost = self._terminal_pose_weight * cost_pose + self._terminal_orientation_weight * cost_orientation
        # terminal_cost = self._terminal_pose_weight * cost_pose

        terminal_cost = (self._gamma ** self.n_horizen) * terminal_cost
        return terminal_cost

    
    def update_control_input(self, samples, tracking_cost, terminal_cost):
        final_cost = torch.zeros((self.n_samples), device = self.device)
        final_cost += torch.sum(tracking_cost, dim=1)
    
        # final_cost += torch.sum(filter_cost, dim=1)
        # final_cost += terminal_cost
        rho = final_cost.min()
        scaledS = (-1.0/self._lambda) * (final_cost - rho)
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

    def computeTraj(self, qInit, vInit, noise):
        uSample = self.u + noise
        # qSample = qInit + vInit * self.dt + 0.5 * uSample * self.dt **2
        # qSample = qSample.reshape(self.samples_ * self.time_windows_, self.dof_)


        n_sample, n_horizon, n_action = uSample.shape

        qdot0 = vInit.unsqueeze(0).unsqueeze(0).expand(n_sample, 1, n_action)  # (n_sample, 1, n_action)
        q0 = qInit.unsqueeze(0).unsqueeze(0).expand(n_sample, 1, n_action)        # (n_sample, 1, n_action)
        v = torch.cumsum(uSample * self.dt, dim=1) + qdot0  # (n_sample, n_horizon, n_action)
        v_prev = torch.cat([qdot0, v[:, :-1, :]], dim=1)  # (n_sample, n_horizon, n_action)

        dq = v_prev * self.dt + 0.5 * uSample * self.dt**2
        qSample = torch.cumsum(dq, dim=1) + q0
        qSample = qSample.reshape(self.n_samples * self.n_horizen, self.n_action)

        
        eefTraj = torch.zeros((self.n_horizen * self.n_samples, 4, 4), device = self.device)
        eefTraj[:,:,:],_ = self.kinematics.computeEEFKinematics(qSample[:,:])

        self.eefTraj = eefTraj.reshape(self.n_samples, self.n_horizen, 4, 4)



    def log(self, cost_data):
        self.cost_log.add_scalar('cost', cost_data)
        self.cnt += 1


    def roll_buffer(self, weight, action):
        self.weight_buffer = torch.roll(self.weight_buffer, shifts=1, dims=0)
        self.action_buffer = torch.roll(self.action_buffer, shifts=1, dims=0)

        self.weight_buffer[0,:] = weight
        self.action_buffer[0,:,:] = action
        return


    def set_joint(self, joint_states : torch.Tensor):
        joint_states = joint_states.to(self.device)
    
        self._q = joint_states[:, 0]
        self._qdot = joint_states[:, 1]
        # self._qddot = joint_states[:, 2]
        return

    def set_ee_pose(self, pos, ori):
        self.ee_pose.pose = pos
        self.ee_pose.orientation = ori
        return

    def set_base_pose(self, pos, ori):
        self.base_pose.pose = pos
        self.base_pose.orientation = ori

        # base pose None
        posWorld = torch.tensor([0.0,0.0,0.0])
        oriWorld = torch.tensor([0.0, 0.0, 0.0, 1.0])
        self.base_pose.pose = posWorld
        self.base_pose.orientation = oriWorld
        return

    def set_target_pose(self, pos, ori):
        self.target_pose.pose = pos
        self.target_pose.orientation = ori
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
