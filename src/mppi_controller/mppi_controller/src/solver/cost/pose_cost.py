import torch
import torch.nn as nn
import numpy as np

from mppi_controller.src.utils.pose import Pose
from mppi_controller.src.utils.rotation_conversions import euler_angles_to_matrix, matrix_to_euler_angles


class PoseCost():
    def __init__(self, n_horizen, device):
        self._tracking_pose_weight = 1.0
        self._tracking_orientation_weight = 0.1

        self._terminal_pose_weight = 10.0
        self._terminal_orientation_weight = 1.0

        self._gamma = 0.98
        self.n_horizen = n_horizen
        self.device = device


    def tracking_cost(self, eefTraj, target_pose: Pose):
        ee_sample_pose = eefTraj[:,:,0:3,3]
        ee_sample_orientation = eefTraj[:,:,0:3,0:3]
        
        diff_pose = ee_sample_pose - target_pose.pose.to(device=self.device)
        diff_orientation = matrix_to_euler_angles(ee_sample_orientation, "ZYX") - target_pose.rpy.to(device=self.device)

        cost_pose = torch.sum(torch.pow(diff_pose, 2), dim=2)
        cost_orientation = torch.sum(torch.abs(diff_orientation), dim=2)

        tracking_cost = self._tracking_pose_weight * cost_pose + self._tracking_orientation_weight * cost_orientation

        gamma = self._gamma ** torch.arange(self.n_horizen, device=self.device)

        tracking_cost = tracking_cost * gamma
        return tracking_cost


    def terminal_cost(self, eefTraj, target_pose: Pose):
        ee_terminal_pose = eefTraj[:,-1,0:3,3]
        ee_terminal_orientation = eefTraj[:,-1,0:3,0:3]

        diff_pose = ee_terminal_pose - target_pose.pose.to(device=self.device)
        diff_orientation = matrix_to_euler_angles(ee_terminal_orientation, "ZYX") - target_pose.rpy.to(device=self.device)

        cost_pose = torch.sum(torch.pow(diff_pose, 2), dim=1)
        cost_orientation = torch.sum(torch.abs(diff_orientation), dim=1)

        terminal_cost = self._terminal_pose_weight * cost_pose + self._terminal_orientation_weight * cost_orientation

        terminal_cost = (self._gamma ** self.n_horizen) * terminal_cost
        return terminal_cost
    

    def predict_tracking_cost(self, eefTraj, target_pose: np.ndarray):
        ee_sample_pose = eefTraj[:,:,0:3,3]
        ee_sample_orientation = eefTraj[:,:,0:3,0:3]
        
        diff_pose = ee_sample_pose - target_pose[:,0:3].to(device=self.device)
        diff_orientation = matrix_to_euler_angles(ee_sample_orientation, "ZYX") - target_pose[:,3:6].to(device=self.device)

        cost_pose = torch.sum(torch.pow(diff_pose, 2), dim=2)
        cost_orientation = torch.sum(torch.abs(diff_orientation), dim=2)

        tracking_cost = self._tracking_pose_weight * cost_pose + self._tracking_orientation_weight * cost_orientation

        gamma = self._gamma ** torch.arange(self.n_horizen, device=self.device)

        tracking_cost = tracking_cost * gamma
        return tracking_cost
    
