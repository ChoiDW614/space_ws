import torch
from mppi_controller.src.utils.rotation_conversions import quaternion_to_matrix, matrix_to_quaternion, matrix_to_euler_angles

class Pose():
    def __init__(self):
        self.__pose = torch.zeros(3)
        self.__orientation = torch.tensor([0.0, 0.0, 0.0, 1.0])
        self.__tf = None

    @property
    def pose(self):
        return self.__pose
    
    @pose.setter
    def pose(self, pos):
        if isinstance(pos, torch.Tensor):
            self.__pose = pos
        else:
            self.__pose = torch.tensor([pos.x, pos.y, pos.z])

    @property
    def orientation(self):
        return self.__orientation
    
    @orientation.setter
    def orientation(self, ori):
        if isinstance(ori, torch.Tensor):
            self.__orientation = ori
            self.__tf = quaternion_to_matrix(self.__orientation)
        else:
            self.__orientation = torch.tensor([ori.x, ori.y, ori.z, ori.w])
            self.__tf = quaternion_to_matrix(self.__orientation)

    def tf_matrix(self, device = None):
        if device is None:
            matrix = torch.eye(4)
        else:
            matrix = torch.eye(4, device=device)
        matrix[0:3, 0:3] = self.__tf
        matrix[0:3, 3] = self.__pose
        return matrix
    
    def from_matrix(self, matrix):
        self.__pose = matrix[0:3, 3]
        self.__orientation = matrix_to_quaternion(matrix[0:3, 0:3])
        self.__tf = matrix[0:3, 0:3]

    def rpy(self):
        return matrix_to_euler_angles(self.__tf, "ZYX")


def pose_diff(ppos1: Pose, ppose2: Pose):
    pose_difference = torch.sum(torch.abs(ppos1.pose - ppose2.pose))
    orientation_difference = torch.sum(torch.abs(ppos1.orientation - ppose2.orientation))
    return pose_difference + orientation_difference

def pos_diff(ppos1: Pose, ppose2: Pose):
    pose_difference = torch.sum(torch.abs(ppos1.pose - ppose2.pose))
    return pose_difference