import torch

class Pose():
    def __init__(self):
        self.__pose = torch.zeros(3)
        self.__orientation = torch.zeros(4)

    @property
    def pose(self):
        return self.__pose
    
    @pose.setter
    def pose(self, pos):
        self.__pose = pos

    @property
    def orientation(self):
        return self.__orientation
    
    @orientation.setter
    def orientation(self, ori):
        self.__orientation = ori


def pose_diff(ppos1: Pose, ppose2: Pose):
    pose_difference = torch.sum(torch.abs(ppos1.pose - ppose2.pose))
    orientation_difference = torch.sum(torch.abs(ppos1.orientation - ppose2.orientation))
    return pose_difference + orientation_difference
