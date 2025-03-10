from typing import Union, List
import numpy as np
import torch

import mppi_controller.src.robot.urdfFks.urdfparser as u2c

class LinkNotInURDFError(Exception):
    pass

class URDFForwardKinematics():
    def __init__(self, urdf: str, root_link: str, end_links: List[str], base_type: str = "holonomic"):
        self._urdf = urdf
        self._root_link = root_link
        self._end_links = end_links
        self._base_type = base_type

        self.robot = u2c.URDFparser(root_link, end_links)
        self.robot.from_file(urdf)

        self.robot._joint_chain_list = self.robot._get_joint_chain(self._end_links)
        self._n_dof = self.robot.degrees_of_freedom()
        self._mount_transformation = torch.eye(4)


    def set_mount_transformation(self, mount_transformation):
        self._mount_transformation  = mount_transformation


    def set_samples_and_timesteps(self, n_samples, n_timesteps):
        self.robot._n_samples = n_samples
        self.robot._n_timestep = n_timesteps
        

    def forward_kinematics(self,
        q: torch.Tensor,
        child_link: str,
        parent_link: Union[str, None] = None,
    ) -> torch.Tensor:
        
        if parent_link is None:
            parent_link = self._root_link

        if child_link not in self.robot.link_names() and child_link != self._root_link:
            raise LinkNotInURDFError(f"The link {child_link} is not in the URDF. Valid links: {self.robot.link_names()}")
        if parent_link not in self.robot.link_names() and parent_link != self._root_link:
            raise LinkNotInURDFError( f"The link {parent_link} is not in the URDF. Valid links: {self.robot.link_names()}")

        if parent_link == self._root_link:
            tf_parent = torch.eye(4, device=q.device).expand(self.robot._n_samples, self.robot._n_timestep, 4, 4).clone()
        else:
            tf_parent = self.robot.forward_kinematics(q)
            tf_parent = self._mount_transformation @ tf_parent

        if child_link == self._root_link:
            tf_child = torch.eye(4, device=q.device).expand(self.robot._n_samples, self.robot._n_timestep, 4, 4).clone()
        else:
            tf_child = self.robot.forward_kinematics(q)
            tf_child = self._mount_transformation @ tf_child

        tf_parent_inv = torch.linalg.inv(tf_parent)
        tf_parent_child = tf_parent_inv @ tf_child
        
        return tf_parent_child
