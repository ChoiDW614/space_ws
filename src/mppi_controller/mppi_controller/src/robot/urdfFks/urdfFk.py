from typing import Union, List
import numpy as np
# import casadi as ca
import mppi_controller.src.robot.urdfFks.casadiConversion.urdfparser as u2c

from mppi_controller.src.robot.fksCommon.fk import ForwardKinematics

class LinkNotInURDFError(Exception):
    pass

class URDFForwardKinematics(ForwardKinematics):
    def __init__(self, urdf: str, root_link: str, end_links: List[str], base_type: str = "holonomic"):
        super().__init__()
        self._urdf = urdf
        self._root_link = root_link
        self._end_links = end_links
        self._base_type = base_type

        self.robot = u2c.URDFparser(root_link, end_links)
        self.robot.from_file(urdf)

        self._n_dof = self.robot.degrees_of_freedom()

    def numpy(self,
        q: np.ndarray,
        child_link: str,
        parent_link: Union[str, None] = None,
        link_transformation: np.ndarray = np.eye(4),
        position_only: bool = False
    ) -> np.ndarray:
        
        if parent_link is None:
            parent_link = self._root_link

        if child_link not in self.robot.link_names() and child_link != self._root_link:
            raise LinkNotInURDFError(f"The link {child_link} is not in the URDF. Valid links: {self.robot.link_names()}")
        if parent_link not in self.robot.link_names() and parent_link != self._root_link:
            raise LinkNotInURDFError( f"The link {parent_link} is not in the URDF. Valid links: {self.robot.link_names()}")

        q_joints = q

        if parent_link == self._root_link:
            T_parent = np.eye(4)
        else:
            T_parent = self.robot.get_forward_kinematics(self.robot._absolute_root_link, parent_link, q_joints)
            T_parent = self._mount_transformation @ T_parent

        if child_link == self._root_link:
            T_child = np.eye(4)
        else:
            T_child = self.robot.get_forward_kinematics(self.robot._absolute_root_link, child_link, q_joints)
            T_child = self._mount_transformation @ T_child

        T_parent_inv = np.linalg.inv(T_parent)
        T_parent_child = T_parent_inv @ T_child

        T_parent_child = link_transformation @ T_parent_child

        if position_only:
            return T_parent_child[:3, 3]
        else:
            return T_parent_child
        