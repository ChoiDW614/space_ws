"""This module is in most parts copied from https://github.com/mahaarbo/urdf2casadi.

Changes are in get_forward_kinematics as it allows to pass the variable as an argument.
"""
# import casadi as ca
import numpy as np
from typing import List, Set
from urdf_parser_py.urdf import URDF
from mppi_controller.src.robot.urdfFks.casadiConversion.geometry.transformation_matrix import prismatic_transform, revolute_transform, make_transform_matrix


class URDFparser(object):
    """Class that turns a chain from URDF to casadi functions."""
    actuated_types = ["prismatic", "revolute", "continuous"]

    def __init__(self, root_link: str = "base_link", end_links: List[str] = None):
        self._root_link = root_link
        if isinstance(end_links, str):
            self._end_links = [end_links]
        else:
            self._end_links = end_links if end_links else []

        self.robot_desc = None
        self._absolute_root_link = None
        self._active_joints: Set[str] = set()
        self._actuated_joints: List[str] = []
        self._joint_map = {}
        self._degrees_of_freedom = 0
        self._link_names = []

    def from_file(self, filename: str):
        self.robot_desc = URDF.from_xml_file(filename)
        self._extract_information()

    def from_string(self, urdfstring: str):
        self.robot_desc = URDF.from_xml_string(urdfstring)
        self._extract_information()

    def _extract_information(self):
        self._detect_link_names()
        self._absolute_root_link = self.robot_desc.get_root()
        self._set_active_joints()
        self._set_actuated_joints()
        self._extract_degrees_of_freedom()
        self._set_joint_variable_map()

    def _set_actuated_joints(self):
        self._actuated_joints = []
        for joint in self.robot_desc.joints:
            if joint.type in self.actuated_types:
                self._actuated_joints.append(joint.name)

    def _extract_degrees_of_freedom(self):
        self._degrees_of_freedom = 0
        for jn in self._active_joints:
            if jn in self._actuated_joints:
                self._degrees_of_freedom += 1

    def degrees_of_freedom(self) -> int:
        return self._degrees_of_freedom

    def _set_joint_variable_map(self):
        self._joint_map = {}
        idx = 0
        for joint_name in self._actuated_joints:
            if joint_name in self._active_joints:
                self._joint_map[joint_name] = idx
                idx += 1

    def joint_map(self):
        return self._joint_map

    def _is_active_joint(self, joint):
        parent_link = joint.parent
        while parent_link not in [self._root_link, self._absolute_root_link]:
            if parent_link in self._end_links:
                return False
            (parent_joint, parent_link) = self.robot_desc.parent_map[parent_link]
            if parent_joint in self._active_joints:
                return True
        if parent_link == self._root_link:
            return True
        return False

    def _set_active_joints(self):
        for end_lk in self._end_links:
            parent_link = end_lk
            while parent_link not in [self._root_link, self._absolute_root_link]:
                (parent_joint, parent_link) = self.robot_desc.parent_map[parent_link]
                self._active_joints.add(parent_joint)
                if parent_link == self._root_link:
                    break

    def active_joints(self) -> Set[str]:
        return self._active_joints

    def _detect_link_names(self):
        self._link_names = []
        for link in self.robot_desc.links:
            # parent_map에 존재해야 체인에 연결된 링크로 본다
            if link.name in self.robot_desc.parent_map:
                self._link_names.append(link.name)
        return self._link_names

    def link_names(self):
        return self._link_names

    def _get_joint_chain(self, tip: str):
        if self.robot_desc is None:
            raise ValueError("Robot description not loaded.")
        chain = self.robot_desc.get_chain(self._absolute_root_link, tip)
        joint_list = []
        for item in chain:
            if item in self.robot_desc.joint_map:
                jnt = self.robot_desc.joint_map[item]
                if jnt.name in self._active_joints:
                    joint_list.append(jnt)
        return joint_list

    def get_forward_kinematics(
        self,
        root: str,
        tip: str,
        q: np.ndarray,
        link_transformation: np.ndarray = np.eye(4)
    ):
        if self.robot_desc is None:
            raise ValueError("Robot description not loaded.")
        T_fk = np.eye(4)
        joint_list = self._get_joint_chain(tip)

        for jt in joint_list:
            jtype = jt.type
            xyz = np.array(jt.origin.xyz, dtype=float)
            rpy = np.array(jt.origin.rpy, dtype=float)

            # axis가 없는 경우(continuous 조인트 등)는 기본 x축
            if jt.axis is None:
                axis = np.array([1.0, 0.0, 0.0], dtype=float)
            else:
                axis = np.array(jt.axis, dtype=float)

            # joint_map에서 해당 조인트의 q 인덱스가 있는 경우
            # fixed joint는 DOF가 없으므로 맵핑이 없을 수도 있음
            if jt.name in self._joint_map:
                q_idx = self._joint_map[jt.name]
                q_val = q[q_idx]
            else:
                q_val = 0.0

            if jtype == "fixed":
                T_local = make_transform_matrix(xyz, rpy)
                T_fk = T_fk @ T_local
            elif jtype == "prismatic":
                T_local = prismatic_transform(xyz, rpy, axis, q_val)
                T_fk = T_fk @ T_local
            elif jtype in ["revolute", "continuous"]:
                T_local = revolute_transform(xyz, rpy, axis, q_val)
                T_fk = T_fk @ T_local
            else:
                # 예외적인 joint 타입 등
                T_local = make_transform_matrix(xyz, rpy)
                T_fk = T_fk @ T_local

        # tip 링크 자체에 link_transformation이 있으면 곱해준다
        T_fk = T_fk @ link_transformation
        return T_fk
