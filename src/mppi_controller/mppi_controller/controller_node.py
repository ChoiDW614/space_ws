import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile
from rclpy.qos import HistoryPolicy
from rclpy.qos import DurabilityPolicy
from rclpy.qos import ReliabilityPolicy


from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, PoseStamped, Twist
from visualization_msgs.msg import Marker, MarkerArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from gazebo_msgs.srv import SetEntityState
from std_srvs.srv import Empty

import math
import time
import numpy as np
import torch
import functools
import random
from typing import Tuple

from mppi_controller.src.solver.mppi import MPPI
from mppi_controller.src.robot.robot_wrapper import CuriosityWrapper


class mppiControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("mppi_controller_node")

        self.curiosityWrapper = CuriosityWrapper()

        self.controller = MPPI()
        self.controller.compute_control()

        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # arm
        self.arm_publisher_ = self.create_publisher(Float64MultiArray, '/arm_joint_effort_controller/commands', 10)

        # mast
        self.mast_publisher_ = self.create_publisher(JointTrajectory, '/mast_joint_trajectory_controller/joint_trajectory', 10)

        # motion
        self.curr_action = Twist()
        self.motion_publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
    
    
    def timer_callback(self):
            self.motion_publisher_.publish(self.curr_action)


def main():
    rclpy.init()
    node = mppiControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    