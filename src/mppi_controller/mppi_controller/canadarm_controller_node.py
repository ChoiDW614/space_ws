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
from nav_msgs.msg import Odometry
from control_msgs.msg import DynamicJointState

from gazebo_msgs.srv import SetEntityState
from std_srvs.srv import Empty

import math
import time
import numpy as np
import torch
import functools
import random
from typing import Tuple

from mppi_controller.src.solver.mppi_canadarm import MPPI
from mppi_controller.src.robot.canadarm_wrapper import CanadarmWrapper


class header():
    def __init__(self):
        self.frame_id = str()
        self.stamp = None


class header():
    def __init__(self):
        self.frame_id = str()
        self.stamp = None


class mppiControllerNode(Node):
    def __init__(self):
        super().__init__("mppi_controller_node")

        # self.canadarmWrapper = CanadarmWrapper()

        self.controller = MPPI()

        self.header = header()
        self.cnt = 0
        self.interface_name = None
        self.interface_values = None

        # model state subscriber
        subscribe_qos_profile = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        
        self.joint_state_subscriber = self.create_subscription(DynamicJointState, '/dynamic_joint_states', self.joint_state_callback, subscribe_qos_profile)
        # self.base_state_subscriber = self.create_subscription(Odometry, '/model/curiosity_mars_rover/odometry', self.base_state_callback, subscribe_qos_profile)

        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # # arm publisher
        # self.arm_publisher_ = self.create_publisher(Float64MultiArray, '/arm_joint_effort_controller/commands', 10)
    
    
    def timer_callback(self):
        self.controller.compute_control()

    def joint_state_callback(self, msg):
        self.joint_names = msg.joint_names
        self.interface_name = [iv.interface_names for iv in msg.interface_values]
        self.interface_values = torch.tensor([list(iv.values) for iv in msg.interface_values])


def main():
    rclpy.init()
    node = mppiControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    