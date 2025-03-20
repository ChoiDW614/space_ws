import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile
from rclpy.qos import DurabilityPolicy
from rclpy.qos import ReliabilityPolicy

from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import DynamicJointState
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float64MultiArray

from gazebo_msgs.srv import SetEntityState
from std_srvs.srv import Empty

import time
import numpy as np
import torch

from mppi_controller.src.solver.mppi_canadarm import MPPI
from mppi_controller.src.robot.canadarm_wrapper import CanadarmWrapper
from mppi_controller.src.utils.pose import Pose


class mppiControllerNode(Node):
    def __init__(self):
        super().__init__("mppi_controller_node")

        # self.canadarmWrapper = CanadarmWrapper()

        self.controller = MPPI()

        self.target_pose = Pose()

        # joint control states
        self.joint_names = None
        self.interface_name = None
        self.interface_values = None

        # model state subscriber
        subscribe_qos_profile = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        
        self.joint_state_subscriber = self.create_subscription(DynamicJointState, '/dynamic_joint_states', self.joint_state_callback, subscribe_qos_profile)
        self.base_state_subscriber = self.create_subscription(PoseArray, '/world/default/pose/info', self.base_state_callback, subscribe_qos_profile)

        cal_timer_period = 0.1  # seconds
        pub_timer_period = 1  # seconds
        self.cal_timer = self.create_timer(cal_timer_period, self.cal_timer_callback)
        self.pub_timer = self.create_timer(pub_timer_period, self.pub_timer_callback)

        # arm publisher
        self.arm_msg = Float64MultiArray()
        self.arm_publisher = self.create_publisher(Float64MultiArray, '/canadarm_joint_controller/commands', 10)

        self.target_state_callback()


    def target_state_callback(self):
        self.target_pose.pose = torch.tensor([-2.1649, 4.4368, 18.3509])
        self.target_pose.orientation = torch.tensor([0.4630, -0.4653, -0.4581, 0.5994]) # euler angle rpy : -1.43, 0.16, 1.71
        self.controller.set_target_pose(self.target_pose)
    

    def cal_timer_callback(self):
        # start_time = time.time()
        u = self.controller.compute_control_input()
        self.arm_msg.data = u.tolist()
        # end_time = time.time()

        # self.get_logger().info(str(end_time-start_time))


    def pub_timer_callback(self):
        self.arm_publisher.publish(self.arm_msg)


    def joint_state_callback(self, msg):
        self.joint_names = msg.joint_names
        self.interface_name = [iv.interface_names for iv in msg.interface_values]
        self.interface_values = torch.tensor([list(iv.values) for iv in msg.interface_values])
        self.controller.set_init_joint(self.interface_values)


    def base_state_callback(self, msg):
        self.controller.set_base_pose(msg.poses[1].position, msg.poses[1].orientation) # poses[1] -> base pose (ros_gz_bridge)


def main():
    rclpy.init()
    node = mppiControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    