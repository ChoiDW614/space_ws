import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile
from rclpy.qos import DurabilityPolicy
from rclpy.qos import ReliabilityPolicy

from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import DynamicJointState
from geometry_msgs.msg import PoseArray

from gazebo_msgs.srv import SetEntityState
from std_srvs.srv import Empty

import torch

from mppi_controller.src.solver.mppi_canadarm import MPPI
from mppi_controller.src.robot.canadarm_wrapper import CanadarmWrapper
from mppi_controller.src.utils.pose import Pose


class mppiControllerNode(Node):
    def __init__(self):
        super().__init__("mppi_controller_node")

        # self.canadarmWrapper = CanadarmWrapper()

        self.controller = MPPI()

        # joint control states
        self.joint_names = None
        self.interface_name = None
        self.interface_values = None

        # model state subscriber
        subscribe_qos_profile = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        
        self.joint_state_subscriber = self.create_subscription(DynamicJointState, '/dynamic_joint_states', self.joint_state_callback, subscribe_qos_profile)
        self.base_state_subscriber = self.create_subscription(PoseArray, '/world/default/pose/info', self.base_state_callback, subscribe_qos_profile)

        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # arm publisher
        self.traj_msg = JointTrajectory()
        self.traj_msg.joint_names = ["Base_Joint", "Shoulder_Roll", "Shoulder_Yaw", "Elbow_Pitch", "Wrist_Pitch", "Wrist_Yaw", "Wrist_Roll"]
        self.arm_publisher = self.create_publisher(JointTrajectory, '/canadarm_joint_trajectory_controller/joint_trajectory', 10)
    
    
    def timer_callback(self):
        self.controller.compute_control_input()

        point1 = JointTrajectoryPoint()
        point1.positions = [1.0, -1.5, 2.0, -3.2, 0.8, 0.5, -1.0]
        point1.time_from_start = Duration(sec=0)

        self.traj_msg.points.append(point1)
        self.arm_publisher.publish(self.traj_msg)


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
    