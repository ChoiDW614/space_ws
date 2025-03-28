import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile
from rclpy.qos import DurabilityPolicy
from rclpy.qos import ReliabilityPolicy

from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import DynamicJointState
from geometry_msgs.msg import TransformStamped
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
        self.joint_order = [
            "v_x_joint", "v_y_joint", "v_z_joint", "v_r_joint", "v_p_joint", "v_yaw_joint",
            "Base_Joint", "Shoulder_Roll", "Shoulder_Yaw", "Elbow_Pitch", "Wrist_Pitch", "Wrist_Yaw", "Wrist_Roll"
        ]
        self.joint_names = None
        self.interface_name = None
        self.interface_values = None

        # model state subscriber
        subscribe_qos_profile = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        
        self.joint_state_subscriber = self.create_subscription(DynamicJointState, '/dynamic_joint_states', self.joint_state_callback, subscribe_qos_profile)
        self.base_state_subscriber = self.create_subscription(TransformStamped, '/model/canadarm/pose', self.model_state_callback, subscribe_qos_profile)
        self.target_state_subscriber = self.create_subscription(TransformStamped, '/model/ets_vii/pose', self.target_state_callback, subscribe_qos_profile)

        cal_timer_period = 0.1  # seconds
        pub_timer_period = 1  # seconds
        self.cal_timer = self.create_timer(cal_timer_period, self.cal_timer_callback)
        self.pub_timer = self.create_timer(pub_timer_period, self.pub_timer_callback)

        # arm publisher
        self.arm_msg = Float64MultiArray()
        self.arm_publisher = self.create_publisher(Float64MultiArray, '/floating_canadarm_joint_controller/commands', 10)


    def target_state_callback(self, msg):
        if msg.child_frame_id == 'ets_vii':
            self.controller.set_target_pose(msg.transform.translation, msg.transform.rotation)
    

    def cal_timer_callback(self):
        # start_time = time.time()
        u = self.controller.compute_control_input()
        self.arm_msg.data = u.tolist()
        # self.arm_msg.data = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        # for i in range(0, 13):
        #     self.arm_msg.data[i] = 0.0

        # self.arm_msg.data[0] = 1.0
        # self.arm_msg.data[1] = 2.0
        # self.arm_msg.data[2] = 5.0
        # self.arm_msg.data[3] = 1.0
        # end_time = time.time()
        # self.get_logger().info(str(end_time-start_time))


    def pub_timer_callback(self):
        self.arm_publisher.publish(self.arm_msg)
        return


    def joint_state_callback(self, msg):
        self.joint_names = msg.joint_names
        self.interface_name = [iv.interface_names for iv in msg.interface_values]
        values = [list(iv.values) for iv in msg.interface_values]

        index_map = [self.joint_names.index(joint) for joint in self.joint_order]
        self.interface_values = torch.tensor([values[i] for i in index_map])
        self.controller.set_joint(self.interface_values)


    def model_state_callback(self, msg):
        if msg.child_frame_id == 'canadarm/ISS':
            self.controller.set_base_pose(msg.transform.translation, msg.transform.rotation)
            # self.get_logger().info(f"x: {msg.transform.translation.x:.3f}, y: {msg.transform.translation.y:.3f}, z: {msg.transform.translation.z:.3f}")

# ISS
def main():
    rclpy.init()
    node = mppiControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
