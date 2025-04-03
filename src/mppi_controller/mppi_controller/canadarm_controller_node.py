import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile
from rclpy.qos import DurabilityPolicy
from rclpy.qos import ReliabilityPolicy

from control_msgs.msg import DynamicJointState
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image

from gazebo_msgs.srv import SetEntityState
from std_srvs.srv import Empty
from rclpy.logging import get_logger

import cv2
import time
import numpy as np
import torch

from mppi_controller.src.solver.mppi_canadarm import MPPI
from mppi_controller.src.solver.target.target_state import DockingInterface

from mppi_controller.src.robot.canadarm_wrapper import CanadarmWrapper

from mppi_controller.src.utils.pose import Pose
from mppi_controller.src.utils.time import Time
from mppi_controller.src.utils.image_pipeline import ros_to_cv2


class mppiControllerNode(Node):
    def __init__(self):
        super().__init__("mppi_controller_node")
        # self.canadarmWrapper = CanadarmWrapper()

        # target states
        init_interface_pose = Pose()
        init_interface_pose.pose = torch.tensor([-2.1649, 4.4368, 4.3509])
        init_interface_pose.orientation = torch.tensor([-0.4744, -0.4535,  0.6023,  0.4544])
        self.docking_interface = DockingInterface(init_pose=init_interface_pose, predict_step=32)
        
        # camera images
        self.hand_eye_image = None
        self.base_image = None

        # joint control states
        self.isBaseMoving = True
        if self.isBaseMoving:
            self.joint_order = [
                "v_x_joint", "v_y_joint", "v_z_joint", "v_r_joint", "v_p_joint", "v_yaw_joint",
                "Base_Joint", "Shoulder_Roll", "Shoulder_Yaw", "Elbow_Pitch", "Wrist_Pitch", "Wrist_Yaw", "Wrist_Roll"]
        else:
            self.joint_order = [
                "Base_Joint", "Shoulder_Roll", "Shoulder_Yaw", "Elbow_Pitch", "Wrist_Pitch", "Wrist_Yaw", "Wrist_Roll"]
        self.joint_names = None
        self.interface_name = None
        self.interface_values = None

        # controller
        self.controller = MPPI(isBaseMoving=self.isBaseMoving)

        # model state subscriber
        subscribe_qos_profile = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        
        self.joint_state_subscriber = self.create_subscription(DynamicJointState, '/dynamic_joint_states', self.joint_state_callback, subscribe_qos_profile)
        self.base_state_subscriber = self.create_subscription(TransformStamped, '/model/canadarm/pose', self.model_state_callback, subscribe_qos_profile)
        self.target_state_subscriber = self.create_subscription(TransformStamped, '/model/ets_vii/pose', self.target_state_callback, subscribe_qos_profile)
        self.hand_eye_camera_subscriber = self.create_subscription(Image, '/SSRMS_camera/ee/image_raw', self.hand_eye_image_callback, subscribe_qos_profile)
        self.base_camera_subscriber = self.create_subscription(Image, '/SSRMS_camera/base/image_raw', self.base_image_callback, subscribe_qos_profile)

        # publisher
        cal_timer_period = 0.1  # seconds
        pub_timer_period = 1  # seconds
        self.cal_timer = self.create_timer(cal_timer_period, self.cal_timer_callback)
        self.pub_timer = self.create_timer(pub_timer_period, self.pub_timer_callback)

        self.arm_msg = Float64MultiArray()
        self.arm_publisher = self.create_publisher(Float64MultiArray, '/floating_canadarm_joint_controller/commands', 10)


    def target_state_callback(self, msg):
        if msg.child_frame_id == 'ets_vii':
            self.docking_interface.time = msg.header.stamp

            # true pose
            self.docking_interface.pose.pose = msg.transform.translation
            self.docking_interface.pose.orientation = msg.transform.rotation
            self.docking_interface.pose.x = msg.transform.translation.x - 1.0

            self.docking_interface.update_velocity()

            # kalman filter update
            self.docking_interface.ekf_update()
            self.controller.set_target_pose(self.docking_interface.pose)
            self.controller.set_predict_target_pose(self.docking_interface.predict_pose)

            # prev state
            self.docking_interface.pose_prev = self.docking_interface.pose
            self.docking_interface.time_prev = self.docking_interface.time
        return


    def cal_timer_callback(self):
        # start_time = time.time()
        u = self.controller.compute_control_input()
        self.arm_msg.data = u.tolist()
        # end_time = time.time()
        # self.get_logger().info(str(end_time-start_time))
        return


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
        return


    def model_state_callback(self, msg):
        if msg.child_frame_id == 'canadarm/ISS':
            self.controller.set_base_pose(msg.transform.translation, msg.transform.rotation)
        return
    

    def hand_eye_image_callback(self, msg):
        self.hand_eye_image = ros_to_cv2(msg)
        return


    def base_image_callback(self, msg):
        self.base_image = ros_to_cv2(msg)
        return


def main():
    rclpy.init()
    node = mppiControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
