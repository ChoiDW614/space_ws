from http.server import executable
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler
from launch.substitutions import TextSubstitution, PathJoinSubstitution, LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import OnProcessExit, OnExecutionComplete
import os
from os import environ

from ament_index_python.packages import get_package_share_directory

import xacro


def generate_launch_description():
    simulation_models_path = get_package_share_directory('simulation')

    ets_vii_urdf_model_path = os.path.join(simulation_models_path, 'models', 'ets_vii', 'urdf', 'ets_vii.urdf.xacro')

    ets_vii_doc = xacro.process_file(ets_vii_urdf_model_path)
    ets_vii_robot_description = {'robot_description': ets_vii_doc.toxml()}


    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='ets_vii_robot_state_publisher',
        namespace='ets_vii',
        output='screen',
        parameters=[ets_vii_robot_description]
    )

    spawn = Node(
        package='ros_ign_gazebo', executable='create',
        namespace='ets_vii',
        arguments=[
            '-name', 'ets_vii',
            '-topic', ets_vii_robot_description,
            '-x', '-2.1649', '-y', '4.4368', '-z', '12.3509',
            '-R', '-1.43', '-P', '0.16', '-Y', '1.71',
        ],
        output='screen'
    )

    # Control
    # load_ets_vii_joint_state_broadcaster = ExecuteProcess(
    #     cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
    #          'ets_vii_joint_state_broadcaster'],
    #     output='screen'
    # )

    # load_ets_vii_joint_controller = ExecuteProcess(
    #     cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
    #          'ets_vii_joint_controller'],
    #     output='screen'
    # )

    return LaunchDescription([
        robot_state_publisher,
        spawn,

        # RegisterEventHandler(
        #     OnProcessExit(
        #         target_action=spawn,
        #         on_exit=[load_ets_vii_joint_state_broadcaster],
        #     )
        # ),
        # RegisterEventHandler(
        #     OnProcessExit(
        #         target_action=load_ets_vii_joint_state_broadcaster,
        #         on_exit=[load_ets_vii_joint_controller],
        #     )
        # ),
    ])
