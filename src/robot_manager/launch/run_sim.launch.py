import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    TimerAction,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    LogInfo,
)

from launch.actions import ExecuteProcess, RegisterEventHandler
from launch.event_handlers.on_shutdown import OnShutdown
from ament_index_python.packages import get_package_share_directory

def launch_setup(context, *args, **kwargs):
    run_sh_path = os.path.join(get_package_share_directory('robot_manager'), '..', '..', 'lib', 'robot_manager', 'ros_run.sh')

    env_setup = ExecuteProcess(
        cmd=['/usr/bin/env', 'bash', run_sh_path],
        output='screen'
    )

    stop_env = ExecuteProcess(
        cmd=['docker', 'stop', 'openrobotics_space_robots_demo'],
        output='screen',
        shell=False
    )

    on_shutdown_handler = RegisterEventHandler(
        event_handler=OnShutdown(
            on_shutdown=[stop_env]
        )
    )


    nodes_to_start =[
        env_setup,
        on_shutdown_handler,
    ]
    return nodes_to_start


def generate_launch_description():
    declared_arguments = []

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
