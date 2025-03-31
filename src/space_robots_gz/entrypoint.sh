#!/bin/bash
set -e

# Setup the Demo environment
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source "/opt/ros/${ROS_DISTRO}/setup.bash"
if [ ! -f /tmp/_initialized ]; then
    echo "alias cball='cd \${DEMO_DIR} && colcon build --parallel-workers 4 --cmake-args -DCMAKE_BUILD_TYPE=Release'" >> ~/.bashrc
    echo "alias cb='cd \${DEMO_DIR} && colcon build --packages-select franka canadarm ets_vii simulation --parallel-workers 4 --cmake-args -DCMAKE_BUILD_TYPE=Release'" >> ~/.bashrc
    echo "alias cb2='cd \${DEMO_DIR} && colcon build --packages-select mppi_controller robot_manager --symlink-install --parallel-workers 4 --cmake-args -DCMAKE_BUILD_TYPE=Release'" >> ~/.bashrc
    echo "alias sb='source \${DEMO_DIR}/install/setup.bash'" >> ~/.bashrc
    echo "alias rl='ros2 launch canadarm floating_canadarm_camera.launch.py'" >> ~/.bashrc
    echo "alias rl2='ros2 launch canadarm floating_canadarm.launch.py'" >> ~/.bashrc
    echo "alias rl3='ros2 launch canadarm canadarm.launch.py'" >> ~/.bashrc
    echo "alias rl4='ros2 launch canadarm canadarm2.launch.py'" >> ~/.bashrc
    echo "alias rl5='ros2 launch mars_rover mars_rover.launch.py'" >> ~/.bashrc
    echo "alias ctl='ros2 launch robot_manager run_canadarm_control.launch.py'" >> ~/.bashrc

    cd ~/demos_ws && colcon build --packages-select franka canadarm ets_vii simulation --parallel-workers 4 --cmake-args -DCMAKE_BUILD_TYPE=Release
    cd ~/demos_ws && colcon build --packages-select mppi_controller robot_manager --symlink-install --parallel-workers 3 --cmake-args -DCMAKE_BUILD_TYPE=Release
    touch /tmp/_initialized
fi
source "${DEMO_DIR}/install/setup.bash"
exec "$@"
