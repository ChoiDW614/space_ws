#!/bin/bash
set -e

# Setup the Demo environment
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${DEMO_DIR}/install/setup.bash"
echo "alias cball='cd \${DEMO_DIR} && colcon build --parallel-workers 6 --cmake-args -DCMAKE_BUILD_TYPE=Release'" >> ~/.bashrc
echo "alias cb='cd \${DEMO_DIR} && colcon build --packages-select mars_rover simulation --parallel-workers 4 --cmake-args -DCMAKE_BUILD_TYPE=Release'" >> ~/.bashrc
echo "alias rl='ros2 launch mars_rover mars_rover.launch.py'" >> ~/.bashrc
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
exec "$@"
