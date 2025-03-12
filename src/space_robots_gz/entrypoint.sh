#!/bin/bash
set -e

# Setup the Demo environment
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${DEMO_DIR}/install/setup.bash"
echo "alias cball='cd \${DEMO_DIR} && colcon build --parallel-workers 6 --cmake-args -DCMAKE_BUILD_TYPE=Release'" >> ~/.bashrc
echo "alias cb='cd \${DEMO_DIR} && colcon build --packages-select canadarm canadarm_gz_plugin mars_rover simulation --parallel-workers 4 --cmake-args -DCMAKE_BUILD_TYPE=Release'" >> ~/.bashrc
echo "alias rl='ros2 launch canadarm canadarm.launch.py'" >> ~/.bashrc
echo "alias rl2='ros2 launch canadarm canadarm2.launch.py'" >> ~/.bashrc
echo "alias rl3='ros2 launch mars_rover mars_rover.launch.py'" >> ~/.bashrc
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
exec "$@"
