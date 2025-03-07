#!/bin/bash
set -e

# Setup the Demo environment
source "/opt/ros/${ROS_DISTRO}/setup.bash"
colcon build build --packages-select mars_rover simulation --parallel-workers 4 --cmake-args -DCMAKE_BUILD_TYPE=Release
source "${DEMO_DIR}/install/setup.bash"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
bash -c "ros2 launch mars_rover mars_rover.launch.py"
exec "$@"
