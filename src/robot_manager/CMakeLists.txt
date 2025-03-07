cmake_minimum_required(VERSION 3.5)
project(robot_manager)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(
  PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/../space_robots_gz/ros_run.sh
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
