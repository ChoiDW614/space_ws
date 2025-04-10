# space_ws

This repository provides an implementation of satellite robot control using Model Predictive Path Integral (MPPI)

## Usage
use it after modifying the path in the the <strong>src/space_robots_gz/run.sh</strong> file

1. shortcuts in docker terminal   
```
rlcanadarm  = ros2 launch canadarm canadarm.launch.py
rlfranka    = ros2 launch franka franka.launch.py
rlfloating  = ros2 launch canadarm floating_canadarm_camera.launch.py
```
2. Register shortcuts in local   
```
echo "alias rldocker='cd $SPACE_WS/src/space_robots_gz && ./build.sh && ./run.sh'" >> ~/.bashrc
echo "alias execdocker='cd $SPACE_WS/src/space_robots_gz && ./exec.sh'" >> ~/.bashrc
echo "alias stopdocker='docker stop openrobotics_space_robots_demo'" >> ~/.bashrc
```

# Installation

Install Docker:   
https://docs.docker.com/engine/install/ubuntu/

Install Earthly:   
https://earthly.dev/get-earthly

```
cd ~/space_ws/src/space_robots_gz/moveit2
./build.sh

cd ~/space_ws/src/space_robots_gz/space_robots
./build.sh
```

run the following to allow GUI passthrough:
```
xhost +local:docker
```

Install the nvidia Toolkit

```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
  && \
    sudo apt-get update
```

Validate the nvidia Toolkit installation
```
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

## To-do List

- Modify a cost function
- Calculate the docking interface(target) pose

## Changelog

- **2025-04-04**  
  - Separate the robot wrapper from the solver
  - Improving solver performance
- **2025-03-31**  
  - Apply a Kalman filter to a target pose
  - Mounting the camera on CANADARM's end effector and connecting the image to ROS2
  - Assigning initial velocity to target satellite for tumblig task
- **2025-03-24**  
  - Porting an mppi controller to a docker container
  - Add an algorithm to update the distribution
- **2025-03-18**  
  - Implementing the MPPI Controller
- **2025-03-11**  
  - `README.md`: Add To-do List
- **2025-03-09**  
  - Create initial files and set up basic structure
