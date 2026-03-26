# safe-bimanual-rl
Safe Reinforcement learning for tray pickup with Safety Filters

## Project Structure

```
dual_arm_description/              # URDF, meshes, and RViz configuration for the dual-arm robot
lbr_description/                   # URDF and related files for the single-arm LBR robot              
robotiq_hande_description/         # URDF and assets for the Robotiq gripper
dual_arm_iiwa.xml                  # Original dual-arm XML URDF file
README.md                          # Project description and documentation
Makefile                           # Make commands for build/launch scripts
```

## Visualization

To visualize the project using Rviz2 with ROS2 you can clone this project in a ```src/``` folder in a ros2 workspace.

Then you can build and source the project.

(Source each terminal before using the following commands)

- In one terminal, launch the robot_state_publisher node :

(Work only for me now)
```bash
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:="$(xacro /home/mga/Documents/project_rl_ws/src/safe-bimanual-rl/dual_arm_iiwa.xml)"
```

- In a second terminal, launch the joint_state_publisher node :

```bash
ros2 run joint_state_publisher_gui joint_state_publisher_gui
```

- In a third terminal, launch rviz2 :

```bash
ros2 run rviz2 rviz2
```


In rviz2 :

- Put the fixed frame as ```bh_robot_base```

- Add ```RobotModel```, in Description Source select ```file``` and in Description File select the ```dual_arm_iiwa.xml``` file