# session_3 (Files for workshop related to PPO & Mujoco)
This package provides an environment for testing different RL algorithms under manipulation scenarios enabling a flexible test environment


## Build ROS package
``` 
    source /opt/ros/humble/setup.bash
    cd rl_wksp/session_3
    colcon build
    source install/setup.bash
```

## To train the model
``` python3 src/ros2_gym_env/scripts/run_ros2_gym_env.py --render_mode human --mode Train ```

## To test the trained model
``` python3 src/ros2_gym_env/scripts/run_ros2_gym_env.py --render_mode human --mode Test ```

# References
[Manipulator Mujoco](https://github.com/ian-chuang/Manipulator-Mujoco.git)
