import time
import os
import numpy as np
import threading
from dm_control import mjcf
import mujoco.viewer
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from gymnasium import spaces, Env
from scipy.spatial.transform import Rotation as R
from ament_index_python.packages import get_package_share_directory
from lib.arenas import StandardArena
from lib.robots import Arm, AG95
from lib.mocaps import Target
from lib.controllers import OperationalSpaceController, JointEffortController
from lib.utils.transform_utils import mat2quat


class RlManipEnv(Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": None,
    }

    def __init__(
        self,
        render_mode = None,
        env_mode = None,
        robot_model = None,
        gripper_model = None,
        task = None,
        initial_pose = [0, -1.5707, 1.5707, -1.5707, -1.5707, 0.0],
        target_pose = [-0.7, 0.0, 0.02, 0.0, 0.0, 0.0, 1.0],
        max_steps = 500,
        action_mode = "joint"
    ):
        # Initialize ROS 2 Node
        rclpy.init(args=None)
        super(RlManipEnv, self).__init__()
        self.node = rclpy.create_node('rl_manip_env_ros2')
        
        # Environment parameters
        self._render_mode = render_mode
        self._env_mode = env_mode
        self._gripper_model = gripper_model
        self._task = task
        self._initial_pose = initial_pose
        self._target_pose = target_pose
        self._action_mode = action_mode

        # ROS 2 publishers and subscribers
        self.ee_pos_publisher = self.node.create_publisher(Float64MultiArray, 'ee_position', 10)
        self.observation_publisher = self.node.create_publisher(Float64MultiArray, 'observation', 10)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-0.05, high=0.05, shape=(6,), dtype=np.float64  
        )
        
        self.received_action = None
        self._render_mode = render_mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        
        # Create MJCF model components
        self._arena = StandardArena()
        self._target = Target(self._arena.mjcf_model, self._arena, [0.0, 0.0, 0.0], self._task)

        package_share_dir = get_package_share_directory('ros2_gym_env')

        # Load Robot Arm and attach it to the arena
        self._arm = Arm(
            os.path.join(package_share_dir, f'lib/assets/robots/{robot_model}/{robot_model}.xml'),
            eef_site_name='eef_site',
            attachment_site_name='attachment_site'
        )
        self._gripper_offset = [0.0, 0.0, 0.0]
        # Load Robot Arm and attach it to the arena
        # self._gripper = AG95()
        if self._gripper_model:
            # self._gripper = AG95()
            self._gripper = globals()[gripper_model]() if gripper_model in globals() else None
            self._arm.attach_tool(self._gripper.mjcf_model, pos=[0, 0, 0], quat=[0, 0, 0, 1])
            self._gripper_offset = [0.0, 0.0, 0.18]

        self._arena.attach(self._arm.mjcf_model, pos=[0, 0, 0])
        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)

        # Set up controllers
        self._arm_controller = OperationalSpaceController(
            physics=self._physics,
            joints=self._arm.joints,
            eef_site=self._arm.eef_site,
            min_effort=-150.0,
            max_effort=150.0,
            kp=200,
            ko=200,
            kv=50,
            vmax_xyz=1.0,
            vmax_abg=2.0,
        )
        if self._gripper_model:
            self._gripper_controller = JointEffortController(
                physics=self._physics,
                joints=[self._gripper.joint],
                min_effort=np.array([-5.0]),
                max_effort=np.array([5.0]),
            )

        self._timestep = self._physics.model.opt.timestep
        self._viewer = None
        self._step_start = None
        self._counter = 0
        self._max_steps = max_steps
        
        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self.spin, daemon=True)
        self.spin_thread.start()

    def spin(self):
        rclpy.spin(self.node)

    def _get_obs(self) -> np.ndarray:
        joint_positions = self._physics.bind(self._arm.joints).qpos
        joint_velocities = self._physics.bind(self._arm.joints).qvel
        ee_pos = self._physics.bind(self._arm.eef_site).xpos
        ee_xmat = self._physics.bind(self._arm.eef_site).xmat
        ee_xquat = mat2quat(ee_xmat.reshape(3, 3))  # [x, y, z, w]
        
        target_pos = self._target.get_mocap_pose(self._physics)[:3]
        target_quat = self._target.get_mocap_pose(self._physics)[3:]  # [x, y, z, w]

        # Compute position and orientation errors
        position_error = ee_pos - target_pos
        ee_rotation = R.from_quat(ee_xquat)
        target_rotation = R.from_quat(target_quat)
        orientation_error = (target_rotation * ee_rotation.inv()).as_rotvec()
        
        return np.concatenate([joint_positions, joint_velocities, position_error, orientation_error])
    
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._counter = 0
        with self._physics.reset_context():
            self._physics.bind(self._arm.joints).qpos = self._initial_pose
            self._target.set_mocap_pose(self._physics, position=self._target_pose[:3], quaternion=self._target_pose[3:])

            if self._task in {"reach_box_static", "reach_box_dynamic"}:
                self._target.set_box_pose(self._physics, position=self._target_pose[:3], quaternion=self._target_pose[3:])
                self._init_box_pose = self._target.get_box_pose(self._physics)
        
        observation = self._get_obs()
        return observation, {}


    def step(self, action=None):
        # Retrieve action if not provided
        if action is None:
            while self.received_action is None:
                rclpy.spin_once(self.node)
            action = self.received_action
            self.received_action = None

        if self._env_mode == "test":
            time.sleep(0.1)

        # Increment step counter
        self._counter += 1

        # Set gripper action if necessary
        self._gripper_action = "open"  # Modify as needed based on your implementation

        # Apply action based on action mode
        if self._action_mode == 'cartesian':
            self._apply_cartesian_action(action)
        elif self._action_mode == 'joint':
            self._apply_joint_action(action)
        else:
            raise ValueError(f"Unknown action mode: {self._action_mode}")

        # Step the simulation
        self._physics.step()

        # Get observations and compute reward
        observation = self._get_obs()
        position_error = observation[12:15]
        orientation_error = observation[15:18]
        position_distance = np.linalg.norm(position_error)
        orientation_distance = np.linalg.norm(orientation_error)
        reward, terminated = self._compute_reward_and_done(position_distance, orientation_distance)


        # Gripper control
        if self._gripper_model:
            gripper_effort = np.array([5000.0]) if self._gripper_action == "close" else np.array([-5000.0])
            self._gripper_controller.run(gripper_effort)


        # Check for maximum steps
        if self._counter >= self._max_steps:
            print("Max step reached, resetting environment.")
            terminated = True

        # Render if needed
        if self._render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, {}

    def _apply_cartesian_action(self, action):
        # Get current end-effector pose
        ee_pos = self._physics.bind(self._arm.eef_site).xpos
        ee_xmat = self._physics.bind(self._arm.eef_site).xmat
        ee_xquat = mat2quat(ee_xmat.reshape(3, 3))  # [x, y, z, w]

        # Split the action into position and orientation deltas
        position_delta = action[:3]
        orientation_delta = action[3:]

        # Update position
        new_target_position = ee_pos + position_delta

        # Update orientation using small angle approximation
        ee_rotation = R.from_quat(ee_xquat)
        delta_rotation = R.from_rotvec(orientation_delta)
        new_target_rotation = delta_rotation * ee_rotation
        new_target_quat = new_target_rotation.as_quat()

        # Set the controller's target pose
        target_pose = np.concatenate([new_target_position, new_target_quat])

        # Run the controller to move the arm based on the new target pose
        self._arm_controller.run(target_pose)

    def _apply_joint_action(self, action):
        # Apply action to joint positions
        joint_positions = self._physics.bind(self._arm.joints).qpos
        new_joint_positions = joint_positions + action[:len(joint_positions)]
        new_joint_positions = np.clip(new_joint_positions, -np.pi, np.pi)
        self._physics.bind(self._arm.joints).qpos = new_joint_positions

    def render(self):
        if self._render_mode == "rgb_array":
            return self._physics.render()
        elif self._render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        if self._viewer is None and self._render_mode == "human":
            self._viewer = mujoco.viewer.launch_passive(
                self._physics.model.ptr,
                self._physics.data.ptr,
            )
        if self._step_start is None:
            self._step_start = time.time()

        self._viewer.sync()
        time_until_next_step = self._timestep - (time.time() - self._step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        self._step_start = time.time()

    def _compute_reward_and_done(self, position_distance, orientation_distance):
        position_threshold = 0.02
        orientation_threshold = 0.1
        total_distance = position_distance #+ orientation_distance
        reward = -total_distance 
        terminated = (position_distance < position_threshold) #and (orientation_distance < orientation_threshold)
        if terminated:
            print("TARGET REACHED..")
        return reward, terminated

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
        self.node.destroy_node()
        rclpy.shutdown()
        self.spin_thread.join()
