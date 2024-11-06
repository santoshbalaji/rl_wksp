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

def get_ur5e_xml_path():
    package_share_directory = get_package_share_directory('ros2_gym_env')
    return os.path.join(package_share_directory, 'assets/robots/ur5e/ur5e.xml')

def get_ag95_xml_path():
    package_share_directory = get_package_share_directory('ros2_gym_env')
    return os.path.join(package_share_directory, 'assets/robots/ag95/ag95.xml')

class UR5eGrpEnvROS2(Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": None,
    }

    def __init__(self, render_mode=None):
        super(UR5eGrpEnvROS2, self).__init__()
        
        # Initialize ROS 2 Node
        rclpy.init(args=None)
        self.node = rclpy.create_node('ur5e_grp_env_ros2')
        
        # Define action and observation spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-0.05, high=0.05, shape=(6,), dtype=np.float64  
        )
        
        # ROS 2 publishers and subscribers
        self.ee_pos_publisher = self.node.create_publisher(Float64MultiArray, 'ee_position', 10)
        self.observation_publisher = self.node.create_publisher(Float64MultiArray, 'observation', 10)
        self.action_subscription = self.node.create_subscription(
            Float64MultiArray, 'action_topic', self.action_callback, 10)
        
        self.received_action = None
        self._render_mode = render_mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        
        # Create MJCF model components
        self._arena = StandardArena()
        self._target = Target(self._arena.mjcf_model, self._arena, [0.0, 0.0, 0.0])

        # Load UR5e arm and attach it to the arena
        self._arm = Arm(
            xml_path=get_ur5e_xml_path(),
            eef_site_name='eef_site',
            attachment_site_name='attachment_site'
        )
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

        # Uncommented gripper setup and controllers
        # self._gripper = AG95()
        # self._arm.attach_tool(self._gripper.mjcf_model, pos=[0, 0, 0], quat=[0, 0, 0, 1])
        
        # self._gripper_controller = JointEffortController(
        #     physics=self._physics,
        #     joints=[self._gripper.joint],
        #     min_effort=np.array([-5.0]),
        #     max_effort=np.array([5.0]),
        # )

        # Timekeeping
        self._timestep = self._physics.model.opt.timestep
        self._viewer = None
        self._step_start = None
        self._counter = 0
        self._max_steps = 500
        
        # Start spinning in a separate thread for ROS 2
        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self.spin, daemon=True)
        self.spin_thread.start()

    def spin(self):
        rclpy.spin(self.node)
        
    def action_callback(self, msg):
        self.received_action = np.array(msg.data, dtype=np.float64)

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
            self._physics.bind(self._arm.joints).qpos = [0.0, -1.5707, 1.5707, -1.5707, -1.5707, 0.0]
            self._target.set_mocap_pose(self._physics, position=[0.7, 0.0, 0.18], quaternion=[0, 0, 0, 1])
            self._target.set_box_pose(self._physics, position=[0.7, 0.0, 0.02], quaternion=[0, 0, 0, 1])
            self._init_box_pose = self._target.get_box_pose(self._physics)
        
        observation = self._get_obs()
        return observation, {}

    def step(self, action=None):
        if action is None:
            while self.received_action is None:
                rclpy.spin_once(self.node)
                # time.sleep(0.1)
            action = self.received_action
            self.received_action = None
        time.sleep(0.1)
        self._counter += 1

        # Apply action and step simulation
        joint_positions = self._physics.bind(self._arm.joints).qpos
        new_joint_positions = joint_positions + action[:len(joint_positions)]
        self._physics.bind(self._arm.joints).qpos = np.clip(new_joint_positions, -np.pi, np.pi)
        self._physics.step()

        # Get observations and reward
        observation = self._get_obs()
        position_error = observation[12:15]
        orientation_error = observation[15:18]
        position_distance = np.linalg.norm(position_error)
        orientation_distance = np.linalg.norm(orientation_error)
        reward, terminated = self._compute_reward_and_done(position_distance, orientation_distance)
        
        # Gripper control (commented out for now)
        # gripper_effort = np.array([5000.0]) if self._gripper_action == "close" else np.array([-5000.0])
        # self._gripper_controller.run(gripper_effort)
        if self._counter >= self._max_steps:
            print("Max step reached, resetting environment.")
            terminated = True

        if self._render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, {}

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
        position_threshold = 0.03
        orientation_threshold = 0.1  # Radians
        total_distance = position_distance + orientation_distance
        reward = -position_distance  # Use only position distance for reward
        terminated = position_distance < position_threshold  # Terminate based on position threshold
        if terminated:
            print("TARGET REACHED..")
        return reward, terminated

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
        self.node.destroy_node()
        rclpy.shutdown()
        self.spin_thread.join()
