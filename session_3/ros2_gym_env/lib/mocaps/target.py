import numpy as np

class Target(object):
    """
    A class representing a target with motion capture capabilities.
    """

    def __init__(self, mjcf_root, arena, gripper_offset):
        """
        Initializes a new instance of the Target class.

        Args:
            mjcf_root: The root element of the MJCF model.
            arena: The arena in which the box will be attached.
            box_size: The size of the box (default is [0.02, 0.02, 0.02]).
            box_rgba: The RGBA color of the box (default is green).
        """
        self._mjcf_root = mjcf_root
        self._arena = arena

        # Add a mocap body to the worldbody
        self._mocap = self._mjcf_root.worldbody.add("body", name="mocap", mocap=True)
        self._mocap.add(
            "geom",
            type="box",
            size=[0.015] * 3,
            rgba=[1, 0, 0, 0.2],
            conaffinity=0,
            contype=0,
        )

        # Attach a free-moving box to the arena
        self._box_size = [0.02, 0.02, 0.02]
        self._box_rgba = [0, 1, 0, 1.0]
        self._box = self._arena._mjcf_model.worldbody.add("body", name="box", pos=[0, 0, 0])
        self._box.add("geom", type="box", size=self._box_size, rgba=self._box_rgba)
        self._box.add("freejoint", name='box_freejoint')

        self._gripper_offset = gripper_offset

    @property
    def mjcf_root(self):
        """Gets the root element of the MJCF model."""
        return self._mjcf_root

    @property
    def mocap(self):
        """Gets the mocap body."""
        return self._mocap

    def set_mocap_pose(self, physics, position=None, quaternion=None):
        """
        Sets the pose of the mocap body.

        Args:
            physics: The physics simulation.
            position: The position of the mocap body.
            quaternion: The quaternion orientation of the mocap body.
        """
        if quaternion is not None:
            # Flip quaternion from xyzw to wxyz format
            quaternion_wxyz = np.roll(np.array(quaternion), 1)
        else:
            quaternion_wxyz = None

        if position is not None:
            physics.bind(self.mocap).mocap_pos[:] = position
        if quaternion_wxyz is not None:
            physics.bind(self.mocap).mocap_quat[:] = quaternion_wxyz

    def set_box_pose(self, physics, position=None, quaternion=None):
        """
        Sets the pose of the box.

        Args:
            physics: The physics simulation.
            position: The position of the box.
            quaternion: The quaternion orientation of the box.
        """
        # Bind the box's freejoint to the physics simulation
        box_joint = self._box.find('joint', 'box_freejoint')
        box_joint_physics = physics.bind(box_joint)

        if quaternion is not None:
            # Flip quaternion from xyzw to wxyz format
            quaternion_wxyz = np.roll(np.array(quaternion), 1)
        else:
            quaternion_wxyz = None

        if position is not None:
            box_joint_physics.qpos[:3] = position
        if quaternion_wxyz is not None:
            box_joint_physics.qpos[3:] = quaternion_wxyz

    def get_mocap_pose(self, physics):
        """
        Retrieves the pose of the mocap body.

        Args:
            physics: The physics simulation.

        Returns:
            pose: The position and orientation (quaternion) of the mocap body.
        """
        position = physics.bind(self.mocap).mocap_pos[:] + self._gripper_offset
        quaternion_wxyz = physics.bind(self.mocap).mocap_quat[:]

        # Flip quaternion from wxyz to xyzw format
        quaternion = np.roll(np.array(quaternion_wxyz), -1)

        pose = np.concatenate([position, quaternion])
        return pose

    def get_box_pose(self, physics):
        """
        Retrieves the pose of the box.

        Args:
            physics: The physics simulation.

        Returns:
            pose: The position and orientation (quaternion) of the box.
        """
        # Bind the box's freejoint to the physics simulation
        box_joint = self._box.find('joint', 'box_freejoint')
        box_joint_physics = physics.bind(box_joint)

        # Get position and quaternion
        position = box_joint_physics.qpos[:3]
        quaternion_wxyz = box_joint_physics.qpos[3:]

        # Flip quaternion from wxyz to xyzw format
        quaternion = np.roll(np.array(quaternion_wxyz), -1)

        pose = np.concatenate([position, quaternion])
        return pose
