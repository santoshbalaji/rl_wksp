import mujoco

# Loading a specific model description as an imported module.
from robot_descriptions import panda_mj_description
model = mujoco.MjModel.from_xml_path(panda_mj_description.MJCF_PATH)

# Directly loading an instance of MjModel.
from robot_descriptions.loaders.mujoco import load_robot_description
model = load_robot_description("panda_mj_description")

# Loading a variant of the model, e.g. panda without a gripper.
model = load_robot_description("panda_mj_description", variant="panda_nohand")
