from setuptools import setup
import os
from glob import glob

package_name = 'ros2_gym_env'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name, 'lib', 'lib.helpers', 'lib.another', 'lib.arenas', 'lib.controllers', 'lib.helpers', 'lib.mocaps', 'lib.props', 'lib.robots', 'lib.utils'],
    install_requires=['setuptools', 'gymnasium', 'numpy', 'rclpy'],
    zip_safe=True,
    maintainer='Shalman Khan',
    maintainer_email='shabashkhan@artc.a-star.edu.sg',
    description='A ROS 2 compatible Gymnasium environment for reinforcement learning with robotic manipulation',
    license='TBD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ros2_gym_env_node = ros2_gym_env.ros2_gym_env:main',
        ],
    },
    data_files=[
        # ROS2 package index
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        # Package XML
        ('share/' + package_name, ['package.xml']),
        
        # Install AG95 assets including meshes in the share directory
        (os.path.join('share', package_name, 'lib/assets/robots/ag95'), [
            'lib/assets/robots/ag95/ag95.xml',
            'lib/assets/robots/ag95/scene.xml'
        ]),
        (os.path.join('share', package_name, 'lib/assets/robots/ag95/meshes'), glob('lib/assets/robots/ag95/meshes/*.stl')),

        # Include UR5e assets
        (os.path.join('share', package_name, 'lib/assets/robots/ur5e'), [
            'lib/assets/robots/ur5e/ur5e.xml',
            'lib/assets/robots/ur5e/scene.xml',
            'lib/assets/robots/ur5e/ur5e.png'
        ]),
        (os.path.join('share', package_name, 'lib/assets/robots/ur5e/assets'), glob('lib/assets/robots/ur5e/assets/*.obj')),
        # Include Aubo_I5 assets
        (os.path.join('share', package_name, 'lib/assets/robots/aubo_i5'), [
            'lib/assets/robots/aubo_i5/aubo_i5.xml',
            'lib/assets/robots/aubo_i5/scene.xml',
        ]),
        (os.path.join('share', package_name, 'lib/assets/robots/aubo_i5/meshes/visual'), glob('lib/assets/robots/aubo_i5/meshes/visual/*.obj')),
        (os.path.join('share', package_name, 'lib/assets/robots/aubo_i5/meshes/collision'), glob('lib/assets/robots/aubo_i5/meshes/collision/*.stl')),

    ],
    include_package_data=True,
)
