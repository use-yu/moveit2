from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("wheel_robot", package_name="qj2m_moveit_config").to_moveit_configs()
    return LaunchDescription([
        Node(
            package="hello_moveit",
            executable="hello_left_arm",
            output="screen",
            parameters=[moveit_config.to_dict()],
        ),
    ])
