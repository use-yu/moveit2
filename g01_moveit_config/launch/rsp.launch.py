import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from moveit_configs_utils import MoveItConfigsBuilder


def _load_topics():
    cfg_path = os.path.join(
        get_package_share_directory("g01_moveit_config"),
        "config",
        "real_hardware_topics.yaml",
    )
    import yaml  # type: ignore

    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "state_topic" not in data or "command_topic" not in data:
        raise RuntimeError(
            f"Missing keys in {cfg_path}. Required: state_topic, command_topic"
        )
    return str(data["state_topic"]), str(data["command_topic"])


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_real_hardware", default_value="false"),
            DeclareLaunchArgument("publish_frequency", default_value="15.0"),
            OpaqueFunction(function=_launch_setup),
        ]
    )


def _launch_setup(context):
    state_topic, command_topic = _load_topics()
    use_real = LaunchConfiguration("use_real_hardware").perform(context) == "true"

    hardware_plugin = (
        "g01_topic_hardware/G01TopicSystem"
        if use_real
        else "mock_components/GenericSystem"
    )

    moveit_config = (
        MoveItConfigsBuilder("G01", package_name="g01_moveit_config")
        .robot_description(
            file_path="config/G01.urdf.xacro",
            mappings={
                "ros2_control_hardware_plugin": hardware_plugin,
                "ros2_control_state_topic": state_topic,
                "ros2_control_command_topic": command_topic,
            },
        )
        .to_moveit_configs()
    )

    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            respawn=True,
            output="screen",
            remappings=[("joint_states", state_topic)],
            parameters=[
                moveit_config.robot_description,
                {
                    "publish_frequency": float(
                        LaunchConfiguration("publish_frequency").perform(context)
                    ),
                },
            ],
        )
    ]
