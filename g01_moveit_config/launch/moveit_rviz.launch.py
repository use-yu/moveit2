import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launch_utils import DeclareBooleanLaunchArg, add_debuggable_node


def _load_topics():
    cfg_path = os.path.join(
        get_package_share_directory("g01_moveit_config"),
        "config",
        "real_hardware_topics.yaml",
    )
    import yaml  # type: ignore

    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "state_topic" not in data:
        raise RuntimeError(f"Missing state_topic in {cfg_path}")
    return str(data["state_topic"]), str(data.get("command_topic", ""))


def _launch_setup(context):
    state_topic, command_topic = _load_topics()

    hardware_plugin = (
        "g01_topic_hardware/G01TopicSystem"
        if LaunchConfiguration("use_real_hardware").perform(context) == "true"
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
        .moveit_cpp("config/moveit_cpp.yaml")
        .to_moveit_configs()
    )

    ld = LaunchDescription()
    ld.add_action(DeclareBooleanLaunchArg("debug", default_value=False))
    ld.add_action(
        DeclareLaunchArgument(
            "rviz_config",
            default_value=str(moveit_config.package_path / "config/moveit.rviz"),
        )
    )

    add_debuggable_node(
        ld,
        package="rviz2",
        executable="rviz2",
        output="log",
        respawn=False,
        arguments=["-d", LaunchConfiguration("rviz_config")],
        parameters=[
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
        ],
        remappings=[("joint_states", state_topic)],
    )
    return ld.entities


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_real_hardware", default_value="false"),
            OpaqueFunction(function=_launch_setup),
        ]
    )
