import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PythonExpression

from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch


def generate_launch_description():
    use_real_arg = DeclareLaunchArgument(
        "use_real_hardware",
        default_value="false",
    )

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
    state_topic = str(data["state_topic"])
    command_topic = str(data["command_topic"])

    hardware_plugin = PythonExpression(
        [
            "'g01_topic_hardware/G01TopicSystem' if '",
            LaunchConfiguration("use_real_hardware", default="false"),
            "' == 'true' else 'mock_components/GenericSystem'",
        ]
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

    ld = generate_move_group_launch(moveit_config)
    ld.add_action(use_real_arg)
    return ld
