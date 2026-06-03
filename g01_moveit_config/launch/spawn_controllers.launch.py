from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_real_hardware",
                default_value="false",
                description="If true, skip joint_state_broadcaster (state comes from state_topic).",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )


def _launch_setup(context):
    use_real = LaunchConfiguration("use_real_hardware").perform(context) == "true"
    moveit_config = (
        MoveItConfigsBuilder("G01", package_name="g01_moveit_config").to_moveit_configs()
    )
    controller_names = moveit_config.trajectory_execution.get(
        "moveit_simple_controller_manager", {}
    ).get("controller_names", [])

    # 单个 spawner 按顺序激活，避免并行 spawner 导致 right_arm 等激活失败
    spawner_args = list(controller_names) + [
        "--controller-manager-timeout",
        "60",
    ]
    if not use_real:
        spawner_args = ["joint_state_broadcaster"] + spawner_args

    return [
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=spawner_args,
            output="screen",
        )
    ]
