"""move_group：在默认 MoveIt 参数基础上加载 chomp_optimizer.yaml（供 chomp/OptimizerAdapter 使用）。"""
import os
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launch_utils import DeclareBooleanLaunchArg, add_debuggable_node
from launch_param_builder.utils import load_yaml

def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("wheel_robot", package_name="qj2m_moveit_config").to_moveit_configs()
 
    chomp_path = Path(moveit_config.package_path) / "config" / "chomp_optimizer.yaml"
    chomp_params = load_yaml(chomp_path) if chomp_path.is_file() else {}
    ld = LaunchDescription()
    ld.add_action(DeclareBooleanLaunchArg("debug", default_value=False))
    ld.add_action(DeclareBooleanLaunchArg("allow_trajectory_execution", default_value=True))
    ld.add_action(DeclareBooleanLaunchArg("publish_monitored_planning_scene", default_value=True))
    ld.add_action(
        DeclareLaunchArgument(
            "capabilities",
            default_value=moveit_config.move_group_capabilities["capabilities"],
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            "disable_capabilities",
            default_value=moveit_config.move_group_capabilities["disable_capabilities"],
        )
    )
    ld.add_action(DeclareBooleanLaunchArg("monitor_dynamics", default_value=False))
    should_publish = LaunchConfiguration("publish_monitored_planning_scene")
    move_group_configuration = {
        "publish_robot_description_semantic": True,
        "allow_trajectory_execution": LaunchConfiguration("allow_trajectory_execution"),
        "capabilities": ParameterValue(LaunchConfiguration("capabilities"), value_type=str),
        "disable_capabilities": ParameterValue(LaunchConfiguration("disable_capabilities"), value_type=str),
        "publish_planning_scene": should_publish,
        "publish_geometry_updates": should_publish,
        "publish_state_updates": should_publish,
        "publish_transforms_updates": should_publish,
        "monitor_dynamics": False,
    }
    move_group_params = [
        moveit_config.to_dict(),
        chomp_params,
        move_group_configuration,
    ]
    gdb_cmds = moveit_config.package_path / "launch" / "gdb_settings.gdb"
    add_debuggable_node(
        ld,
        package="moveit_ros_move_group",
        executable="move_group",
        commands_file=str(gdb_cmds) if gdb_cmds.is_file() else None,
        output="screen",
        parameters=move_group_params,
        extra_debug_args=["--debug"],
        additional_env={"DISPLAY": os.environ.get("DISPLAY", "")},
    )
    return ld
