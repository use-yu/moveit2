from launch import LaunchDescription
import launch_ros.actions
import json
import os

# 当前 launch 文件所在目录，例如：.../dobot_bringup_v4/launch/
cur_path = os.path.split(os.path.realpath(__file__))[0] + '/'
# 配置目录：.../dobot_bringup_v4/config
cur_config_path = cur_path + '../config'
# 参数文件路径：.../dobot_bringup_v4/config/param.json
cur_json_path = os.path.join(cur_config_path, 'param.json')

# 读取 JSON 配置文件（包含机器人数量、当前机器人索引、每台机器人的参数）
with open(cur_json_path, 'r') as file:
    json_data = json.load(file)

# 系统中机器人总数量
robot_number = json_data["robot_number"]
# 当前要启动的机器人编号（通常从 1 开始）
current_robot = json_data['current_robot']
# 所有机器人节点配置列表
node_info = json_data["node_info"]

def generate_launch_description():
    actions = []

    # 单机模式：按 current_robot 选择一个节点
    # 多机模式：按 node_info 前 robot_number 项依次启动
    if robot_number <= 1:
        indices = [max(current_robot - 1, 0)]
    else:
        indices = list(range(min(robot_number, len(node_info))))

    for idx in indices:
        if idx >= len(node_info):
            continue

        info = node_info[idx]
        ip_address = info["ip_address"]
        robot_type = info["robot_type"]
        trajectory_duration = info["trajectory_duration"]
        robot_node_name = info["robot_node_name"]
        ros_prefix = robot_node_name if robot_node_name else f"dobot_{idx + 1}"

        dobot_ros2_params = [
            {"robot_ip_address": ip_address},
            {"robot_type": robot_type},
            {"trajectory_duration": trajectory_duration},
            {"robot_node_name": ros_prefix},
            {"robot_number": robot_number},
        ]

        actions.append(
            launch_ros.actions.Node(
                package='cr_robot_ros2',
                executable='cr_robot_ros2_node',
                name=ros_prefix,
                output='screen',
                parameters=dobot_ros2_params,
                remappings=[
                    ("joint_states_robot", f"{ros_prefix}/joint_states_robot"),
                    ("dobot_msgs_v4/msg/RobotStatus",
                     f"{ros_prefix}/dobot_msgs_v4/msg/RobotStatus"),
                    ("dobot_msgs_v4/msg/ToolVectorActual",
                     f"{ros_prefix}/dobot_msgs_v4/msg/ToolVectorActual"),
                ]
                # respawn=True
            )
        )

    return LaunchDescription(actions)
