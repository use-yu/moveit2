#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 关节空间规划不到1s，给末端位姿4s左右
"""
G01 MoveIt 演示脚本

功能（按顺序执行）：
  1. 向规划场景添加半透明「深框」碰撞体（底板 + 四面墙，顶部敞开）
  2. 对 ACTIVE_GROUP 做关节空间规划并执行（例如 dual_arm）
  3. 对 left_body 组做末端位姿（L6）规划并执行
  4. 程序退出时在 finally 中自动移除深框

前提：
  - 已启动 move_group： ros2 launch g01_moveit_config demo.launch.py
  - 本节点与 move_group 在同一 ROS 域
  ros2 launch g01_moveit_config demo.launch.py use_real_hardware:=true

运行（Python）：
  colcon build --packages-select hello_moveit && source install/setup.bash
  ros2 run hello_moveit g01.py

实际测试遇到的问题：
1. moveit关节空间规划时由于时间参数化（TOTG）有时候规划成功，执行失败
可以通过修改 ompl_planning.yaml 中的 longest_valid_segment_fraction 参数来解决，值越小，规划时间越长，但是规划成功率越高
本代码采用设置一个较好的初始构型，再加上失败重新规划方法来解决这个问题，相比改参数这样求解速度更快

2. 做末端直线运动时由于机械臂构型不同，实际有解但有时候会规划失败，
本代码采用求解多个逆解，再从多个逆解中选择一个直线规划可以求解成功的逆解，保证构型合理

3. moveit先plan再execute时，耗时较长
本代码采用先plan+execute的方法

腰部30度/0.523598 放置和交换
"""
# solution_rad: r_arm_joint1=-1.944364063, r_arm_joint2=-1.597215489, r_arm_joint3=-0.565983840, r_arm_joint4=-0.979311147, r_arm_joint5=0.851759446, r_arm_joint6=3.140000000
# solution_deg: r_arm_joint1=-111.403854650, r_arm_joint2=-91.513706512, r_arm_joint3=-32.428485315, r_arm_joint4=-56.110395575, r_arm_joint5=48.802221443, r_arm_joint6=179.908747671
# solution_rad: l_arm_joint1=0.070556798, l_arm_joint2=-1.414919441, l_arm_joint3=-1.053385853, l_arm_joint4=-0.663103266, l_arm_joint5=0.585734181, l_arm_joint6=-0.010876821
# solution_deg: l_arm_joint1=4.042606739, l_arm_joint2=-81.068912306, l_arm_joint3=-60.354563592, l_arm_joint4=-37.993018506, l_arm_joint5=33.560096466, l_arm_joint6=-0.623195926
from __future__ import annotations

import copy
import json
import math
import multiprocessing
import random
import re
import socket
import sys
import time
from typing import Iterable, Sequence

import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import Point, Pose, PoseStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    CollisionObject,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    ObjectColor,
    OrientationConstraint,
    PlanningScene,
    PositionConstraint,
    RobotState,
    RobotTrajectory,
)
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath, GetPositionFK, GetPositionIK
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker
from dobot_msgs_v4.srv import SetToolPower

# =============================================================================
# 用户可调参数（改这里即可，无需动下面逻辑）
# =============================================================================
GREEN = '\033[32m'
RESET = '\033[0m'
# 第一步：关节空间规划使用哪个 SRDF 组
ACTIVE_GROUP = "dual_arm_y"

# 第二步：末端位姿规划组与目标（连杆 L6，坐标系 base_link）
POSE_GROUP = "right_arm"

PLAN_FRAME = "r_base_link"  # 末端位姿约束坐标系

# EE_LINK 必须在 末端linkL6 下游、用 fixed joint 连上去的子 link（例如 l_tool）
EE_LINK = "r_tool"

# r_base_link r_tool
# EE_POSE2 = dict(
#     x=-0.865,
#     y=0.240,
#     z=0.103,
#     roll=-2.125,
#     pitch=-0.131,
#     yaw=1.044,
# )

# L6
# EE_POSE2 = dict(
#     x=-0.75,
#     y=-0.35,
#     z=0.0,
#     roll=-math.pi / 4,
#     pitch=-math.pi / 4,
#     yaw=-math.pi,
# )
# R6
# EE_POSE2 = dict(
#     x=-0.788,
#     y=-0.101,
#     z=0.054,
#     roll=-1.603,
#     pitch=-1.340,
#     yaw=3.071,
# )
# 各组的关节目标 [rad]（键名须与 URDF/SRDF 一致）
JOINT_TARGETS = {
    "body": {
        # "base_joint1": 1.0,
        # "base_joint2": 0.0,
        # "body_joint1": 0.0,
        "body_joint2": 0.0,
    },
    # 预备位：底盘略前伸、躯干抬起、双臂零位（与 hello_go1.cpp dual_arm 一致）
    "dual_arm": {
        "base_joint1": 1.25,
        "base_joint2": 0.0,
        "body_joint1": -0.25,
        "body_joint2": 1.1,
        "l_arm_joint1": -20 * math.pi / 180,
        "l_arm_joint2": -102 * math.pi / 180,
        "l_arm_joint3": -92 * math.pi / 180,
        "l_arm_joint4": 137 * math.pi / 180,
        "l_arm_joint5": -0 * math.pi,
        "l_arm_joint6": -0 * math.pi / 180,
        "r_arm_joint1": 80 * math.pi / 180,
        "r_arm_joint2": -102 * math.pi / 180,
        "r_arm_joint3": -92 * math.pi / 180,
        "r_arm_joint4": 137 * math.pi / 180,
        "r_arm_joint5": -0 * math.pi,
        "r_arm_joint6": -0 * math.pi / 180,
    },
    "dual_arm_y": {
        "l_arm_joint1": -20 * math.pi / 180,
        "l_arm_joint2": -102 * math.pi / 180,
        "l_arm_joint3": -92 * math.pi / 180,
        "l_arm_joint4": 137 * math.pi / 180,
        "l_arm_joint5": -0 * math.pi,
        "l_arm_joint6": -0 * math.pi / 180,
        "r_arm_joint1": 80 * math.pi / 180,
        "r_arm_joint2": -102 * math.pi / 180,
        "r_arm_joint3": -92 * math.pi / 180,
        "r_arm_joint4": 137 * math.pi / 180,
        "r_arm_joint5": -0 * math.pi,
        "r_arm_joint6": -0 * math.pi / 180,
    },
    "left_body": {
        # "body_joint1": 0.0,
        "body_joint2": 1.313,
        "l_arm_joint1": 1.8697,
        "l_arm_joint2": 0.2,
        "l_arm_joint3": 0.135997,
        "l_arm_joint4": 1.23459,
        "l_arm_joint5": 2.1201,
        "l_arm_joint6": -1.5702,
    },
    "left": {
        "base_joint1": 1.25,
        "base_joint2": 0.0,
        "body_joint1": 0.0,
        "body_joint2": 1.313,
        "l_arm_joint1": 1.8697,
        "l_arm_joint2": 0.2,
        "l_arm_joint3": 0.135997,
        "l_arm_joint4": 1.23459,
        "l_arm_joint5": 2.1201,
        "l_arm_joint6": -1.5702,
    },
    "right_body": {
        # "body_joint1": 0.0,
        "body_joint2": -1.313,
        "r_arm_joint1": -1.8697,
        "r_arm_joint2": 0.2,
        "r_arm_joint3": 0.135997,
        "r_arm_joint4": 1.23459,
        "r_arm_joint5": 2.1201,
        "r_arm_joint6": -1.5702,
    },
    "right_arm": {
        "r_arm_joint1": -1.8697,
        "r_arm_joint2": 0.2,
        "r_arm_joint3": 0.135997,
        "r_arm_joint4": 1.23459,
        "r_arm_joint5": 2.1201,
        "r_arm_joint6": -1.5702,
    },
    "left_arm": {
        "l_arm_joint1": 1.8697,
        "l_arm_joint2": 0.2,
        "l_arm_joint3": 0.135997,
        "l_arm_joint4": 1.23459,
        "l_arm_joint5": 2.1201,
        "l_arm_joint6": -1.5702,
    },
}

# 位姿规划时作为起始状态的关节（不含底盘，与 left_body 组一致）
POSE_START_JOINTS = list(JOINT_TARGETS.get(POSE_GROUP, {}).keys())
if not POSE_START_JOINTS:
    raise KeyError(f"JOINT_TARGETS 中未找到 POSE_GROUP={POSE_GROUP} 的关节列表")


# 规划器
PLANNER_ID = "RRTConnect"
NUM_ATTEMPTS = 20
PLAN_TIME_SEC = 10.0
MOVE_MAX_RETRIES = 5  # move_action 失败后再试几次（应对 INVALID_MOTION_PLAN 等 OMPL 随机性失败）

# 执行速度缩放（同时用于 velocity / acceleration；范围 0~1，越大越快）
DEFAULT_SPEED_SCALE = 0.5

# 深框障碍物（相对于 base_link 发布）
SCENE_FRAME = "base_link"
FRAME_ID = "深框"
FRAME_SIZE = (1.18, 1.18, 0.58)  # 深框整体外尺寸：长×宽×高 [m]
WALL_T = 0.09  # 壁厚向内部收缩，外轮廓尺寸保持 FRAME_SIZE
FRAME_CENTER = (-0.81 - 0.005, -0.35, -0.08)  # 深框整体外轮廓中心，相对于 base_link [m]
FRAME_RPY_DEG = (90.0, -0.0, 180.0)  # 深框整体姿态，相对于 base_link [degree]
FRAME_COLOR = ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.5)
FRAME_CUTOFF_ID = "深框隔离面"
FRAME_CUTOFF_THICKNESS = 0.01  # 薄隔离面厚度 [m]，沿深框局部 z 轴；在 SCENE_FRAME 中等价于 y 方向厚度
FRAME_CUTOFF_COLOR = ColorRGBA(r=1.0, g=0.25, b=0.1, a=1.0)

# 位姿标记圆柱（Marker 保持仅显示；q_pre OMPL 阶段另建同尺寸临时碰撞体）
CYLINDER_MARKER_ID = "ee_pose_cylinder"
CYLINDER_MARKER_NS = "g01_pose_cylinder"
CYLINDER_MARKER_TOPIC = "g01_pose_cylinder"
CYLINDER_DIAMETER = 0.15   # 直径 [m]
CYLINDER_HEIGHT = 0.06     # 高度 [m]（沿位姿局部 z 轴）
CYLINDER_COLOR = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.45)
GRASP_OBJECT_COLLISION_ID = "待抓取物体"
GRASP_OBJECT_COLLISION_COLOR = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
Z_AXIS_MARKER_ID = "ee_pose_z_axis"
Z_AXIS_LENGTH = CYLINDER_HEIGHT / 2.0  # 从圆柱中心到局部 +z 端面，不超出物体
Z_AXIS_COLOR = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)

# ROS 接口名（move_group 默认）
SVC_APPLY_SCENE = "apply_planning_scene"
SVC_CARTESIAN_PATH = "compute_cartesian_path"
SVC_COMPUTE_IK = "compute_ik"
SVC_COMPUTE_FK = "compute_fk"
LEFT_TOOL_COMMAND_SERVICE = "/g01/left/tool_commands"
RIGHT_TOOL_COMMAND_SERVICE = "/g01/right/tool_commands"
GRASP_CMD_TOPIC = "/grasp_cmd"
GRASP_CMD_RESULT_TOPIC = "/grasp_cmd_result"
WAIST_BODY_GROUP = "body"
WAIST_BODY_JOINT = "body_joint2"
WAIST_RESET_ANGLE_RAD = math.radians(30.0)
ACT_MOVE_GROUP = "move_action"
ACT_EXEC_TRAJ = "execute_trajectory"

# 末端目标容差（与 MoveGroupInterface 默认一致）没用到
_POS_TOL = 1e-5
_ORI_TOL = 1e-5

# 笛卡尔直线运动参数
CART_EEF_STEP = 0.005     # 服务端 IK 离散步长（m）
CART_MIN_FRACTION = 0.99  # 接受的最小成功比例（<1 表示直线被截断）
CART_JUMP_THRESHOLD = 2.0  # 相对关节跳变阈值；0 表示关闭，容易接受绕腕跳解
CART_REVOLUTE_JUMP_THRESHOLD = 0.2  # 单步任一转动关节超过该值 [rad] 视为跳解
CART_PRISMATIC_JUMP_THRESHOLD = 0.02  # 单步任一移动关节超过该值 [m] 视为跳解
CART_JUMP_STEP_FACTOR = 3.0  # 单步关节空间距离超过平均值该倍数，直接淘汰
CART_MAX_POINT_FACTOR = 2.0  # 实际点数超过理论直线点数该倍数时，认为 IK 分支不稳
CART_MAX_POINT_EXTRA = 5

# 放置位关节目标 [rad]（数组顺序与 JOINT_TARGETS[group] 一致）。
# 单臂使用 yubei → fang 两段式放置：先到预备位，再做末端笛卡尔直线到放置位。
PLACE_JOINTS = {
    "left_body": [
        -0.01292, 1.015203, -0.712975, -0.550402, 1.300752, 0.543868, -0.143126, -0.338787
    ],
    "left": [
        1.25, 0.0, -0.01292, 1.015203, -0.712975, -0.550402, 1.300752, 0.543868, -0.143126, -0.338787
    ],
    "right_body": [
        -0.01292, 1.015203, -0.712975, -0.550402, 1.300752, 0.543868, -0.143126, -0.338787
    ],
    "right": [
        1.25, 0.0, -0.01292, 1.015203, -0.712975, -0.550402, 1.300752, 0.543868, -0.143126, -0.338787
    ],
    "dual_arm": [
        1.25, 0.0, -0.01292, 1.015203, -0.712975, -0.550402, 1.300752, 0.543868, -0.143126, -0.338787
        ,3.13, -1.419584, 1.578090, 1.370549, 1.672852, 0.588477
    ],
    "right_arm": {
        "yubei_j": [
            0.9902063012123108, -0.6606103777885437, -1.4372066259384155,
            -1.0370503664016724, -2.6659367084503174, -2.29879448,
        ],
        "fang_j": [
            1.15645432472229, -0.7307567596435547, -1.304102897644043,
            -1.1022419929504395, -2.5003864765167236, -2.29879448,
        ],
        "yubei": [
            0.9875547289848328, -0.6653450727462769, -1.4346156120300293,
            -1.0350865125656128, -2.6696741580963135, -2.2988312244415283,
        ],
        "fang": [
            1.153889536857605, -0.734650731086731, -1.302738904953003,
            -1.0998507738113403, -2.5040321350097656, -2.3008768558502197,
        ],
    },
    "left_arm": {
        "yubei_j": [
            -0.9828664660453796, 0.6264781951904297, 1.393784523010254,
            1.1182692050933838, 2.6711831092834473, 0.891918420791626,
        ],
        "fang_j": [
            -1.1497267484664917, 0.698388397693634, 1.2621753215789795,
            1.1796667575836182, 2.5052123069763184, 0.893162727355957,
        ],
        "yubei": [
            -0.9856565594673157, 0.6248205900192261, 1.403222680091858,
            1.110521912574768, 2.6683006286621094, 0.7956140637397766,
        ],
        "fang": [
            -1.1526858806610107, 0.6968747973442078, 1.2712609767913818,
            1.1721081733703613, 2.5021605491638184, 0.7968405485153198,
        ],
    },
}

# 抓取流程默认参数
PRE_GRASP_OFFSET = -0.1  # 预备抓取点沿末端坐标系 z 轴外移的距离 [m]
PLACE_SPEED_SCALE = 0.5     # 5/8 OMPL 运动到放置位的速度缩放
FIRST_RETURN_MODE = 2       # 1: 只反向直线回到 q_pre；2: 直线+OMPL 回到 1/8 初始位置
EXCHANGE_Q1 = [
    -1.57, -0.15, -1.578090, -1.370549, -1.672852, -0.588477,
    1.57, 0.15, 1.578090, 1.370549, 1.672852, 0.588477,
]
EXCHANGE_Q3 = {
    "right": [
        -0.9828664660453796, 0.6264781951904297, 1.393784523010254,
        1.1182692050933838, 2.6711831092834473, 0.891918420791626,
        1.57, 0.15, 1.578090, 1.370549, 1.672852, 0.588477,
    ],
    "left": [
        -1.57, -0.15, -1.578090, -1.370549, -1.672852, -0.588477,
        0.9902063012123108, -0.6606103777885437, -1.4372066259384155,
        -1.0370503664016724, -2.6659367084503174, -2.29879448,
    ],
}
EXCHANGE_Q2 = {
    "right": [
        0.113298830,-1.437132861,-0.988730272,-0.706701233,0.628914759,-0.533279984,
        -1.894577193, -1.514854650, -0.775937512, -0.852444980, 0.728284757, 0.001667991,
    ],
    "left": [
        1.894577193, 1.514854650, 0.775937512, 0.852444980, -0.728284757, -0.001667991,
        -0.113298828, 1.437132878, 0.988730233, 0.706701256, -0.628914758, -0.533279984,
    ],
}

# 视觉 TCP 配置
VISION_IP = "192.168.5.110"
VISION_PORT = 50000
VISION_TRIGGER_COMMAND = "p,1"
VISION_CONNECT_TIMEOUT = 3.0
VISION_RECV_TIMEOUT = 30.0
# 正则表达式，用来匹配字符串里的数字：
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
# 标定输入：x, y, z 单位米，四元数顺序为 w, x, y, z。
# 下面会转成平移单位为毫米的 4x4 矩阵，与 viewer pose 的毫米单位保持一致。
VISION_RIGHT_TRANSFORM_XYZ_WXYZ = [
    -0.152770, -0.139391, -0.109405, 0.648141, -0.270300, -0.657271, -0.273580
]
VISION_LEFT_TRANSFORM_XYZ_WXYZ = [
    0.154525, -0.138222, -0.190644, 0.654745, -0.277931, 0.647655, 0.273140
]

def _transform_xyz_wxyz_m_to_matrix_mm(transform: Sequence[float]) -> list[list[float]]:
    if len(transform) != 7:
        raise ValueError(f"标定参数需要 7 个数: x, y, z, qw, qx, qy, qz，实际 {len(transform)} 个")

    x_m, y_m, z_m, qw, qx, qy, qz = [float(value) for value in transform]
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1e-12:
        raise ValueError("标定四元数长度为 0")
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    rot = [
        [
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ],
        [
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ],
        [
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
    ]
    return [
        [rot[0][0], rot[0][1], rot[0][2], x_m * 1000.0],
        [rot[1][0], rot[1][1], rot[1][2], y_m * 1000.0],
        [rot[2][0], rot[2][1], rot[2][2], z_m * 1000.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


VISION_RIGHT_TRANSFORM_MM = _transform_xyz_wxyz_m_to_matrix_mm(VISION_RIGHT_TRANSFORM_XYZ_WXYZ)
VISION_LEFT_TRANSFORM_MM = _transform_xyz_wxyz_m_to_matrix_mm(VISION_LEFT_TRANSFORM_XYZ_WXYZ)

# IK 多解枚举参数（抓取流程选 IK 解 + approach 预检用）
IK_N_CANDIDATES = 200          # 总共尝试的 IK 种子数（含 1 次以当前关节为种子）
IK_SEED_PERTURB = math.pi/2     # 随机种子各关节的最大扰动幅度 [rad]，越大解越分散
IK_TIMEOUT_SEC = 0.2          # 每次 /compute_ik 超时（KDL 对边界姿态需更长收敛时间）
IK_RANDOM_SEED = 42           # 让 IK 多解枚举可复现；改成 None 则每次随机


# =============================================================================
# 几何与消息构造（纯函数，无 ROS 通信）
# =============================================================================


def connect_vision(log) -> socket.socket | None:
    """连接视觉 TCP，后续可复用同一个 socket 多次读取。"""
    try:
        sock = socket.create_connection(
            (VISION_IP, VISION_PORT),
            timeout=VISION_CONNECT_TIMEOUT,
        )
        sock.settimeout(VISION_RECV_TIMEOUT)
        log.info(f"已连接视觉 TCP：{VISION_IP}:{VISION_PORT}")
        return sock
    except OSError as exc:
        message = f"viewer 连接失败：{exc}"
        log.error(message)
        print(message)
        return None


def read_vision_pose(sock: socket.socket, log) -> list[tuple[int, list[float]]] | None:
    """解析视觉数据：第 1 个数忽略，后面每 8 个数为 xyz(mm)+quat(wxyz)+模式码。"""
    try:
        sock.sendall(VISION_TRIGGER_COMMAND.encode("utf-8"))
        raw_text = sock.recv(4096).decode("utf-8", errors="ignore").strip()
        numbers = [float(item) for item in NUMBER_PATTERN.findall(raw_text)]
        if len(numbers) < 9:
            message = f"viewer 返回数字不足 9 个，至少需要 1 + 8：{raw_text}"
            log.error(message)
            print(message)
            return None

        payload = numbers[1:]
        if len(payload) % 8 != 0:
            message = (
                f"viewer 返回格式错误：忽略第 1 个数后剩余 {len(payload)} 个，"
                f"不是 8 的整数倍，原始返回：{raw_text}"
            )
            log.error(message)
            print(message)
            return None

        poses: list[tuple[int, list[float]]] = []
        for index in range(0, len(payload), 8):
            group = payload[index:index + 8]
            pose = group[:7]
            mode_value = group[7]
            if mode_value not in (1.0, 2.0):
                point_index = index // 8 + 1
                message = (
                    f"viewer 第 {point_index} 个点模式码无效：{mode_value}，"
                    f"原始返回：{raw_text}"
                )
                log.error(message)
                print(message)
                return None
            poses.append((int(mode_value), pose))

        print(f"viewer 点个数: {len(poses)}")
        log.info(f"viewer 点个数: {len(poses)}")
        return poses
    except OSError as exc:
        message = f"viewer 获取 pose 失败：{exc}"
        log.error(message)
        print(message)
        return None


def close_vision(sock: socket.socket | None, log) -> None:
    """关闭视觉 TCP 连接。"""
    if sock is None:
        return
    sock.close()
    log.info("已关闭视觉 TCP 连接")


def matmul4(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def invert_transform4(t: list[list[float]]) -> list[list[float]]:
    """刚体 4x4 齐次矩阵求逆。"""
    rot_t = [[t[j][i] for j in range(3)] for i in range(3)]
    trans = [t[i][3] for i in range(3)]
    inv_trans = [-sum(rot_t[i][k] * trans[k] for k in range(3)) for i in range(3)]
    return [
        [rot_t[0][0], rot_t[0][1], rot_t[0][2], inv_trans[0]],
        [rot_t[1][0], rot_t[1][1], rot_t[1][2], inv_trans[1]],
        [rot_t[2][0], rot_t[2][1], rot_t[2][2], inv_trans[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rpy_to_rot(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def rot_to_rpy(rot: list[list[float]]) -> tuple[float, float, float]:
    pitch = math.atan2(-rot[2][0], math.hypot(rot[0][0], rot[1][0]))
    if abs(math.cos(pitch)) < 1e-9:
        roll = 0.0
        yaw = math.atan2(-rot[0][1], rot[1][1])
    else:
        roll = math.atan2(rot[2][1], rot[2][2])
        yaw = math.atan2(rot[1][0], rot[0][0])
    return roll, pitch, yaw


def pose_mm_deg_to_matrix(pose: Sequence[float]) -> list[list[float]]:
    x, y, z, roll_deg, pitch_deg, yaw_deg = [float(value) for value in pose[:6]]
    rot = rpy_to_rot(math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg))
    return [
        [rot[0][0], rot[0][1], rot[0][2], x],
        [rot[1][0], rot[1][1], rot[1][2], y],
        [rot[2][0], rot[2][1], rot[2][2], z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def quat_wxyz_to_rot(qw: float, qx: float, qy: float, qz: float) -> list[list[float]]:
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1e-12:
        raise ValueError("viewer 四元数长度为 0")
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return [
        [
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ],
        [
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ],
        [
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
    ]


def pose_mm_wxyz_to_matrix(pose: Sequence[float]) -> list[list[float]]:
    x, y, z, qw, qx, qy, qz = [float(value) for value in pose[:7]]
    rot = quat_wxyz_to_rot(qw, qx, qy, qz)
    return [
        [rot[0][0], rot[0][1], rot[0][2], x],
        [rot[1][0], rot[1][1], rot[1][2], y],
        [rot[2][0], rot[2][1], rot[2][2], z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def matrix_to_xyz_rpy(matrix: list[list[float]]) -> tuple[float, float, float, float, float, float]:
    rot = [row[:3] for row in matrix[:3]]
    roll, pitch, yaw = rot_to_rpy(rot)
    return (
        matrix[0][3],
        matrix[1][3],
        matrix[2][3],
        roll,
        pitch,
        yaw,
    )


def xyz_rpy_to_pose(values: Sequence[float], position_scale: float = 1.0) -> Pose:
    """xyz+rpy 转 Pose；position_scale 可用于将毫米转换成米。"""
    x, y, z, roll, pitch, yaw = [float(value) for value in values]
    return make_pose(
        x * position_scale,
        y * position_scale,
        z * position_scale,
        roll,
        pitch,
        yaw,
    )


def xyz_rpy_to_matrix(values: Sequence[float], position_scale: float = 1.0) -> list[list[float]]:
    """xyz+rpy 转 4×4 矩阵；position_scale 可用于将毫米转换成米。"""
    x, y, z, roll, pitch, yaw = [float(value) for value in values]
    rot = rpy_to_rot(roll, pitch, yaw)
    return [
        [rot[0][0], rot[0][1], rot[0][2], x * position_scale],
        [rot[1][0], rot[1][1], rot[1][2], y * position_scale],
        [rot[2][0], rot[2][1], rot[2][2], z * position_scale],
        [0.0, 0.0, 0.0, 1.0],
    ]


def pose_to_matrix(pose: Pose) -> list[list[float]]:
    """geometry_msgs/Pose 转 4×4 齐次变换矩阵，平移单位为米。"""
    q = pose.orientation
    rot = quat_wxyz_to_rot(q.w, q.x, q.y, q.z)
    p = pose.position
    return [
        [rot[0][0], rot[0][1], rot[0][2], p.x],
        [rot[1][0], rot[1][1], rot[1][2], p.y],
        [rot[2][0], rot[2][1], rot[2][2], p.z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def pose_to_xyz_rpy(pose: Pose) -> tuple[float, float, float, float, float, float]:
    """geometry_msgs/Pose 转 xyz+rpy。"""
    return matrix_to_xyz_rpy(pose_to_matrix(pose))


def transform_vision_pose(
    pose: Sequence[float],
    transform_mm: list[list[float]],
) -> tuple[float, float, float, float, float, float]:
    """viewer pose 为 xyz(mm) + quat(wxyz)；左乘指定标定矩阵后返回 xyz(mm), rpy(rad)。"""
    return matrix_to_xyz_rpy(matmul4(transform_mm, pose_mm_wxyz_to_matrix(pose)))


def read_vision_object_pose(node, log):
    """视觉识别封装：返回所有点的模式码和 xyz_rpy。"""
    vision_sock = connect_vision(log)
    if vision_sock is None:
        return None

    try:
        vision_results = read_vision_pose(vision_sock, log)
        if vision_results is None:
            return None

        # 读取实际腰部角度；FK 返回 T_SJ_r_base_link，再左乘右侧物体位姿得到 T_SJ_object。
        body_joints = node._get_joints(["body_joint2"], wait_new=True)
        if body_joints is None:
            log.error("读取实际腰部关节失败，无法转换到 SJ")
            return None
        print(f"实际腰部角度: body_joint2={body_joints['body_joint2']:.6f} rad")

        r_base_in_sj = node._get_link_pose_fk(
            "r_base_link",
            joints=body_joints,
            plan_frame="SJ",
        )
        if r_base_in_sj is None:
            log.error("计算 r_base_link → SJ 变换失败")
            return None
        l_base_in_sj = node._get_link_pose_fk(
            "l_base_link",
            joints=body_joints,
            plan_frame="SJ",
        )
        if l_base_in_sj is None:
            log.error("计算 l_base_link → SJ 变换失败")
            return None

        t_sj_r_base = pose_to_matrix(r_base_in_sj)
        t_sj_l_base = pose_to_matrix(l_base_in_sj)
        first_return_modes: list[int] = []
        all_xyz_rpy: list[dict[str, tuple[float, float, float, float, float, float]]] = []

        for point_index, (first_return_mode, pose) in enumerate(vision_results, start=1):
            print(
                f"\033[32mviewer point {point_index}: "
                f"first_return_mode = {first_return_mode}, pose = {pose}\033[0m"
            )

            try:
                right_xyz_rpy_mm = transform_vision_pose(pose, VISION_RIGHT_TRANSFORM_MM)
                left_xyz_rpy_mm = transform_vision_pose(pose, VISION_LEFT_TRANSFORM_MM)
            except ValueError as exc:
                message = f"viewer 第 {point_index} 个 pose 解析失败：{exc}"
                log.error(message)
                print(message)
                return None

            # 标定输出的位置为毫米，统一转换成米制 4x4 矩阵。
            right_matrix = xyz_rpy_to_matrix(right_xyz_rpy_mm, position_scale=0.001)
            left_matrix = xyz_rpy_to_matrix(left_xyz_rpy_mm, position_scale=0.001)
            right_sj_matrix = matmul4(t_sj_r_base, right_matrix)
            left_sj_matrix = matmul4(t_sj_l_base, left_matrix)
            xyz_rpy = {
                "right": matrix_to_xyz_rpy(right_matrix),
                "left": matrix_to_xyz_rpy(left_matrix),
                "right_sj": matrix_to_xyz_rpy(right_sj_matrix),
                "left_sj": matrix_to_xyz_rpy(left_sj_matrix),
                "sj": matrix_to_xyz_rpy(right_sj_matrix),
            }

            first_return_modes.append(first_return_mode)
            all_xyz_rpy.append(xyz_rpy)
            print(f"xyz_rpy[{point_index}] [m, rad]: {xyz_rpy}")

        return first_return_modes, all_xyz_rpy
    finally:
        close_vision(vision_sock, log)


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """固定轴 XYZ：欧拉角 → 四元数 (x, y, z, w)。"""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def joint_names_for_group(group: str) -> list[str]:
    """从 JOINT_TARGETS 读取 group 的关节名列表（顺序与 place_joints 向量一致）。"""
    if group not in JOINT_TARGETS:
        raise KeyError(f"未知 group={group}，JOINT_TARGETS 可选: {list(JOINT_TARGETS)}")
    return list(JOINT_TARGETS[group].keys())


def _side_order_from_xyz_rpy(xyz_rpy: dict) -> tuple[str, str]:
    """根据物体在 SJ 下的 z 值决定左右优先级。"""
    return ("left", "right") if xyz_rpy["sj"][2] > 0.0 else ("right", "left")


def _reachability_attempts_for_point(xyz_rpy: dict) -> list[dict[str, str]]:
    """先只用手臂验证；两只手臂都不行，再验证手臂+腰部。"""
    side_order = _side_order_from_xyz_rpy(xyz_rpy)
    attempts: list[dict[str, str]] = []
    for side in side_order:
        attempts.append({
            "side": side,
            "group": f"{side}_arm",
            "link": "l_tool" if side == "left" else "r_tool",
            "plan_frame": "l_base_link" if side == "left" else "r_base_link",
            "xyz_key": side,
        })
    for side in side_order:
        attempts.append({
            "side": side,
            "group": f"{side}_body",
            "link": "l_tool" if side == "left" else "r_tool",
            "plan_frame": "SJ",
            "xyz_key": f"{side}_sj",
        })
    return attempts


def arm_context_for_body_group(group: str) -> tuple[str, str] | None:
    """body 抓取组对应的纯臂组和纯臂规划坐标系。"""
    if group == "left_body":
        return "left_arm", "l_base_link"
    if group == "right_body":
        return "right_arm", "r_base_link"
    return None


def place_joints_for_group(group: str) -> dict[str, Sequence[float]]:
    """获取 pick_and_return 使用的放置关节配置，body 组自动补 body_joint2。"""
    place_joints = PLACE_JOINTS.get(group)
    if isinstance(place_joints, dict):
        return place_joints

    if group == "left_body":
        arm_group = "left_arm"
    elif group == "right_body":
        arm_group = "right_arm"
    else:
        raise KeyError(f"PLACE_JOINTS 中没有可用的 {group} 放置配置")

    body_joint2 = JOINT_TARGETS[group]["body_joint2"]
    arm_place = PLACE_JOINTS[arm_group]
    return {
        name: [body_joint2, *values]
        for name, values in arm_place.items()
    }

# 按视觉点顺序遍历：
#   点1 → 点2 → 点3 ...

# 每个点内按 side_value = xyz_rpy["sj"][2] 判断左右顺序：

# side_value > 0:
#   left_arm → right_arm → left_body → right_body
def validate_reachable_grasp(node, all_xyz_rpy: list[dict], speed_scale: float = 0.2):
    """按点和候选 group 顺序验证 IK 可解 + cartesian approach 可行。"""
    log = node.get_logger()
    if not all_xyz_rpy:
        log.error("[reach] 没有可验证的视觉点")
        return None

    for point_index, xyz_rpy in enumerate(all_xyz_rpy):
        attempts = _reachability_attempts_for_point(xyz_rpy)
        order_text = " → ".join(item["group"] for item in attempts)
        log.info(
            f"[reach] 点 {point_index + 1}/{len(all_xyz_rpy)}: "
            f"SJ.z={xyz_rpy['sj'][2]:.6f}, 验证顺序 {order_text}"
        )

        for attempt in attempts:
            pick_group = attempt["group"]
            pick_link = attempt["link"]
            pick_frame = attempt["plan_frame"]
            xyz_key = attempt["xyz_key"]
            if pick_group not in JOINT_TARGETS:
                log.warning(f"[reach] 跳过未知 group={pick_group}")
                continue
            if xyz_key not in xyz_rpy:
                log.warning(f"[reach] 点 {point_index + 1} 缺少 xyz_rpy[{xyz_key!r}]，跳过 {pick_group}")
                continue

            pick_joint_names = joint_names_for_group(pick_group)
            pick_target_pose = xyz_rpy_to_pose(xyz_rpy[xyz_key])
            pre_pose = pose_offset_local_z(pick_target_pose, PRE_GRASP_OFFSET)
            log.info(
                f"[reach] 验证点 {point_index + 1}: group={pick_group}, "
                f"link={pick_link}, frame={pick_frame}, xyz_key={xyz_key}"
            )
            if arm_context_for_body_group(pick_group) is not None:
                picked = node._select_feasible_grasp_pair_with_waist(
                    pick_group,
                    pick_link,
                    pick_target_pose,
                    pre_pose,
                    joint_names=pick_joint_names,
                    speed_scale=speed_scale,
                    plan_frame=pick_frame,
                )
            else:
                picked = node._select_feasible_grasp_pair(
                    pick_group,
                    pick_link,
                    pick_target_pose,
                    pre_pose,
                    joint_names=pick_joint_names,
                    speed_scale=speed_scale,
                    plan_frame=pick_frame,
                )
            if picked is None:
                log.warning(f"[reach] 点 {point_index + 1} / {pick_group}: 不可达")
                continue

            q_pre, q_target, _approach_traj = picked
            log.info(f"[reach] 点 {point_index + 1} / {pick_group}: 可达 ✓")
            # log.info(
            #     f"{GREEN}[reach] 可行点 xyz_rpy[{xyz_key}]: "
            #     f"{xyz_rpy[xyz_key]}{RESET}"
            # )
            log.info(
                "[reach] 抓取点 q_target: "
                + ", ".join(f"{n}={q_target[n]:.6f}" for n in pick_joint_names)
            )
            return {
                "point_index": point_index,
                "pick_side": attempt["side"],
                "pick_group": pick_group,
                "pick_link": pick_link,
                "pick_frame": pick_frame,
                "pick_xyz_key": xyz_key,
                "pick_joint_names": pick_joint_names,
                "pick_target_pose": pick_target_pose,
                "pick_q_pre": q_pre,
                "pick_q_target": q_target,
            }

    log.error("[reach] 所有视觉点的所有候选 group 均不可达")
    return None


def tool_side_for_link(link: str) -> str | None:
    """只根据末端工具 link 判断左右臂。"""
    link_l = link.lower()
    if link_l == "l_tool":
        return "left"
    if link_l == "r_tool":
        return "right"
    return None


def make_pose(x: float, y: float, z: float, roll=0.0, pitch=0.0, yaw=0.0) -> Pose:
    """构造 geometry_msgs/Pose。"""
    p = Pose()
    p.position.x, p.position.y, p.position.z = x, y, z
    qx, qy, qz, qw = quat_from_rpy(roll, pitch, yaw)
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = qx, qy, qz, qw
    return p


def rotate_xyz_by_quat(
    x: float,
    y: float,
    z: float,
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> tuple[float, float, float]:
    """用四元数旋转向量。"""
    return (
        (1.0 - 2.0 * (qy * qy + qz * qz)) * x
        + 2.0 * (qx * qy - qz * qw) * y
        + 2.0 * (qx * qz + qy * qw) * z,
        2.0 * (qx * qy + qz * qw) * x
        + (1.0 - 2.0 * (qx * qx + qz * qz)) * y
        + 2.0 * (qy * qz - qx * qw) * z,
        2.0 * (qx * qz - qy * qw) * x
        + 2.0 * (qy * qz + qx * qw) * y
        + (1.0 - 2.0 * (qx * qx + qy * qy)) * z,
    )


def pose_offset_local_z(pose: Pose, dz: float) -> Pose:
    """沿 pose 自身坐标系 z 轴方向平移 dz 米，得到新位姿（姿态不变）。

    用旋转矩阵第三列（= 局部 +z 在 base 系下的方向）做平移：
        new_p = p + dz * R[:, 2]
    其中 R 由 (qx, qy, qz, qw) 构造。
    """
    qx, qy, qz, qw = (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    zx = 2.0 * (qx * qz + qw * qy)
    zy = 2.0 * (qy * qz - qw * qx)
    zz = 1.0 - 2.0 * (qx * qx + qy * qy)
    out = Pose()
    out.position.x = pose.position.x + dz * zx
    out.position.y = pose.position.y + dz * zy
    out.position.z = pose.position.z + dz * zz
    out.orientation.x = qx
    out.orientation.y = qy
    out.orientation.z = qz
    out.orientation.w = qw
    return out


def make_deep_frame() -> CollisionObject:
    """
    深框 = 1 块底板 + 4 块侧墙（BOX  primitive），顶部无盖。
    FRAME_SIZE 为深框整体外尺寸，FRAME_CENTER 为外轮廓中心。
    WALL_T 只向内部收缩，外轮廓保持 L × W × H。
    """
    L, W, H = FRAME_SIZE
    t = WALL_T
    bx, by, bz = FRAME_CENTER
    roll, pitch, yaw = (math.radians(v) for v in FRAME_RPY_DEG)
    qx, qy, qz, qw = quat_from_rpy(roll, pitch, yaw)

    obj = CollisionObject()
    obj.header.frame_id = SCENE_FRAME
    obj.id = FRAME_ID
    obj.operation = CollisionObject.ADD

    def add_box(dx, dy, dz, ox, oy, oz):
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [dx, dy, dz]
        rx, ry, rz = rotate_xyz_by_quat(ox, oy, oz, qx, qy, qz, qw)
        pose = make_pose(bx + rx, by + ry, bz + rz, roll, pitch, yaw)
        obj.primitives.append(prim)
        obj.primitive_poses.append(pose)

    wall_h = H - t
    wall_z = -H / 2 + t + wall_h / 2
    add_box(L, W, t, 0, 0, -H / 2 + t / 2)                    # 底板
    add_box(t, W, wall_h, L / 2 - t / 2, 0, wall_z)           # +X 侧墙
    add_box(t, W, wall_h, -(L / 2 - t / 2), 0, wall_z)        # -X 侧墙
    add_box(L - 2 * t, t, wall_h, 0, W / 2 - t / 2, wall_z)   # +Y 侧墙
    add_box(L - 2 * t, t, wall_h, 0, -(W / 2 - t / 2), wall_z)  # -Y 侧墙
    return obj


def make_deep_frame_cutoff(scene_y: float) -> CollisionObject:
    """在深框内生成一张薄隔离面，顶面位于 SCENE_FRAME 的 scene_y。"""
    L, W, _ = FRAME_SIZE
    t = WALL_T
    bx, _, bz = FRAME_CENTER
    roll, pitch, yaw = (math.radians(v) for v in FRAME_RPY_DEG)
    qx, qy, qz, qw = quat_from_rpy(roll, pitch, yaw)

    obj = CollisionObject()
    obj.header.frame_id = SCENE_FRAME
    obj.id = FRAME_CUTOFF_ID
    obj.operation = CollisionObject.ADD

    prim = SolidPrimitive()
    prim.type = SolidPrimitive.BOX
    prim.dimensions = [
        max(0.0, L - 2.0 * t),
        max(0.0, W - 2.0 * t),
        FRAME_CUTOFF_THICKNESS,
    ]
    ox, oy, oz = rotate_xyz_by_quat(
        0.0, 0.0, -FRAME_CUTOFF_THICKNESS / 2.0, qx, qy, qz, qw
    )
    obj.primitives.append(prim)
    obj.primitive_poses.append(make_pose(bx + ox, scene_y + oy, bz + oz, roll, pitch, yaw))
    return obj


def make_grasp_object_collision(pose: Pose) -> CollisionObject:
    """创建与 RViz 圆柱 Marker 同尺寸、同位姿的 MoveIt 碰撞体。"""
    obj = CollisionObject()
    obj.header.frame_id = SCENE_FRAME
    obj.id = GRASP_OBJECT_COLLISION_ID
    obj.operation = CollisionObject.ADD

    prim = SolidPrimitive()
    prim.type = SolidPrimitive.CYLINDER
    prim.dimensions = [CYLINDER_HEIGHT, CYLINDER_DIAMETER / 2.0]
    obj.primitives.append(prim)
    obj.primitive_poses.append(copy.deepcopy(pose))
    return obj


def cylinder_half_extent_y(pose: Pose) -> float:
    """圆柱按当前姿态投影到 Y 轴后的半尺寸，用于让隔板与圆柱刚好相切。"""
    q = pose.orientation
    _, axis_y, _ = rotate_xyz_by_quat(0.0, 0.0, 1.0, q.x, q.y, q.z, q.w)
    axis_y = max(-1.0, min(1.0, axis_y))
    radius = CYLINDER_DIAMETER / 2.0
    half_height = CYLINDER_HEIGHT / 2.0
    radial_projection = math.sqrt(max(0.0, 1.0 - axis_y * axis_y))
    return abs(axis_y) * half_height + radial_projection * radius


def pose_from_dict(ep: dict) -> Pose:
    """EE_POSE2 风格字典 → geometry_msgs/Pose（位置 + roll/pitch/yaw）。"""
    return make_pose(
        ep["x"], ep["y"], ep["z"],
        ep.get("roll", 0.0), ep.get("pitch", 0.0), ep.get("yaw", 0.0),
    )


def make_cylinder_marker(
    pose: Pose,
    marker_id: int,
    frame_id: str = PLAN_FRAME,
    diameter: float = CYLINDER_DIAMETER,
    height: float = CYLINDER_HEIGHT,
    color: ColorRGBA | None = None,
    ns: str = CYLINDER_MARKER_NS,
    action: int = Marker.ADD,
) -> Marker:
    """构造 RViz 圆柱 Marker（仅可视化，不参与碰撞检测）。

    圆柱中心在 pose 原点，轴线沿 pose 局部 z；scale.x/y=直径，scale.z=高度。
    """
    m = Marker()
    m.header.frame_id = frame_id
    m.ns = ns
    m.id = marker_id
    m.type = Marker.CYLINDER
    m.action = action
    m.pose = pose
    m.scale.x = diameter
    m.scale.y = diameter
    m.scale.z = height
    m.color = color or CYLINDER_COLOR
    return m


def make_z_axis_marker(
    pose: Pose,
    marker_id: int,
    frame_id: str = PLAN_FRAME,
    length: float = Z_AXIS_LENGTH,
    color: ColorRGBA | None = None,
    ns: str = CYLINDER_MARKER_NS,
) -> Marker:
    """构造从 pose 原点指向其局部 +z 方向的红色箭头。"""
    q = pose.orientation
    dx, dy, dz = rotate_xyz_by_quat(0.0, 0.0, length, q.x, q.y, q.z, q.w)

    start = Point(x=pose.position.x, y=pose.position.y, z=pose.position.z)
    end = Point(x=start.x + dx, y=start.y + dy, z=start.z + dz)

    m = Marker()
    m.header.frame_id = frame_id
    m.ns = ns
    m.id = marker_id
    m.type = Marker.ARROW
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    m.points = [start, end]
    m.scale.x = 0.01  # 箭杆直径
    m.scale.y = 0.015  # 箭头直径
    m.scale.z = 0.012  # 箭头长度
    m.color = color or Z_AXIS_COLOR
    return m


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _pose_distance(a: Pose, b: Pose) -> float:
    dx = a.position.x - b.position.x
    dy = a.position.y - b.position.y
    dz = a.position.z - b.position.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _pose_position_delta(actual: Pose, target: Pose) -> tuple[float, float, float, float]:
    dx = actual.position.x - target.position.x
    dy = actual.position.y - target.position.y
    dz = actual.position.z - target.position.z
    return dx, dy, dz, math.sqrt(dx * dx + dy * dy + dz * dz)


def _pose_orientation_error_rad(actual: Pose, target: Pose) -> float:
    aq = actual.orientation
    tq = target.orientation
    actual_norm = math.sqrt(aq.x * aq.x + aq.y * aq.y + aq.z * aq.z + aq.w * aq.w)
    target_norm = math.sqrt(tq.x * tq.x + tq.y * tq.y + tq.z * tq.z + tq.w * tq.w)
    if actual_norm <= 0.0 or target_norm <= 0.0:
        return float("nan")
    dot = (
        aq.x * tq.x + aq.y * tq.y + aq.z * tq.z + aq.w * tq.w
    ) / (actual_norm * target_norm)
    dot = max(-1.0, min(1.0, abs(dot)))
    return 2.0 * math.acos(dot)


def _trajectory_joint_stats(traj: RobotTrajectory) -> tuple[float, float]:
    """返回 (关节空间累计路程, 最大单步关节变化)。"""
    pts = traj.joint_trajectory.points
    if len(pts) < 2:
        return 0.0, 0.0

    total = 0.0
    max_step = 0.0
    for prev, cur in zip(pts, pts[1:]):
        if not prev.positions or not cur.positions:
            continue
        deltas = [abs(b - a) for a, b in zip(prev.positions, cur.positions)]
        if not deltas:
            continue
        total += sum(deltas)
        max_step = max(max_step, max(deltas))
    return total, max_step


def _trajectory_joint_jump_reason(traj: RobotTrajectory) -> str | None:
    """检测笛卡尔插补中的关节跳变；有跳变时返回原因，否则返回 None。"""
    pts = traj.joint_trajectory.points
    if len(pts) < 3:
        return None

    step_norms = []
    max_joint_step = 0.0
    for prev, cur in zip(pts, pts[1:]):
        if not prev.positions or not cur.positions:
            continue
        deltas = [abs(b - a) for a, b in zip(prev.positions, cur.positions)]
        if not deltas:
            continue
        max_joint_step = max(max_joint_step, max(deltas))
        step_norms.append(math.sqrt(sum(d * d for d in deltas)))

    if not step_norms:
        return None

    if max_joint_step > CART_REVOLUTE_JUMP_THRESHOLD:
        return (
            f"单关节单步 {max_joint_step:.3f} rad "
            f"> {CART_REVOLUTE_JUMP_THRESHOLD:.3f} rad"
        )

    avg_step = sum(step_norms) / len(step_norms)
    max_step = max(step_norms)
    if avg_step > 1e-9 and max_step > avg_step * CART_JUMP_STEP_FACTOR:
        return (
            f"关节空间单步 {max_step:.3f} "
            f"> 平均 {avg_step:.3f} * {CART_JUMP_STEP_FACTOR:.1f}"
        )

    return None


def _moveit_error_name(val: int) -> str:
    # 只覆盖常见错误，其他按数值输出
    mapping = {
        MoveItErrorCodes.SUCCESS: "SUCCESS",
        MoveItErrorCodes.FAILURE: "FAILURE",
        MoveItErrorCodes.PLANNING_FAILED: "PLANNING_FAILED",
        MoveItErrorCodes.INVALID_MOTION_PLAN: "INVALID_MOTION_PLAN",
        MoveItErrorCodes.MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE: "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE",
        MoveItErrorCodes.CONTROL_FAILED: "CONTROL_FAILED",
        MoveItErrorCodes.UNABLE_TO_AQUIRE_SENSOR_DATA: "UNABLE_TO_AQUIRE_SENSOR_DATA",
        MoveItErrorCodes.TIMED_OUT: "TIMED_OUT",
        MoveItErrorCodes.PREEMPTED: "PREEMPTED",
        MoveItErrorCodes.START_STATE_IN_COLLISION: "START_STATE_IN_COLLISION",
        MoveItErrorCodes.START_STATE_VIOLATES_PATH_CONSTRAINTS: "START_STATE_VIOLATES_PATH_CONSTRAINTS",
        MoveItErrorCodes.GOAL_IN_COLLISION: "GOAL_IN_COLLISION",
        MoveItErrorCodes.GOAL_VIOLATES_PATH_CONSTRAINTS: "GOAL_VIOLATES_PATH_CONSTRAINTS",
        MoveItErrorCodes.GOAL_CONSTRAINTS_VIOLATED: "GOAL_CONSTRAINTS_VIOLATED",
        MoveItErrorCodes.INVALID_GROUP_NAME: "INVALID_GROUP_NAME",
        MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS: "INVALID_GOAL_CONSTRAINTS",
        MoveItErrorCodes.INVALID_ROBOT_STATE: "INVALID_ROBOT_STATE",
        MoveItErrorCodes.INVALID_LINK_NAME: "INVALID_LINK_NAME",
        MoveItErrorCodes.INVALID_OBJECT_NAME: "INVALID_OBJECT_NAME",
        MoveItErrorCodes.FRAME_TRANSFORM_FAILURE: "FRAME_TRANSFORM_FAILURE",
        MoveItErrorCodes.COLLISION_CHECKING_UNAVAILABLE: "COLLISION_CHECKING_UNAVAILABLE",
        MoveItErrorCodes.ROBOT_STATE_STALE: "ROBOT_STATE_STALE",
        MoveItErrorCodes.SENSOR_INFO_STALE: "SENSOR_INFO_STALE",
        MoveItErrorCodes.NO_IK_SOLUTION: "NO_IK_SOLUTION",
    }
    return mapping.get(int(val), f"UNKNOWN({int(val)})")


def make_joint_constraints(group: str, joints: dict[str, float]) -> Constraints:
    """关节目标（dict）→ MoveIt goal_constraints（每个关节一个 JointConstraint）。"""
    c = Constraints()
    c.name = f"{group}_joint_goal"
    for name, pos in joints.items():
        jc = JointConstraint()
        jc.joint_name = name
        jc.position = pos
        jc.tolerance_above = jc.tolerance_below = 1e-5
        jc.weight = 1.0
        c.joint_constraints.append(jc)
    return c


def make_joint_constraints_from_vector(
    group: str, joint_names: Sequence[str], joint_values: Sequence[float]
) -> Constraints:
    """关节目标（vector）→ MoveIt goal_constraints。"""
    if len(joint_names) != len(joint_values):
        raise ValueError(f"joint_names 与 joint_values 长度不一致: {len(joint_names)} vs {len(joint_values)}")
    return make_joint_constraints(group, dict(zip(joint_names, joint_values)))


def make_pose_constraints(link: str, pose: Pose, frame_id: str = PLAN_FRAME) -> Constraints:
    """
    末端位姿目标 → PositionConstraint（位置球）+ OrientationConstraint。
    约束在 frame_id 下表达，link_name 为末端连杆名。
    """
    c = Constraints()
    c.name = "pose_goal"

    sphere = SolidPrimitive()
    sphere.type = SolidPrimitive.SPHERE
    sphere.dimensions = [_POS_TOL]
    region = BoundingVolume()
    region.primitives.append(sphere)
    pos_only = Pose()
    pos_only.position = pose.position
    pos_only.orientation.w = 1.0
    region.primitive_poses.append(pos_only)

    pc = PositionConstraint()
    pc.header.frame_id = frame_id
    pc.link_name = link
    pc.constraint_region = region
    pc.weight = 1.0
    c.position_constraints.append(pc)

    oc = OrientationConstraint()
    oc.header.frame_id = frame_id
    oc.link_name = link
    oc.orientation = pose.orientation
    oc.absolute_x_axis_tolerance = _ORI_TOL
    oc.absolute_y_axis_tolerance = _ORI_TOL
    oc.absolute_z_axis_tolerance = _ORI_TOL
    oc.weight = 1.0
    c.orientation_constraints.append(oc)
    return c


# =============================================================================
# ROS 节点：场景管理 + move_action 调用
# =============================================================================


class G01Demo(Node):
    """封装 apply_planning_scene 与 move_action，对外提供少量高层接口。"""

    def __init__(self):
        super().__init__("g01_demo")
        self._scene_cli = self.create_client(ApplyPlanningScene, SVC_APPLY_SCENE)
        self._cart_cli = self.create_client(GetCartesianPath, SVC_CARTESIAN_PATH)
        self._ik_cli = self.create_client(GetPositionIK, SVC_COMPUTE_IK)
        self._fk_cli = self.create_client(GetPositionFK, SVC_COMPUTE_FK)
        self._left_tool_cli = self.create_client(SetToolPower, LEFT_TOOL_COMMAND_SERVICE)
        self._right_tool_cli = self.create_client(SetToolPower, RIGHT_TOOL_COMMAND_SERVICE)
        self._move_cli = ActionClient(self, MoveGroup, ACT_MOVE_GROUP)
        self._exec_cli = ActionClient(self, ExecuteTrajectory, ACT_EXEC_TRAJ)
        # 缓存最新 joint_states，供规划起点使用
        self._joints: dict[str, float] = {}
        self._js_count = 0
        self._latest_grasp_cmd: dict | None = None
        self._start_grasp_cmd: dict | None = None
        self.last_pick_failure_reason: str | None = None
        self.create_subscription(JointState, "/g01/joint_states", self._on_js, 10)
        self.create_subscription(String, GRASP_CMD_TOPIC, self._on_grasp_cmd, 10)
        self._grasp_result_pub = self.create_publisher(String, GRASP_CMD_RESULT_TOPIC, 10)
        self._cylinder_marker_pub = self.create_publisher(Marker, CYLINDER_MARKER_TOPIC, 1)
        self._cylinder_marker_ids: dict[str, int] = {}
        self._next_cylinder_marker_id = 0

    def _on_grasp_cmd(self, msg: String):
        """解析 /grasp_cmd 的 JSON 字符串，cmd_type=1 表示启动抓取。"""
        log = self.get_logger()
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            log.error(f"{GRASP_CMD_TOPIC} JSON 解析失败: {exc}; data={msg.data!r}")
            return

        if not isinstance(data, dict):
            log.error(f"{GRASP_CMD_TOPIC} JSON 必须是 object: {msg.data!r}")
            return

        try:
            cmd_type = int(data.get("cmd_type", 0))
            grasp_number = int(data.get("grasp_number", 0))
            release_number = int(data.get("release_number", 0))
        except (TypeError, ValueError) as exc:
            log.error(f"{GRASP_CMD_TOPIC} 字段类型错误: {exc}; data={msg.data!r}")
            return

        parsed = {
            "cmd_type": cmd_type,
            "grasp_number": grasp_number,
            "release_number": release_number,
        }
        self._latest_grasp_cmd = parsed
        log.info(
            f"收到抓取命令: cmd_type={cmd_type}, "
            f"grasp_number={grasp_number}, release_number={release_number}"
        )
        if cmd_type == 1:
            self._start_grasp_cmd = parsed

    def wait_for_grasp_start(self) -> dict | None:
        """等待 /grasp_cmd 收到 cmd_type=1。"""
        log = self.get_logger()
        log.info(f"等待 {GRASP_CMD_TOPIC} JSON 命令 cmd_type=1 后开始视觉识别 …")
        while rclpy.ok():
            if self._start_grasp_cmd is not None:
                return self._start_grasp_cmd
            rclpy.spin_once(self, timeout_sec=0.1)
        return None

    def publish_grasp_cmd_result(self, result: bool, success_grasp_number: int):
        """发布 /grasp_cmd_result 的 JSON 字符串。"""
        data = {
            "result": bool(result),
            "success_grasp_number": int(success_grasp_number),
        }
        msg = String()
        msg.data = json.dumps(data, ensure_ascii=False)
        self._grasp_result_pub.publish(msg)
        self.get_logger().info(f"发布抓取结果 {GRASP_CMD_RESULT_TOPIC}: {msg.data}")

    def move_body_joint2(self, angle_rad: float, speed_scale: float = 0.5) -> bool:
        """读取 body 组当前位置，只改变 body_joint2 后做关节空间规划执行。"""
        log = self.get_logger()
        if WAIST_BODY_GROUP not in JOINT_TARGETS:
            log.error(f"未知腰部规划组 {WAIST_BODY_GROUP}")
            return False

        joint_names = joint_names_for_group(WAIST_BODY_GROUP)
        if WAIST_BODY_JOINT not in joint_names:
            log.error(f"{WAIST_BODY_GROUP} 组不包含 {WAIST_BODY_JOINT}")
            return False

        current = self._get_joints(joint_names, wait_new=True)
        if current is None:
            log.error(f"[waist] 读取 {WAIST_BODY_GROUP} 当前关节失败")
            return False

        target = dict(current)
        target[WAIST_BODY_JOINT] = float(angle_rad)
        log.info(
            f"[waist] {WAIST_BODY_GROUP} 关节空间规划: "
            f"{WAIST_BODY_JOINT} {current[WAIST_BODY_JOINT]:.6f} -> {angle_rad:.6f} rad "
            f"({math.degrees(angle_rad):.2f} deg), speed_scale={_clamp01(speed_scale):.2f}"
        )
        ok, used_ms, _ = self.move(
            WAIST_BODY_GROUP,
            [make_joint_constraints(WAIST_BODY_GROUP, target)],
            start=current,
            plan_only=False,
            speed_scale=speed_scale,
        )
        log.info(f"[waist] move body_joint2: {used_ms:.3f} ms ({'success' if ok else 'failed'})")
        return ok

    def _on_js(self, msg: JointState):
        """每次收到 joint_states 更新缓存并递增计数（用于检测「新帧」）。"""
        self._js_count += 1
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self._joints[name] = msg.position[i]

    def _spin_until(self, future, timeout: float) -> bool:
        """阻塞直到 future 完成或超时。"""
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        return future.done() and future.result() is not None

    def _get_joints(self, names: list[str], wait_new=False, timeout=10.0) -> dict[str, float] | None:
        """
        读取指定关节的当前位置。
        wait_new=True：先等到比调用前更新的 joint_states（关节运动后用）。
        """
        seq0 = self._js_count if wait_new else -1
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if wait_new and self._js_count <= seq0:
                rclpy.spin_once(self, timeout_sec=0.1)
                continue
            missing = [n for n in names if n not in self._joints]
            if not missing:
                return {n: self._joints[n] for n in names}
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().error(f"读取关节超时，缺失: {[n for n in names if n not in self._joints]}")
        return None

    def _get_link_pose_fk(
        self,
        link: str,
        joints: dict[str, float] | None = None,
        joint_names: Sequence[str] | None = None,
        plan_frame: str = PLAN_FRAME,
    ) -> Pose | None:
        """用 /compute_fk 根据关节角求 link 在 plan_frame 下的位姿。"""
        log = self.get_logger()
        if not self._fk_cli.wait_for_service(timeout_sec=5.0):
            log.error(f"服务 {SVC_COMPUTE_FK} 不可用")
            return None

        if joints is None:
            names = list(joint_names) if joint_names is not None else POSE_START_JOINTS
            joints = self._get_joints(names, wait_new=True)
        if joints is None:
            return None

        req = GetPositionFK.Request()
        req.header.frame_id = plan_frame
        req.fk_link_names = [link]
        req.robot_state = RobotState()
        req.robot_state.joint_state.name = list(joints.keys())
        req.robot_state.joint_state.position = list(joints.values())
        req.robot_state.is_diff = True

        fut = self._fk_cli.call_async(req)
        if not self._spin_until(fut, 5.0):
            log.error("compute_fk 超时")
            return None

        res = fut.result()
        if res.error_code.val != MoveItErrorCodes.SUCCESS or not res.pose_stamped:
            log.error(
                f"compute_fk 失败: error_code={res.error_code.val}"
                f"({_moveit_error_name(res.error_code.val)})"
            )
            return None
        return res.pose_stamped[0].pose

    def _log_actual_fk_error(
        self,
        label: str,
        link: str,
        target_pose: Pose,
        plan_frame: str,
        joint_names: Sequence[str],
    ) -> None:
        """用实际 joint_states 做 FK，并打印实际末端与目标位姿的误差。"""
        log = self.get_logger()
        joints = self._get_joints(list(joint_names), wait_new=True, timeout=3.0)
        if joints is None:
            log.error(f"{label} 读取实际关节失败，无法计算 FK 误差")
            return

        actual_pose = self._get_link_pose_fk(
            link,
            joints=joints,
            plan_frame=plan_frame,
        )
        if actual_pose is None:
            log.error(f"{label} FK 计算失败，无法比较实际位姿")
            return

        dx, dy, dz, pos_err = _pose_position_delta(actual_pose, target_pose)
        ori_err = _pose_orientation_error_rad(actual_pose, target_pose)
        ap = actual_pose.position
        tp = target_pose.position
        ao = actual_pose.orientation
        to = target_pose.orientation
        log.info(
            f"{label} 实际 FK {link} @ {plan_frame}: "
            f"actual=({ap.x:.6f}, {ap.y:.6f}, {ap.z:.6f}), "
            f"target=({tp.x:.6f}, {tp.y:.6f}, {tp.z:.6f}), "
            f"actual_quat=({ao.x:.6f}, {ao.y:.6f}, {ao.z:.6f}, {ao.w:.6f}), "
            f"target_quat=({to.x:.6f}, {to.y:.6f}, {to.z:.6f}, {to.w:.6f})"
        )
        log.info(
            f"{GREEN}{label} 位姿误差: "
            f"dxyz=({dx * 1000.0:.2f}, {dy * 1000.0:.2f}, {dz * 1000.0:.2f}) mm, "
            f"|pos|={pos_err * 1000.0:.2f} mm, "
            f"orientation={math.degrees(ori_err):.3f} deg{RESET}"
        )

    def _pose_in_frame(
        self,
        pose: Pose | dict,
        source_frame: str,
        target_frame: str,
        joint_names: Sequence[str] | None = None,
    ) -> Pose | None:
        """把完整 pose（位置和姿态）从 source_frame 转到 target_frame。"""
        p = pose if isinstance(pose, Pose) else pose_from_dict(pose)
        if source_frame == target_frame:
            return copy.deepcopy(p)

        joints = None
        if joint_names is not None:
            joints = self._get_joints(list(joint_names), timeout=2.0)
            if joints is None:
                return None

        source_pose = self._get_link_pose_fk(
            source_frame,
            joints=joints,
            joint_names=joint_names,
            plan_frame=target_frame,
        )
        if source_pose is None:
            return None

        transformed = matmul4(
            pose_to_matrix(source_pose),
            pose_to_matrix(p),
        )
        return xyz_rpy_to_pose(matrix_to_xyz_rpy(transformed))

    def set_tool_power(self, side: str, status: int, timeout: float = 10.0) -> bool:
        """调用左右工具电源服务，要求返回 res=0。"""
        log = self.get_logger()
        if side == "left":
            cli = self._left_tool_cli
            service_name = LEFT_TOOL_COMMAND_SERVICE
        elif side == "right":
            cli = self._right_tool_cli
            service_name = RIGHT_TOOL_COMMAND_SERVICE
        else:
            log.error(f"未知工具电源 side={side}")
            return False

        if not cli.wait_for_service(timeout_sec=timeout):
            log.error(f"服务 {service_name} 不可用")
            return False

        req = SetToolPower.Request()
        req.status = int(status)
        fut = cli.call_async(req)
        if not self._spin_until(fut, timeout):
            log.error(f"{service_name} SetToolPower({status}) 超时")
            return False

        res = fut.result().res
        if res != 0:
            log.error(f"{service_name} SetToolPower({status}) 失败：res={res}")
            return False
        log.info(f"{service_name} SetToolPower({status}) 成功")
        return True

    def _apply_scene(self, objects: list[CollisionObject], colors: list[ObjectColor] | None = None) -> bool:
        """向 move_group 提交规划场景 diff（添加/删除障碍物）。"""
        if not self._scene_cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(f"服务 {SVC_APPLY_SCENE} 不可用")
            return False
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        scene.world.collision_objects.extend(objects)
        if colors:
            scene.object_colors.extend(colors)
        req = ApplyPlanningScene.Request(scene=scene)
        fut = self._scene_cli.call_async(req)
        if not self._spin_until(fut, 10.0) or not fut.result().success:
            self.get_logger().error("apply_planning_scene 失败")
            return False
        return True

    def add_frame(self) -> bool:
        """添加深框并设置 RViz 显示颜色。"""
        color = ObjectColor(id=FRAME_ID, color=FRAME_COLOR)
        return self._apply_scene([make_deep_frame()], [color])

    def remove_frame(self) -> bool:
        """从场景中删除深框。"""
        rm = CollisionObject(id=FRAME_ID, operation=CollisionObject.REMOVE)
        return self._apply_scene([rm])

    def add_frame_cutoff_for_pose(
        self,
        pose: Pose | dict,
        source_frame: str = PLAN_FRAME,
        target_frame: str = SCENE_FRAME,
        joint_names: Sequence[str] | None = None,
    ) -> bool:
        """添加与目标圆柱相切的隔板，并把目标圆柱作为临时碰撞体加入场景。"""
        scene_pose = self._pose_in_frame(
            pose,
            source_frame=source_frame,
            target_frame=target_frame,
            joint_names=joint_names,
        )
        if scene_pose is None:
            self.get_logger().error(f"无法把目标位姿从 {source_frame} 转到 {target_frame}")
            return False

        half_extent_y = cylinder_half_extent_y(scene_pose)
        scene_y = scene_pose.position.y - half_extent_y
        colors = [
            ObjectColor(id=FRAME_CUTOFF_ID, color=FRAME_CUTOFF_COLOR),
            ObjectColor(id=GRASP_OBJECT_COLLISION_ID, color=GRASP_OBJECT_COLLISION_COLOR),
        ]
        self.get_logger().info(
            f"添加碰撞体「{FRAME_CUTOFF_ID}」+「{GRASP_OBJECT_COLLISION_ID}」到 {target_frame}: "
            f"物体中心 y={scene_pose.position.y:.3f} m, Y 半尺寸={half_extent_y:.3f} m, "
            f"隔板表面 y={scene_y:.3f} m"
        )
        return self._apply_scene(
            [make_deep_frame_cutoff(scene_y), make_grasp_object_collision(scene_pose)],
            colors,
        )

    def add_frame_cutoff_only_for_pose(
        self,
        pose: Pose | dict,
        source_frame: str = PLAN_FRAME,
        target_frame: str = SCENE_FRAME,
        joint_names: Sequence[str] | None = None,
    ) -> bool:
        """只添加与目标圆柱相切的隔板，不添加临时抓取物体。"""
        scene_pose = self._pose_in_frame(
            pose,
            source_frame=source_frame,
            target_frame=target_frame,
            joint_names=joint_names,
        )
        if scene_pose is None:
            self.get_logger().error(f"无法把目标位姿从 {source_frame} 转到 {target_frame}")
            return False

        half_extent_y = cylinder_half_extent_y(scene_pose)
        scene_y = scene_pose.position.y - half_extent_y
        self.get_logger().info(
            f"添加碰撞体「{FRAME_CUTOFF_ID}」到 {target_frame}: "
            f"物体中心 y={scene_pose.position.y:.3f} m, Y 半尺寸={half_extent_y:.3f} m, "
            f"隔板表面 y={scene_y:.3f} m"
        )
        return self._apply_scene(
            [make_deep_frame_cutoff(scene_y)],
            [ObjectColor(id=FRAME_CUTOFF_ID, color=FRAME_CUTOFF_COLOR)],
        )

    def remove_frame_cutoff(self) -> bool:
        """从场景中同时删除深框隔离面和临时抓取物体。"""
        removals = [
            CollisionObject(id=FRAME_CUTOFF_ID, operation=CollisionObject.REMOVE),
            CollisionObject(id=GRASP_OBJECT_COLLISION_ID, operation=CollisionObject.REMOVE),
        ]
        return self._apply_scene(removals)

    def _cylinder_marker_numeric_id(self, object_id: str) -> int:
        if object_id not in self._cylinder_marker_ids:
            self._cylinder_marker_ids[object_id] = self._next_cylinder_marker_id
            self._next_cylinder_marker_id += 1
        return self._cylinder_marker_ids[object_id]

    def show_cylinder_at_pose(
        self,
        pose: Pose | dict,
        object_id: str = CYLINDER_MARKER_ID,
        frame_id: str = PLAN_FRAME,
        diameter: float = CYLINDER_DIAMETER,
        height: float = CYLINDER_HEIGHT,
        color: ColorRGBA | None = None,
    ) -> bool:
        """在 RViz 中半透明显示圆柱（仅 Marker，不参与碰撞检测）。

        pose 可为 geometry_msgs/Pose，或 EE_POSE2 风格 dict。
        需在 RViz 添加 Marker 显示，话题订阅 /<node_name>/g01_pose_cylinder。
        """
        p = pose if isinstance(pose, Pose) else pose_from_dict(pose)
        mid = self._cylinder_marker_numeric_id(object_id)
        m = make_cylinder_marker(
            p, mid, frame_id=frame_id, diameter=diameter, height=height, color=color
        )
        m.header.stamp = self.get_clock().now().to_msg()
        self._cylinder_marker_pub.publish(m)
        self.get_logger().info(
            f"已发布圆柱 Marker（仅显示）id={object_id} topic={CYLINDER_MARKER_TOPIC} "
            f"@ {frame_id} pos=({p.position.x:.3f}, {p.position.y:.3f}, {p.position.z:.3f}), "
            f"Ø{diameter * 100:.0f}cm × H{height * 100:.0f}cm"
        )
        return True

    def show_z_axis_at_pose(
        self,
        pose: Pose | dict,
        object_id: str = Z_AXIS_MARKER_ID,
        frame_id: str = PLAN_FRAME,
        length: float = Z_AXIS_LENGTH,
    ) -> bool:
        """在 RViz 中用红色箭头显示 pose 的局部 +z 轴方向。"""
        p = pose if isinstance(pose, Pose) else pose_from_dict(pose)
        mid = self._cylinder_marker_numeric_id(object_id)
        marker = make_z_axis_marker(p, mid, frame_id=frame_id, length=length)
        marker.header.stamp = self.get_clock().now().to_msg()
        self._cylinder_marker_pub.publish(marker)
        self.get_logger().info(
            f"已发布红色 +z 轴 Marker id={object_id} topic={CYLINDER_MARKER_TOPIC} "
            f"@ {frame_id} length={length:.3f} m"
        )
        return True

    def remove_cylinder_at_pose(
        self,
        object_id: str = CYLINDER_MARKER_ID,
        frame_id: str = PLAN_FRAME,
    ) -> bool:
        """删除 show_cylinder_at_pose 发布的 RViz 圆柱 Marker。"""
        if object_id not in self._cylinder_marker_ids:
            return True
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = CYLINDER_MARKER_NS
        m.id = self._cylinder_marker_ids[object_id]
        m.action = Marker.DELETE
        self._cylinder_marker_pub.publish(m)
        del self._cylinder_marker_ids[object_id]
        return True

    def _move_once(
        self,
        group: str,
        goal_constraints: list[Constraints],
        start: dict[str, float] | None,
        plan_only: bool,
        speed_scale: float | None,
        num_attempts: int | None = None,
    ) -> tuple[bool, float, RobotTrajectory | None, int | None]:
        """单次 move_action 调用。返回 (ok, 耗时 ms, trajectory, error_code)。"""
        log = self.get_logger()
        t0 = time.monotonic()
        elapsed_ms = lambda: (time.monotonic() - t0) * 1000.0
        planning_attempts = NUM_ATTEMPTS if num_attempts is None else max(1, int(num_attempts))

        g = MoveGroup.Goal()
        g.request.group_name = group
        g.request.planner_id = PLANNER_ID
        g.request.num_planning_attempts = planning_attempts
        g.request.allowed_planning_time = PLAN_TIME_SEC
        g.request.goal_constraints = goal_constraints
        if speed_scale is not None:
            s = _clamp01(speed_scale)
            g.request.max_velocity_scaling_factor = s
            g.request.max_acceleration_scaling_factor = s
        g.request.start_state.is_diff = True
        if start:
            g.request.start_state.joint_state.name = list(start.keys())
            g.request.start_state.joint_state.position = list(start.values())
        g.planning_options.plan_only = plan_only

        send_fut = self._move_cli.send_goal_async(g)
        if not self._spin_until(send_fut, 15.0) or not send_fut.result().accepted:
            log.error("move_action 目标被拒绝或超时")
            return False, elapsed_ms(), None, None

        res_fut = send_fut.result().get_result_async()
        if not self._spin_until(res_fut, PLAN_TIME_SEC + 30.0):
            log.error("move_action 结果超时")
            return False, elapsed_ms(), None, MoveItErrorCodes.TIMED_OUT

        ar = res_fut.result()
        code_val = ar.result.error_code.val if ar.result else None
        if ar.status != GoalStatus.STATUS_SUCCEEDED:
            log.error(
                f"move_action 状态失败: {ar.status} (GoalStatus)，"
                f" MoveItErrorCodes={code_val}({_moveit_error_name(code_val) if code_val is not None else 'None'})"
            )
            return False, elapsed_ms(), None, code_val
        if code_val != MoveItErrorCodes.SUCCESS:
            log.error(f"MoveIt 错误码: {code_val} ({_moveit_error_name(code_val)})")
            return False, elapsed_ms(), None, code_val

        traj = None
        if ar.result and ar.result.planned_trajectory.joint_trajectory.points:
            traj = ar.result.planned_trajectory
        return True, elapsed_ms(), traj, MoveItErrorCodes.SUCCESS

    @staticmethod
    def _move_error_retryable(code: int | None) -> bool:
        """是否值得因 OMPL 随机性再试（起点/终点本身非法的不重试）。"""
        if code is None:
            return True
        non_retryable = {
            MoveItErrorCodes.START_STATE_IN_COLLISION,
            MoveItErrorCodes.GOAL_IN_COLLISION,
            MoveItErrorCodes.START_STATE_VIOLATES_PATH_CONSTRAINTS,
            MoveItErrorCodes.GOAL_VIOLATES_PATH_CONSTRAINTS,
            MoveItErrorCodes.INVALID_GROUP_NAME,
            MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS,
            MoveItErrorCodes.NO_IK_SOLUTION,
        }
        return code not in non_retryable

    def move(
        self,
        group: str,
        goal_constraints: list[Constraints],
        joint_names: list[str] | None = None,
        start: dict[str, float] | None = None,
        plan_only: bool = False,
        speed_scale: float | None = None,
        max_retries: int | None = None,
        num_attempts: int | None = None,
    ) -> tuple[bool, float, RobotTrajectory | None]:
        """
        调用 move_action：规划（plan_only=True）或规划并执行（False）。

        失败时自动重试最多 MOVE_MAX_RETRIES 次（应对 INVALID_MOTION_PLAN 等
        postprocessing 漏检/重采样碰撞；OMPL 每次随机采样可能换一条路）。

        返回 (是否成功, 墙钟耗时 [ms]（含所有尝试）, planned_trajectory)；
        失败或无轨迹时第三项为 None。
        joint_names + 未传 start：从 joint_states 读当前位置作为起点。
        start：显式指定起点（位姿规划在关节运动后用）。
        """
        log = self.get_logger()
        retries = MOVE_MAX_RETRIES if max_retries is None else max(1, max_retries)
        t0 = time.monotonic()
        no_traj = lambda ok: (ok, (time.monotonic() - t0) * 1000.0, None)

        if not self._move_cli.wait_for_server(timeout_sec=10.0):
            log.error(f"动作 {ACT_MOVE_GROUP} 不可用")
            return no_traj(False)

        if start is None and joint_names:
            start = self._get_joints(joint_names)
            if start is None:
                return no_traj(False)

        last_code: int | None = None
        for attempt in range(1, retries + 1):
            ok, _, traj, code = self._move_once(
                group, goal_constraints, start, plan_only, speed_scale, num_attempts
            )
            last_code = code
            if ok:
                if attempt > 1:
                    log.info(f"[{group}] move_action 第 {attempt}/{retries} 次成功")
                return True, (time.monotonic() - t0) * 1000.0, traj
            if attempt < retries and self._move_error_retryable(code):
                log.warning(
                    f"[{group}] move_action 第 {attempt}/{retries} 次失败 "
                    f"({_moveit_error_name(code) if code is not None else 'unknown'})，重试 …"
                )
                # 若上次已执行部分轨迹，用最新 joint_states 作下次起点
                if start and not plan_only:
                    refreshed = self._get_joints(list(start.keys()), wait_new=True)
                    if refreshed is not None:
                        start = refreshed
                continue
            break

        if retries > 1:
            log.error(
                f"[{group}] move_action {retries} 次均失败，"
                f"末次错误: {_moveit_error_name(last_code) if last_code is not None else 'unknown'}"
            )
        return no_traj(False)

    def plan_execute_joint_waypoints(
        self,
        group: str,
        speed_scale: float,
        joint_names: Sequence[str],
        waypoints: Sequence[Sequence[float]],
        num_attempts: int | None = None,
    ) -> bool:
        """
        关节空间多点规划（vector<vector>）：
        - joint_names: 关节名顺序
        - waypoints: 每个路径点是一组关节角（与 joint_names 等长）
        逐段：一次 move_action（plan + execute 同时），完成后用 joint_states 作为下一段起点。
        """
        log = self.get_logger()
        if not waypoints:
            log.error(f"[{group}] waypoints 为空")
            return False

        start = self._get_joints(list(joint_names))
        if start is None:
            return False
        planning_attempts = NUM_ATTEMPTS if num_attempts is None else max(1, int(num_attempts))

        log.info(
            f"[{group}] 关节多点路径: {len(waypoints)} waypoints, speed_scale={_clamp01(speed_scale):.2f}, "
            f"planner={PLANNER_ID}, attempts={planning_attempts}, time={PLAN_TIME_SEC:.1f}s"
        )

        for idx, q in enumerate(waypoints):
            goal = [make_joint_constraints_from_vector(group, joint_names, q)]

            ok, used_ms, _ = self.move(
                group, goal, start=start, plan_only=False, speed_scale=speed_scale,
                num_attempts=planning_attempts
            )
            log.info(
                f"[{group}] segment {idx + 1}/{len(waypoints)} plan+exec: {used_ms:.3f} ms "
                f"({'success' if ok else 'failed'})"
            )
            if not ok:
                return False

            start = self._get_joints(list(joint_names), wait_new=True)
            if start is None:
                return False

        return True

    def plan_execute_pose(
        self,
        group: str,
        speed_scale: float,
        link: str,
        pose: Pose,
        use_cartesian: bool = False,
    ) -> bool:
        """位姿目标：直接输入 geometry_msgs/Pose。

        use_cartesian=False（默认）：走 move_action（OMPL 关节空间规划，可绕障）。
        use_cartesian=True         ：走 compute_cartesian_path 服务（笛卡尔直线，
            等价于 RViz MotionPlanning 插件中 "Use Cartesian Path" 复选框）。
        """
        p = pose.position
        log = self.get_logger()
        log.info(
            f"[{group}] 位姿目标 {link} @ {PLAN_FRAME}: "
            f"pos({p.x:.3f}, {p.y:.3f}, {p.z:.3f}), "
            f"quat({pose.orientation.x:.3f}, {pose.orientation.y:.3f}, "
            f"{pose.orientation.z:.3f}, {pose.orientation.w:.3f}), "
            f"speed_scale={_clamp01(speed_scale):.2f}, use_cartesian={use_cartesian}"
        )

        if use_cartesian:
            return self.plan_execute_cartesian_line(
                group, link, pose, speed_scale=speed_scale
            )

        start = self._get_joints(POSE_START_JOINTS, wait_new=True)
        if start is None:
            return False

        goal = [make_pose_constraints(link, pose)]

        ok, used_ms, _ = self.move(group, goal, start=start, plan_only=False, speed_scale=speed_scale)
        log.info(f"[{group}] pose plan+exec: {used_ms:.3f} ms ({'success' if ok else 'failed'})")
        return ok

    def plan_execute_pose_xyz_rpy(
        self,
        group: str,
        speed_scale: float,
        link: str,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
        use_cartesian: bool = False,
    ) -> bool:
        """位姿目标：输入 xyz + rpy；内部转 Pose 后调 plan_execute_pose。"""
        pose = make_pose(x, y, z, roll, pitch, yaw)
        return self.plan_execute_pose(group, speed_scale, link, pose, use_cartesian)

    def _cartesian_plan(
        self,
        group: str,
        link: str,
        end_pose: Pose,
        speed_scale: float = 0.2,
        avoid_collisions: bool = True,
        eef_step: float = CART_EEF_STEP,
        min_fraction: float = CART_MIN_FRACTION,
        start_joints: dict | None = None,
        joint_names: Sequence[str] | None = None,
        plan_frame: str = PLAN_FRAME,
        verbose: bool = True,
    ):
        """调 compute_cartesian_path 服务，规划成功返回（已缩放速度的）RobotTrajectory。

        参数：
            start_joints : 起点关节 {name: pos}。None 时从 joint_states 读取当前关节
                （等价机器人「真实」起点）。指定时可在「不动机器人」的前提下
                预演任意起点出发的笛卡尔路径，用于 IK 多解 + approach 预检。
            verbose      : 是否打印 "笛卡尔直线 ..." / "cartesian planning ..." 信息。
                预检批量调用时建议关掉，避免日志刷屏。

        失败返回 None。返回的 trajectory 可直接缓存以备后续反向播放/重发。
        """
        log = self.get_logger()
        t0 = time.monotonic()

        if not self._cart_cli.wait_for_service(timeout_sec=10.0):
            log.error(f"服务 {SVC_CARTESIAN_PATH} 不可用")
            return None

        start = start_joints if start_joints is not None else self._get_joints(
            list(joint_names) if joint_names is not None else POSE_START_JOINTS, wait_new=True
        )
        if start is None:
            return None

        req = GetCartesianPath.Request()
        req.header.frame_id = plan_frame
        req.start_state.is_diff = True
        req.start_state.joint_state.name = list(start.keys())
        req.start_state.joint_state.position = list(start.values())
        req.group_name = group
        req.link_name = link
        req.waypoints = [end_pose]
        req.max_step = eef_step
        req.jump_threshold = CART_JUMP_THRESHOLD
        req.revolute_jump_threshold = CART_REVOLUTE_JUMP_THRESHOLD
        req.prismatic_jump_threshold = CART_PRISMATIC_JUMP_THRESHOLD
        req.avoid_collisions = avoid_collisions

        p = end_pose.position
        if verbose:
            log.info(
                f"[{group}] 笛卡尔直线 {link} @ {plan_frame}: "
                f"end pos({p.x:.3f}, {p.y:.3f}, {p.z:.3f}), "
                f"max_step={eef_step:.4f}, avoid_collisions={avoid_collisions}, "
                f"jump={CART_JUMP_THRESHOLD:.1f}/{CART_REVOLUTE_JUMP_THRESHOLD:.2f}rad, "
                f"speed_scale={_clamp01(speed_scale):.2f}"
            )

        fut = self._cart_cli.call_async(req)
        if not self._spin_until(fut, 15.0):
            log.error("compute_cartesian_path 超时")
            return None

        res = fut.result()
        code_val = res.error_code.val
        plan_ms = (time.monotonic() - t0) * 1000.0
        if verbose:
            log.info(
                f"[{group}] cartesian planning: {plan_ms:.3f} ms, fraction={res.fraction:.3f}, "
                f"error_code={code_val}({_moveit_error_name(code_val)})"
            )
        if res.fraction < min_fraction or code_val != MoveItErrorCodes.SUCCESS:
            if verbose:
                log.error(
                    f"笛卡尔直线规划未达标：fraction={res.fraction:.3f} < {min_fraction}, "
                    f"error_code={code_val}({_moveit_error_name(code_val)})"
                )
            return None

        traj = res.solution
        s = _clamp01(speed_scale)
        if s > 0.0 and abs(s - 1.0) > 1e-6:
            for pt in traj.joint_trajectory.points:
                total = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
                scaled = total / s
                pt.time_from_start.sec = int(scaled)
                pt.time_from_start.nanosec = int(round((scaled - int(scaled)) * 1e9))
                pt.velocities = [v * s for v in pt.velocities]
                pt.accelerations = [a * s * s for a in pt.accelerations]

        return traj

    def _execute_traj(self, traj, timeout: float = 60.0) -> bool:
        """直接把一条 RobotTrajectory 发给 execute_trajectory action。

        traj 必须已是「速度缩放后」的最终轨迹（_cartesian_plan 或 _reverse_trajectory
        的返回值均已满足）。
        """
        log = self.get_logger()
        if not self._exec_cli.wait_for_server(timeout_sec=10.0):
            log.error(f"动作 {ACT_EXEC_TRAJ} 不可用")
            return False

        goal = ExecuteTrajectory.Goal(trajectory=traj)
        send_fut = self._exec_cli.send_goal_async(goal)
        if not self._spin_until(send_fut, 15.0) or not send_fut.result().accepted:
            log.error("execute_trajectory 目标被拒绝或超时")
            return False
        res_fut = send_fut.result().get_result_async()
        if not self._spin_until(res_fut, timeout):
            log.error("execute_trajectory 结果超时")
            return False

        ar = res_fut.result()
        exec_code = ar.result.error_code.val if ar.result else None
        if ar.status != GoalStatus.STATUS_SUCCEEDED or exec_code != MoveItErrorCodes.SUCCESS:
            log.error(
                f"execute_trajectory 失败：status={ar.status}, "
                f"error_code={exec_code}({_moveit_error_name(exec_code) if exec_code is not None else 'None'})"
            )
            return False
        return True

    @staticmethod
    def _reverse_trajectory(traj):
        """反向播放：positions/time 反序、velocity 取负、acceleration 仅反序（不变号）。

        数学：设原 q(t) on [0,T]，反向播放定义 r(τ) = q(T − τ)，则
            r'(τ)  = -q'(T − τ)    → velocity 每分量取负
            r''(τ) = +q''(T − τ)   → acceleration 不变号（只是按时间反序）
        时间戳：t_new[i] = T − t_old[N-1-i]（即反序后的第 0 点时间为 0）。
        """
        out = copy.deepcopy(traj)
        pts = out.joint_trajectory.points
        if len(pts) < 2:
            return out

        total_ns = pts[-1].time_from_start.sec * 1_000_000_000 + pts[-1].time_from_start.nanosec
        rev_pts = []
        for src in reversed(pts):
            new = copy.deepcopy(src)
            if new.velocities:
                new.velocities = [-v for v in new.velocities]
            old_ns = src.time_from_start.sec * 1_000_000_000 + src.time_from_start.nanosec
            delta = total_ns - old_ns
            new.time_from_start.sec = int(delta // 1_000_000_000)
            new.time_from_start.nanosec = int(delta % 1_000_000_000)
            rev_pts.append(new)
        out.joint_trajectory.points = rev_pts
        return out

    def _solve_ik(
        self,
        group: str,
        link: str,
        pose: Pose,
        seed: dict,
        avoid_collisions: bool = True,
        timeout: float = IK_TIMEOUT_SEC,
        return_code: bool = False,
        plan_frame: str = PLAN_FRAME,
    ):
        """调 /compute_ik 服务，求单个 IK 解。

        seed: RobotState 种子关节字典（至少覆盖 group 内所有关节）。
        默认返回 {joint_name: value} 或 None；
        return_code=True 时返回 (dict_or_None, error_code_int)，
            error_code_int 为 MoveItErrorCodes 数值（-31=NO_IK_SOLUTION，
            -12=GOAL_IN_COLLISION，等等）；服务超时/不可用时返回 None。
        """
        if not self._ik_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f"服务 {SVC_COMPUTE_IK} 不可用")
            return (None, None) if return_code else None

        req = GetPositionIK.Request()
        req.ik_request.group_name = group
        req.ik_request.ik_link_name = link
        req.ik_request.avoid_collisions = avoid_collisions
        req.ik_request.pose_stamped = PoseStamped()
        req.ik_request.pose_stamped.header.frame_id = plan_frame
        req.ik_request.pose_stamped.pose = pose
        req.ik_request.robot_state = RobotState()
        req.ik_request.robot_state.joint_state.name = list(seed.keys())
        req.ik_request.robot_state.joint_state.position = list(seed.values())
        req.ik_request.robot_state.is_diff = True
        secs = int(timeout)
        nsecs = int((timeout - secs) * 1e9)
        req.ik_request.timeout = DurationMsg(sec=secs, nanosec=nsecs)

        fut = self._ik_cli.call_async(req)
        if not self._spin_until(fut, 5.0):
            return (None, None) if return_code else None

        res = fut.result()
        code = int(res.error_code.val)
        if code != MoveItErrorCodes.SUCCESS:
            return (None, code) if return_code else None

        js = res.solution.joint_state
        sol = {n: p for n, p in zip(js.name, js.position)}
        return (sol, code) if return_code else sol

    def _solve_ik_multi(
        self,
        group: str,
        link: str,
        pose: Pose,
        joint_names: Sequence[str],
        n_candidates: int = IK_N_CANDIDATES,
        perturb: float = IK_SEED_PERTURB,
        dedup_tol: float = 1e-2,
        avoid_collisions: bool = True,
        plan_frame: str = PLAN_FRAME,
    ) -> list[dict]:
        """通过随机种子枚举 pose 在 group 上的多个不同 IK 解。

        - 第 0 次以当前 joint_states 为种子，能拿到「最自然」的解。
        - 之后每次对 `joint_names` 列出的关节做 ±perturb 的均匀随机扰动作为种子。
        - 用 dedup_tol 在关节空间做去重（任一关节差异 < tol 视为同解）。
        - 失败时统计 error_code，方便区分「数值无解 / 碰撞被拒 / 输入非法」。
        - 若启用 avoid_collisions 全部失败，自动用 avoid_collisions=False 再试一轮，
          用于区分是「IK 数值无解」还是「IK 有解但被碰撞拒绝」。
        """
        rng = random.Random(IK_RANDOM_SEED) if IK_RANDOM_SEED is not None else random
        log = self.get_logger()

        current = self._get_joints(list(joint_names), wait_new=True)
        if current is None:
            log.error("[ik-multi] 读取当前关节失败，无法构造种子")
            return []

        solutions: list[dict] = []
        fail_codes: dict[int, int] = {}

        def _is_dup(cand: dict) -> bool:
            for s in solutions:
                if all(abs(cand[n] - s[n]) < dedup_tol for n in joint_names if n in cand and n in s):
                    return True
            return False

        for i in range(n_candidates):
            if i == 0:
                seed = dict(current)
            else:
                seed = {n: current[n] + rng.uniform(-perturb, perturb) for n in joint_names}
            sol, code = self._solve_ik(
                group, link, pose, seed,
                avoid_collisions=avoid_collisions, return_code=True, plan_frame=plan_frame,
            )
            if sol is None:
                if code is not None:
                    fail_codes[code] = fail_codes.get(code, 0) + 1
                continue
            sub = {n: sol[n] for n in joint_names if n in sol}
            if len(sub) != len(joint_names):
                continue
            if not _is_dup(sub):
                solutions.append(sub)

        log.info(
            f"[ik-multi] {len(solutions)} 个不同 IK 解 / {n_candidates} 次尝试 "
            f"(perturb=±{perturb:.2f} rad, avoid_collisions={avoid_collisions})"
        )
        if fail_codes:
            breakdown = ", ".join(
                f"{_moveit_error_name(c)}={n}" for c, n in sorted(fail_codes.items())
            )
            log.info(f"[ik-multi] 失败原因分布: {breakdown}")

        # 全失败时打印 pose 详情，并尝试一次 avoid_collisions=False 以区分原因
        if not solutions:
            p = pose.position
            o = pose.orientation
            log.error(
                f"[ik-multi] target pose 详情: frame={plan_frame}, link={link}, group={group}\n"
                f"           position=({p.x:.4f}, {p.y:.4f}, {p.z:.4f})\n"
                f"           quat=({o.x:.4f}, {o.y:.4f}, {o.z:.4f}, {o.w:.4f})"
            )
            if avoid_collisions:
                log.warning(
                    "[ik-multi] 重试一轮 avoid_collisions=False，用于区分「数值无解 vs 碰撞被拒」"
                )
                no_col_sols = self._solve_ik_multi(
                    group, link, pose, joint_names,
                    n_candidates=n_candidates, perturb=perturb, dedup_tol=dedup_tol,
                    avoid_collisions=False, plan_frame=plan_frame,
                )
                if no_col_sols:
                    log.warning(
                        f"[ik-multi] 关闭碰撞后找到 {len(no_col_sols)} 个 IK 解 → "
                        f"目标位姿本身可达，但 IK 解全在碰撞中。"
                        f"  对策：检查 default_robot_padding / 自碰撞 ACM / 抬高 target z / 改姿态"
                    )
                else:
                    log.error(
                        "[ik-multi] 关闭碰撞后仍 0 解 → 目标位姿真正不可达（数值无解 / 超出工作空间 / 关节限位）。"
                        "  对策：抬高 target z、放大 IK_SEED_PERTURB / IK_TIMEOUT_SEC、或在 RViz 拖动 IK marker 直接验证可达性"
                    )
        return solutions

    def _solve_ik_from_seeds(
        self,
        group: str,
        link: str,
        pose: Pose,
        joint_names: Sequence[str],
        seeds: Sequence[dict[str, float]],
        dedup_tol: float = 1e-2,
        avoid_collisions: bool = True,
        plan_frame: str = PLAN_FRAME,
    ) -> list[dict]:
        """只使用调用方给定的 seed 求 IK，不额外尝试当前关节或随机初始值。"""
        log = self.get_logger()
        joint_names = list(joint_names)
        solutions: list[dict] = []
        fail_codes: dict[int, int] = {}

        def _is_dup(cand: dict) -> bool:
            for s in solutions:
                if all(abs(cand[n] - s[n]) < dedup_tol for n in joint_names if n in cand and n in s):
                    return True
            return False

        for idx, raw_seed in enumerate(seeds):
            missing_seed = [n for n in joint_names if n not in raw_seed]
            if missing_seed:
                log.warning(
                    f"[ik-seed] seed {idx + 1}/{len(seeds)} 缺少关节 {missing_seed}，跳过"
                )
                continue

            seed = {n: raw_seed[n] for n in joint_names}
            sol, code = self._solve_ik(
                group, link, pose, seed,
                avoid_collisions=avoid_collisions, return_code=True, plan_frame=plan_frame,
            )
            if sol is None:
                if code is not None:
                    fail_codes[code] = fail_codes.get(code, 0) + 1
                continue

            sub = {n: sol[n] for n in joint_names if n in sol}
            if len(sub) != len(joint_names):
                log.warning(
                    f"[ik-seed] seed {idx + 1}/{len(seeds)} 返回解缺少部分关节，跳过"
                )
                continue
            if not _is_dup(sub):
                solutions.append(sub)

        log.info(
            f"[ik-seed] {len(solutions)} 个不同 IK 解 / {len(seeds)} 个指定 seed "
            f"(avoid_collisions={avoid_collisions})"
        )
        if fail_codes:
            breakdown = ", ".join(
                f"{_moveit_error_name(c)}={n}" for c, n in sorted(fail_codes.items())
            )
            log.info(f"[ik-seed] 失败原因分布: {breakdown}")
        return solutions

    def _select_feasible_grasp_pair(
        self,
        group: str,
        link: str,
        target_pose: Pose,
        pre_pose: Pose,
        joint_names: Sequence[str],
        speed_scale: float,
        n_candidates: int = IK_N_CANDIDATES,
        plan_frame: str = PLAN_FRAME,
        target_seeds: Sequence[dict[str, float]] | None = None,
    ):
        """从 target_pose 的多个 IK 解里挑「能直线退回 pre_pose」的一组。

        target_seeds 不为 None 时，只用这些 seed 求 IK，不做当前关节/随机 seed 枚举。
        返回 (q_pre, q_target, approach_traj) 三元组；找不到返回 None。
        approach_traj = reverse( cartesian(start=q_target, end=pre_pose) )
            即真正用来「从 pre_pose 直线接近 target_pose」的轨迹。
        """
        log = self.get_logger()
        if target_seeds is None:
            candidates = self._solve_ik_multi(
                group, link, target_pose, joint_names, n_candidates, plan_frame=plan_frame
            )
        else:
            candidates = self._solve_ik_from_seeds(
                group, link, target_pose, joint_names, target_seeds, plan_frame=plan_frame
            )
        if not candidates:
            log.error("[grasp-select] target_pose 在该 group 上没有任何 IK 解")
            return None

        line_distance = _pose_distance(target_pose, pre_pose)
        expected_points = int(math.ceil(line_distance / max(CART_EEF_STEP, 1e-6))) + 2
        max_points = max(
            expected_points + CART_MAX_POINT_EXTRA,
            int(math.ceil(expected_points * CART_MAX_POINT_FACTOR)),
        )
        best = None
        best_score = float("inf")

        for idx, q_target in enumerate(candidates):
            retreat_traj = self._cartesian_plan(
                group, link, pre_pose,
                speed_scale=speed_scale,
                start_joints=q_target,
                joint_names=joint_names,
                plan_frame=plan_frame,
                verbose=False,
            )
            if retreat_traj is None:
                log.info(
                    f"[grasp-select] 候选 IK {idx + 1}/{len(candidates)}：retreat 不可行 → 淘汰"
                )
                continue

            point_count = len(retreat_traj.joint_trajectory.points)
            total_motion, max_step = _trajectory_joint_stats(retreat_traj)
            if point_count > max_points:
                log.warning(
                    f"[grasp-select] 候选 IK {idx + 1}/{len(candidates)}："
                    f"retreat 点数异常 {point_count}>{max_points} "
                    f"(理论约 {expected_points} 点, 直线距离 {line_distance:.3f} m) → 淘汰"
                )
                continue
            jump_reason = _trajectory_joint_jump_reason(retreat_traj)
            if jump_reason:
                log.warning(
                    f"[grasp-select] 候选 IK {idx + 1}/{len(candidates)}："
                    f"检测到关节跳变（{jump_reason}）→ 淘汰"
                )
                continue

            approach_traj = self._reverse_trajectory(retreat_traj)
            last_pt = retreat_traj.joint_trajectory.points[-1]
            names = list(retreat_traj.joint_trajectory.joint_names)
            q_pre = {n: p for n, p in zip(names, last_pt.positions)}
            score = total_motion + 0.01 * point_count
            log.info(
                f"[grasp-select] 候选 IK {idx + 1}/{len(candidates)}：retreat 可行 ✓ "
                f"(轨迹 {point_count} 点, joint_motion={total_motion:.3f}, "
                f"max_step={max_step:.3f}, score={score:.3f})"
            )
            if score < best_score:
                best_score = score
                best = (q_pre, q_target, approach_traj, idx + 1, point_count, total_motion, max_step)

        if best is not None:
            q_pre, q_target, approach_traj, best_idx, point_count, total_motion, max_step = best
            log.info(
                f"[grasp-select] 选用候选 IK {best_idx}/{len(candidates)}："
                f"{point_count} 点, joint_motion={total_motion:.3f}, max_step={max_step:.3f}"
            )
            return q_pre, q_target, approach_traj

        log.error(
            f"[grasp-select] 共 {len(candidates)} 个候选 IK 解，没有一个能直线退回 pre_pose"
        )
        return None

    def _select_feasible_grasp_pair_with_waist(
        self,
        group: str,
        link: str,
        target_pose: Pose,
        pre_pose: Pose,
        joint_names: Sequence[str],
        speed_scale: float,
        n_candidates: int = IK_N_CANDIDATES,
        plan_frame: str = PLAN_FRAME,
    ):
        """带腰 group 的可达性选择。

        先只在 body group 上求 target_pose IK；对每个 body IK 解，取其中腰部关节
        计算 target/pre 在纯臂 base_link 下的位姿，再只用该 IK 解里的臂关节作为
        seed 调纯臂 _select_feasible_grasp_pair 做 approach 预检。
        """
        log = self.get_logger()
        arm_context = arm_context_for_body_group(group)
        if arm_context is None:
            log.error(f"[grasp-select-body] {group} 不是带腰抓取组")
            return None
        arm_group, arm_plan_frame = arm_context
        arm_joint_names = joint_names_for_group(arm_group)
        joint_names = list(joint_names)

        body_candidates = self._solve_ik_multi(
            group, link, target_pose, joint_names, n_candidates, plan_frame=plan_frame
        )
        if not body_candidates:
            log.error("[grasp-select-body] target_pose 在带腰 group 上没有任何 IK 解")
            return None

        best = None
        best_score = float("inf")

        for idx, q_body in enumerate(body_candidates):
            if WAIST_BODY_JOINT not in q_body:
                log.warning(
                    f"[grasp-select-body] body IK {idx + 1}/{len(body_candidates)} "
                    f"缺少 {WAIST_BODY_JOINT}，跳过"
                )
                continue
            missing_arm = [n for n in arm_joint_names if n not in q_body]
            if missing_arm:
                log.warning(
                    f"[grasp-select-body] body IK {idx + 1}/{len(body_candidates)} "
                    f"缺少臂关节 {missing_arm}，跳过"
                )
                continue

            waist_angle = q_body[WAIST_BODY_JOINT]
            waist_joints = {WAIST_BODY_JOINT: waist_angle}
            arm_base_in_body_frame = self._get_link_pose_fk(
                arm_plan_frame,
                joints=waist_joints,
                plan_frame=plan_frame,
            )
            if arm_base_in_body_frame is None:
                log.warning(
                    f"[grasp-select-body] body IK {idx + 1}/{len(body_candidates)}: "
                    f"无法用 {WAIST_BODY_JOINT}={waist_angle:.6f} 计算 "
                    f"{arm_plan_frame} @ {plan_frame}"
                )
                continue

            t_arm_body = invert_transform4(pose_to_matrix(arm_base_in_body_frame))

            def _to_arm_frame(pose: Pose) -> Pose:
                return xyz_rpy_to_pose(matrix_to_xyz_rpy(matmul4(t_arm_body, pose_to_matrix(pose))))

            arm_target_pose = _to_arm_frame(target_pose)
            arm_pre_pose = _to_arm_frame(pre_pose)
            arm_seed = {n: q_body[n] for n in arm_joint_names}
            log.info(
                f"[grasp-select-body] body IK {idx + 1}/{len(body_candidates)}: "
                f"{WAIST_BODY_JOINT}={waist_angle:.6f} rad，"
                f"转到 {arm_group} @ {arm_plan_frame} 后只用该臂关节 seed 验证"
            )

            picked = self._select_feasible_grasp_pair(
                arm_group,
                link,
                arm_target_pose,
                arm_pre_pose,
                joint_names=arm_joint_names,
                speed_scale=speed_scale,
                plan_frame=arm_plan_frame,
                target_seeds=[arm_seed],
            )
            if picked is None:
                log.info(
                    f"[grasp-select-body] body IK {idx + 1}/{len(body_candidates)}: "
                    "纯臂 seeded approach 不可行"
                )
                continue

            q_pre_arm, q_target_arm, approach_traj = picked
            point_count = len(approach_traj.joint_trajectory.points)
            total_motion, max_step = _trajectory_joint_stats(approach_traj)
            score = total_motion + 0.01 * point_count
            q_pre = {WAIST_BODY_JOINT: waist_angle, **q_pre_arm}
            q_target = {WAIST_BODY_JOINT: waist_angle, **q_target_arm}
            log.info(
                f"[grasp-select-body] body IK {idx + 1}/{len(body_candidates)}: "
                f"可行 ✓ (轨迹 {point_count} 点, joint_motion={total_motion:.3f}, "
                f"max_step={max_step:.3f}, score={score:.3f})"
            )
            log.info(
                f"{GREEN}[grasp-select-body] body IK {idx + 1}/{len(body_candidates)} "
                f"可行 arm_target_pose: {arm_target_pose}{RESET}"
            )
            if score < best_score:
                best_score = score
                best = (
                    q_pre,
                    q_target,
                    approach_traj,
                    idx + 1,
                    point_count,
                    total_motion,
                    max_step,
                )

        if best is not None:
            q_pre, q_target, approach_traj, best_idx, point_count, total_motion, max_step = best
            log.info(
                f"[grasp-select-body] 选用 body IK {best_idx}/{len(body_candidates)}："
                f"{WAIST_BODY_JOINT}={q_target[WAIST_BODY_JOINT]:.6f} rad, "
                f"{point_count} 点, joint_motion={total_motion:.3f}, max_step={max_step:.3f}"
            )
            return q_pre, q_target, approach_traj

        log.error(
            f"[grasp-select-body] 共 {len(body_candidates)} 个带腰 IK 解，"
            "没有一个能通过纯臂 seeded approach 预检"
        )
        return None

    def plan_execute_cartesian_line(
        self,
        group: str,
        link: str,
        end_pose: Pose,
        speed_scale: float = 0.2,
        avoid_collisions: bool = True,
        eef_step: float = CART_EEF_STEP,
        min_fraction: float = CART_MIN_FRACTION,
    ) -> bool:
        """从当前末端位姿直线移动到 end_pose：先 _cartesian_plan 再 _execute_traj。

        - 走 compute_cartesian_path 服务：服务从 start_state 做 FK 得到当前 EE 位姿，
          再沿 waypoints[0]=end_pose 做直线段（关节空间逐段 IK 拼接而成）。
        - 拿到 RobotTrajectory 后用 execute_trajectory action 执行。
        - Humble 的 GetCartesianPath 没有 max_velocity_scaling_factor 字段，速度缩放
          在客户端通过缩放 time_from_start / velocities / accelerations 实现。
        - fraction < min_fraction 视为失败（默认 0.99，要求基本走完整条直线）。
        """
        log = self.get_logger()
        t0 = time.monotonic()
        traj = self._cartesian_plan(
            group, link, end_pose, speed_scale, avoid_collisions, eef_step, min_fraction
        )
        if traj is None:
            return False
        if not self._execute_traj(traj):
            return False
        log.info(f"[{group}] cartesian plan+exec: {(time.monotonic() - t0) * 1000.0:.3f} ms (success)")
        return True

    def _place_and_return(
        self,
        speed: float,
        place_joints: dict[str, Sequence[float]],
        *,
        group: str,
        link: str,
        plan_frame: str,
        first_return_mode: int,
        line_speed: float | None = None,
    ) -> bool:
        """执行放置段：模式 1 使用 *_j 放置点，其余模式使用普通放置点。

        参数：
            speed        : 5/8 OMPL 到 yubei 的速度缩放（0~1）。
            place_joints : 包含 yubei/fang 和 yubei_j/fang_j 的关节目标。
            group        : SRDF 规划组，如 right_arm。
            link         : 做笛卡尔直线运动的末端 link。
            plan_frame   : FK 与笛卡尔路径使用的规划坐标系。
            first_return_mode: 1 选择 *_j；0/2 选择普通 yubei/fang。
            line_speed   : 6/8 yubei → fang 笛卡尔直线速度；None 时复用 speed。
        """
        log = self.get_logger()
        place_joint_names = joint_names_for_group(group)
        line_speed = speed if line_speed is None else line_speed
        yubei_key = "yubei_j" if first_return_mode == 1 else "yubei"
        fang_key = "fang_j" if first_return_mode == 1 else "fang"
        if not isinstance(place_joints, dict) or yubei_key not in place_joints or fang_key not in place_joints:
            log.error(f"[pick] group={group} 的放置配置必须包含 {yubei_key} 和 {fang_key}")
            return False
        yubei_joints = list(place_joints[yubei_key])
        fang_joints = list(place_joints[fang_key])
        if len(yubei_joints) != len(place_joint_names) or len(fang_joints) != len(place_joint_names):
            log.error(
                f"[pick] group={group} 放置配置长度错误：{yubei_key}={len(yubei_joints)}, "
                f"{fang_key}={len(fang_joints)}, 关节数={len(place_joint_names)}"
            )
            return False

        log.info(f"[pick] 5/8  OMPL → {yubei_key} 放置预备位")
        current = self._get_joints(place_joint_names, wait_new=True)
        if current is None:
            log.error("[pick] 读取当前关节失败")
            return False
        place_start_joints = current
        goal = [make_joint_constraints_from_vector(group, place_joint_names, yubei_joints)]
        ok, used_ms, to_yubei_traj = self.move(
            group, goal, start=current, plan_only=False, speed_scale=speed
        )
        log.info(f"[pick] OMPL → {yubei_key}: {used_ms:.3f} ms ({'success' if ok else 'failed'})")
        if not ok:
            log.error(f"[pick] 运动到 {yubei_key} 失败")
            return False
        if to_yubei_traj is None or not to_yubei_traj.joint_trajectory.points:
            log.error(f"[pick] 5/8 未返回到 {yubei_key} 的轨迹，无法原路返回")
            return False
        log.info(
            f"[pick] {yubei_key} 关节: "
            + ", ".join(f"{n}={v:.3f}" for n, v in zip(place_joint_names, yubei_joints))
        )

        log.info("按回车继续 …")
        try:
            input()
        except EOFError:
            pass

        log.info(f"[pick] 6/8  末端笛卡尔直线 {yubei_key} → {fang_key}")
        current = self._get_joints(place_joint_names, wait_new=True)
        if current is None:
            log.error("[pick] 读取当前关节失败")
            return False
        fang_state = dict(zip(place_joint_names, fang_joints))
        fang_pose = self._get_link_pose_fk(
            link,
            joints=fang_state,
            plan_frame=plan_frame,
        )
        if fang_pose is None:
            log.error(f"[pick] 无法由 {fang_key} 关节角计算目标末端位姿")
            return False
        yubei_to_fang_traj = self._cartesian_plan(
            group,
            link,
            fang_pose,
            speed_scale=line_speed,
            avoid_collisions=False,
            start_joints=current,
            joint_names=place_joint_names,
            plan_frame=plan_frame,
        )
        if yubei_to_fang_traj is None:
            log.error(f"[pick] {yubei_key} → {fang_key} 笛卡尔直线规划失败")
            return False
        if not self._execute_traj(yubei_to_fang_traj):
            log.error(f"[pick] {yubei_key} → {fang_key} 笛卡尔直线执行失败")
            return False
        log.info(
            f"[pick] {fang_key} 目标关节（用于 FK 生成直线终点）: "
            + ", ".join(f"{n}={v:.3f}" for n, v in zip(place_joint_names, fang_joints))
        )

        log.info("按回车继续 …")
        try:
            input()
        except EOFError:
            pass

        tool_side = tool_side_for_link(link)
        if tool_side is None:
            log.error(f"[pick] link={link} 不是 l_tool 或 r_tool，无法判断工具侧")
            return False
        tool_label = "左臂" if tool_side == "left" else "右臂"
        if not self.set_tool_power(tool_side, 0):
            log.error(f"[pick] {tool_label}工具下电失败")
            return False
        print(f"\033[32m{tool_label}下电成功\033[0m")

        log.info(
            f"[pick] 7/8  原路返回 ①：反向播放 6/8，{fang_key} → {yubei_key} "
            f"（{len(yubei_to_fang_traj.joint_trajectory.points)} 点）"
        )
        if not self._execute_traj(self._reverse_trajectory(yubei_to_fang_traj)):
            log.error("[pick] 7/8 反向播放 6/8 失败")
            return False

        q1_count = len(place_joint_names)
        if group == "left_arm":
            q1_source = list(EXCHANGE_Q1[:6])
            q1_joints = q1_source[:q1_count]
            q1_slice_text = "前6"
        elif group == "right_arm":
            q1_source = list(EXCHANGE_Q1[-6:])
            q1_joints = q1_source[-q1_count:]
            q1_slice_text = "后6"
        else:
            log.error(f"[pick] 8/8 group={group} 不支持按 Q1 固定点返回")
            return False
        if len(q1_joints) != q1_count:
            log.error(
                f"[pick] 8/8 Q1 固定点长度错误：Q1={len(q1_joints)}, "
                f"关节数={q1_count}"
            )
            return False

        log.info(
            f"[pick] 8/8  从当前位置运动到 Q1 固定点 "
            f"（{q1_slice_text} 个数，使用 {q1_count} 个关节）"
        )
        current = self._get_joints(place_joint_names, wait_new=True)
        if current is None:
            log.error("[pick] 8/8 读取当前关节失败")
            return False
        goal = [make_joint_constraints_from_vector(group, place_joint_names, q1_joints)]
        ok, used_ms, _to_q1_traj = self.move(
            group, goal, start=current, plan_only=False, speed_scale=speed
        )
        log.info(f"[pick] OMPL → Q1 固定点: {used_ms:.3f} ms ({'success' if ok else 'failed'})")
        if not ok:
            log.error("[pick] 8/8 运动到 Q1 固定点失败")
            return False
        log.info(
            "[pick] Q1 固定点返回完成: "
            + ", ".join(f"{n}={v:.3f}" for n, v in zip(place_joint_names, q1_joints))
        )
        return True

    def pick_and_return(
        self,
        target_pose: Pose,
        speed_scale: float,
        group: str,
        link: str,
        plan_frame: str,
        joint_names: Sequence[str],
        place_joints: dict[str, Sequence[float]],
        place_speed_scale: float = PLACE_SPEED_SCALE,
        cutoff_joint_names: Sequence[str] | None = None,
        first_return_mode: int = FIRST_RETURN_MODE,
        waist_moved: bool = False,
        waist_reset_angle: float = WAIST_RESET_ANGLE_RAD,
    ) -> bool:
        """抓取流程（IK 多解 + approach 预检 + 放置 + 原路返回）。

        参数：
            target_pose       : 末端抓取位姿（geometry_msgs/Pose，在 plan_frame 下表达）
            speed_scale       : 抓取段速度缩放（0~1）
            group             : SRDF 规划组（如 left_body）
            link              : 末端连杆名（如 L6）
            plan_frame        : 位姿/IK/笛卡尔规划使用的坐标系（如 base_link、world）
            joint_names       : group 内关节名顺序（与 yubei/fang 数组一一对应）
            place_joints      : 含 yubei、fang 的放置关节目标 [rad]
            place_speed_scale : 放置段的速度缩放（0~1）
            cutoff_joint_names: 用于计算 PLAN_FRAME → SCENE_FRAME 的关节名；None 时使用 joint_names
            first_return_mode : 1=使用 *_j 并交换；2=使用普通放置点；0 兼容旧普通模式
            waist_moved       : 抓取前是否为了可达性移动过腰部；若移动过，放置/交换前先回 30°
            waist_reset_angle : 腰部回正角度 [rad]
        """
        log = self.get_logger()
        self.last_pick_failure_reason = None
        joint_names = list(joint_names)
        cutoff_joint_names = list(cutoff_joint_names) if cutoff_joint_names is not None else joint_names
        first_return_mode = int(first_return_mode)
        if first_return_mode not in (0, 1, 2):
            log.error(f"[pick] first_return_mode 只能是 0、1 或 2，当前={first_return_mode}")
            return False
        yubei_key = "yubei_j" if first_return_mode == 1 else "yubei"
        fang_key = "fang_j" if first_return_mode == 1 else "fang"
        if not isinstance(place_joints, dict) or yubei_key not in place_joints or fang_key not in place_joints:
            log.error(f"[pick] place_joints 必须包含 {yubei_key} 和 {fang_key}")
            return False
        if any(len(place_joints[name]) != len(joint_names) for name in (yubei_key, fang_key)):
            log.error(
                f"[pick] {yubei_key}/{fang_key} 的关节数必须等于 joint_names 长度 {len(joint_names)}"
            )
            return False

        log.info(
            f"[pick] group={group}, link={link}, plan_frame={plan_frame}"
        )
        # 抓取前先下电
        tool_side = tool_side_for_link(link)
        if tool_side is None:
            log.error(f"[pick] link={link} 不是 l_tool 或 r_tool，无法判断抓取工具侧")
            return False
        tool_label = "左臂" if tool_side == "left" else "右臂"
        if not self.set_tool_power(tool_side, 0):
            log.error(f"[pick] {tool_label}工具下电失败")
            return False

        pre_pose = pose_offset_local_z(target_pose, PRE_GRASP_OFFSET)
        pp = pre_pose.position
        tp = target_pose.position
        log.info(
            f"[pick] 抓取目标 @ {plan_frame} pos({tp.x:.3f}, {tp.y:.3f}, {tp.z:.3f}); "
            f"预备点（沿末端 z 退 {PRE_GRASP_OFFSET:.3f} m）pos({pp.x:.3f}, {pp.y:.3f}, {pp.z:.3f})"
        )

        log.info("[pick] 0/8  IK 多解枚举 + approach 预检 …")
        picked = self._select_feasible_grasp_pair(
            group, link, target_pose, pre_pose,
            joint_names=joint_names,
            speed_scale=speed_scale,
            plan_frame=plan_frame,
        )
        if picked is None:
            log.error("[pick] 未找到「IK 可解 + cartesian approach 可行」的 IK 解")
            self.last_pick_failure_reason = "no_ik"
            return False
        q_pre, q_target, approach_traj = picked
        log.info(
            "[pick] 选定 q_pre: "
            + ", ".join(f"{n}={q_pre[n]:.3f}" for n in joint_names)
        )

        log.info("[pick] 1/8  OMPL  → q_pre（关节目标，IK 解已确定）")
        current = self._get_joints(joint_names, wait_new=True)
        if current is None:
            log.error("[pick] 读取当前关节失败")
            return False
        pick_start_joints = current
        goal = [make_joint_constraints(group, q_pre)]

        # IK 多解与 q_pre → q_target 直线 approach 预检在无隔板场景下完成；
        # 隔板只用于这一段 OMPL：从当前状态运动到 q_pre。
        cutoff_added = False
        cutoff_removed = True
        try:
            if not self.add_frame_cutoff_for_pose(
                target_pose,
                source_frame=plan_frame,
                target_frame=SCENE_FRAME,
                joint_names=cutoff_joint_names,
            ):
                log.error("[pick] 添加 q_pre OMPL 隔板失败")
                return False
            cutoff_added = True
            ok, used_ms, to_pre_traj = self.move(
                group, goal, start=current, plan_only=False, speed_scale=speed_scale
            )
        finally:
            if cutoff_added:
                log.info(f"[pick] 移除「{FRAME_CUTOFF_ID}」，后续直线 approach 不使用隔板")
                cutoff_removed = self.remove_frame_cutoff()

        log.info(f"[pick] OMPL → q_pre: {used_ms:.3f} ms ({'success' if ok else 'failed'})")
        if not cutoff_removed:
            log.error("[pick] 移除 q_pre OMPL 隔板失败")
            return False
        if not ok:
            log.error("[pick] OMPL 到 q_pre 失败")
            self.last_pick_failure_reason = "q_pre"
            return False
        if to_pre_traj is None or not to_pre_traj.joint_trajectory.points:
            log.error("[pick] 1/8 未返回 OMPL 轨迹，无法原路退回初始位置")
            return False

        log.info(
            f"[pick] 2/8  执行已缓存的 approach 轨迹 → q_target "
            f"（{len(approach_traj.joint_trajectory.points)} 点，免重规划）"
        )
        if not self._execute_traj(approach_traj):
            log.error("[pick] 直线接近执行失败")
            return False

        self._log_actual_fk_error(
            "[pick] 3/8",
            link=link,
            target_pose=target_pose,
            plan_frame=plan_frame,
            joint_names=joint_names,
        )
        log.info("[pick] 3/8  到达抓取位置，按回车继续 …")
        try:
            input()
        except EOFError:
            pass

        if not self.set_tool_power(tool_side, 1):
            log.error(f"[pick] {tool_label}工具上电失败")
            return False
        print(f"\033[32m{tool_label}上电成功\033[0m")
        # 日志
        return_desc = (
            "反向 approach → q_pre"
            if first_return_mode == 1
            else "反向 approach + 1/8 OMPL → 1/8 初始位置"
        )
        log.info(f"[pick] 4/8  第一段复位：{return_desc}")
        retreat_approach = self._reverse_trajectory(approach_traj)
        if not self._execute_traj(retreat_approach):
            log.error("[pick] 反向播放 approach 失败")
            return False
        
        if not self.add_frame_cutoff_only_for_pose(
            target_pose,
            source_frame=plan_frame,
            target_frame=SCENE_FRAME,
            joint_names=cutoff_joint_names,
        ):
            log.error("[pick] 第一段复位后添加隔板失败")
            return False

        if waist_moved:
            log.info(
                f"[pick] 抓取前腰部运动过，交换/复位前先回腰到 "
                f"{waist_reset_angle:.6f} rad ({math.degrees(waist_reset_angle):.2f} deg)"
            )
            if not self.move_body_joint2(waist_reset_angle, speed_scale=0.5):
                log.error("[pick] 腰部回 30 度失败")
                return False

        if first_return_mode == 1:
            log.info(
                "[pick] 第一段复位完成，已反向直线回到 q_pre: "
                + ", ".join(f"{n}={q_pre[n]:.3f}" for n in joint_names)
            )
            
            log.info("[pick] 3/8  到达抓取位置，按回车继续 …")
            try:
                input()
            except EOFError:
                pass

            exchange_q3 = EXCHANGE_Q3.get(tool_side)
            if exchange_q3 is None:
                log.error(f"[pick] EXCHANGE_Q3 未配置 {tool_side}")
                return False

            if not self.dual_arm_exchange(
                exchange_q3,
                EXCHANGE_Q2,
                source_link=link,
                dual_speed=place_speed_scale,
                cartesian_speed=speed_scale,
            ):
                return False

            if tool_side == "right":
                place_group = "left_arm"
                place_link = "l_tool"
                place_frame = "l_base_link"
            else:
                place_group = "right_arm"
                place_link = "r_tool"
                place_frame = "r_base_link"

            if not self._place_and_return(
                place_speed_scale,
                place_joints=PLACE_JOINTS[place_group],
                group=place_group,
                link=place_link,
                plan_frame=place_frame,
                first_return_mode=first_return_mode,
                line_speed=speed_scale,
            ):
                return False

            
        else:
            # retreat_to_start = self._reverse_trajectory(to_pre_traj)
            # if not self._execute_traj(retreat_to_start):
            #     log.error("[pick] 反向播放 1/8 OMPL 回到初始位置失败")
            #     return False
            # log.info(
            #     "[pick] 第一段复位完成，已回到 1/8 初始位置: "
            #     + ", ".join(f"{n}={pick_start_joints[n]:.3f}" for n in joint_names)
            # )
        
            if not self._place_and_return(
                place_speed_scale,
                place_joints,
                group=group,
                link=link,
                plan_frame=plan_frame,
                first_return_mode=first_return_mode,
                line_speed=speed_scale,
            ):
                return False

        log.info("[pick] 抓取流程完成。")
        return True

    def dual_arm_exchange(
        self,
        q1: Sequence[float],
        q2_by_side: dict[str, Sequence[float]],
        *,
        source_link: str,
        dual_speed: float = 0.2,
        cartesian_speed: float = 0.2,
        z_down_distance: float | None = None,
    ) -> bool:
        """双臂交换：根据持物末端选择 Q2、运动臂和相反侧接物工具。"""
        log = self.get_logger()
        dual_group = "dual_arm_y"
        dual_joint_names = joint_names_for_group(dual_group)

        source_side = tool_side_for_link(source_link)
        if source_side is None:
            log.error(f"[exchange] source_link={source_link} 不是 l_tool 或 r_tool")
            return False
        if z_down_distance is None:
            z_down_distance = -0.1095 if source_side == "right" else -0.1215
        receiver_side = "left" if source_side == "right" else "right"
        source_label = "右臂" if source_side == "right" else "左臂"
        receiver_label = "左臂" if receiver_side == "left" else "右臂"

        if source_side == "right":
            source_group = "right_arm"
            source_plan_frame = "r_base_link"
        else:
            source_group = "left_arm"
            source_plan_frame = "l_base_link"
        source_joint_names = joint_names_for_group(source_group)

        if source_side not in q2_by_side:
            log.error(f"[exchange] EXCHANGE_Q2 未配置 {source_side}")
            return False
        q2 = q2_by_side[source_side]

        if not self.set_tool_power(receiver_side, 0):
            log.error(f"[exchange] {receiver_label}工具下电失败")
            return False

        if len(q1) != len(dual_joint_names):
            log.error(f"[exchange] q1 长度错误: {len(q1)} != {len(dual_joint_names)}")
            return False
        if len(q2) != len(dual_joint_names):
            log.error(f"[exchange] q2 长度错误: {len(q2)} != {len(dual_joint_names)}")
            return False

        log.info("[exchange] 1/4  dual_arm 规划执行到 q2")
        if not self.plan_execute_joint_waypoints(dual_group, dual_speed, dual_joint_names, [q2], num_attempts=20):
            log.error("[exchange] dual_arm 到 q2 失败")
            return False

        log.info(
            f"[exchange] 2/4  {source_group} 沿 {source_link} 末端坐标系 -z 轴直线移动 "
            f"{z_down_distance:.3f} m"
        )

        log.info("按回车继续 …")
        try:
            input()
        except EOFError:
            pass

        current = self._get_joints(source_joint_names, wait_new=True)
        if current is None:
            log.error(f"[exchange] 读取 {source_group} 当前关节失败")
            return False

        ee_pose = self._get_link_pose_fk(
            source_link,
            current,
            joint_names=source_joint_names,
            plan_frame=source_plan_frame,
        )
        if ee_pose is None:
            log.error(f"[exchange] FK 读取 {source_group} 当前末端位姿失败")
            return False

        down_pose = pose_offset_local_z(ee_pose, abs(z_down_distance))
        down_traj = self._cartesian_plan(
            source_group,
            source_link,
            down_pose,
            speed_scale=cartesian_speed,
            avoid_collisions=False,
            joint_names=source_joint_names,
            plan_frame=source_plan_frame,
        )
        if down_traj is None or not down_traj.joint_trajectory.points:
            log.error(f"[exchange] {source_group} z 向下直线规划失败")
            return False
        if not self._execute_traj(down_traj):
            log.error(f"[exchange] {source_group} z 向下直线执行失败")
            return False

        log.info("按回车继续 …")
        try:
            input()
        except EOFError:
            pass

        if not self.set_tool_power(source_side, 0):
            log.error(f"[exchange] {source_label}工具下电失败")
            return False
        if not self.set_tool_power(receiver_side, 1):
            log.error(f"[exchange] {receiver_label}工具上电失败")
            return False
        print(f"\033[32m{source_label}下电、{receiver_label}上电成功\033[0m")


        log.info(
            f"[exchange] 3/4  {source_group} 沿刚才直线返回 "
            f"（{len(down_traj.joint_trajectory.points)} 点）"
        )
        if not self._execute_traj(self._reverse_trajectory(down_traj)):
            log.error(f"[exchange] {source_group} 反向直线返回失败")
            return False

        log.info("[exchange] 4/4  dual_arm 规划执行到 q1")
        if not self.plan_execute_joint_waypoints(dual_group, dual_speed, dual_joint_names, [q1]):
            log.error("[exchange] dual_arm 到 q1 失败")
            return False

        log.info("[exchange] 双臂交换流程完成")
        return True


# =============================================================================
# 主流程
# =============================================================================


def main(argv: list[str] | None = None) -> int:
    rclpy.init(args=argv)
    node = G01Demo()
    log = node.get_logger()
    code = 1
    frame_added = False

    def cleanup_grasp_display(remove_scene_objects: bool = True) -> None:
        node.remove_cylinder_at_pose()
        node.remove_cylinder_at_pose(object_id=Z_AXIS_MARKER_ID)
        if remove_scene_objects and not node.remove_frame_cutoff():
            log.warning(f"清理「{FRAME_CUTOFF_ID}」/「{GRASP_OBJECT_COLLISION_ID}」失败")

    def prompt_exit(message: str) -> bool:
        log.info(message)
        try:
            return input().strip().lower() == "q"
        except EOFError:
            return False

    def publish_attempt_result(result: bool, success_grasp_number: int) -> None:
        node.publish_grasp_cmd_result(result, success_grasp_number)

    def run_one_grasp() -> bool | None:
        """执行一轮抓取；True=成功，False=失败，None=用户选择退出。"""
        # --- 2. 关节空间运动 ---
        targets = JOINT_TARGETS["dual_arm"]
        joint_names = list(targets.keys())
        Q1 = [
            0.0, 0.0, 0.0, 30 * math.pi / 180,
            -1.57, -0.15, -1.578090, -1.370549, -1.672852, -0.588477,
            1.57, 0.15, 1.578090, 1.370549, 1.672852, 0.588477,
        ]
        waypoints = [Q1]  # 需要多点时：继续 waypoints.append(q3) ...

        # log.info(f"关节规划组: {"dual_arm"}")
        current = node._get_joints(joint_names)
        if current is None:
            log.error("读取当前关节位置失败，无法规划")
            return False
        log.info("规划前当前关节位置 [rad]:")
        log.info("  " + ", ".join(joint_names))
        log.info("  " + ", ".join(f"{current[name]:.6f}" for name in joint_names))

        if not node.plan_execute_joint_waypoints("dual_arm", 0.2, joint_names, waypoints):
            log.error("关节规划/执行失败")
            return False

        time.sleep(1)

        # 视觉识别
        vision_pose = read_vision_object_pose(node, log)
        if vision_pose is None:
            return False
        first_return_modes, all_xyz_rpy = vision_pose

        # 可达性验证：按视觉点顺序，先 left/right_arm，再 left/right_body。
        reachable = validate_reachable_grasp(node, all_xyz_rpy, speed_scale=0.2)
        if reachable is None:
            log.error("没有找到 IK 可解 + cartesian approach 可行的视觉点")
            return False

        point_index = reachable["point_index"]
        first_return_mode = first_return_modes[point_index]
        pick_group = reachable["pick_group"]
        pick_link = reachable["pick_link"]
        pick_frame = reachable["pick_frame"]
        pick_joint_names = reachable["pick_joint_names"]
        pick_target_pose = reachable["pick_target_pose"]
        pick_label = "左臂" if tool_side_for_link(pick_link) == "left" else "右臂"
        pick_q_target = reachable.get("pick_q_target", {})
        waist_pick_angle = (
            pick_q_target.get("body_joint2") if isinstance(pick_q_target, dict) else None
        )
        waist_moved = False

        if waist_pick_angle is not None and pick_group not in ("left_arm", "right_arm"):
            pick_side = reachable["pick_side"]
            log.info(
                f"可达性选中 {pick_group}，先用 body 组规划腰部抓取角度 "
                f"body_joint2={waist_pick_angle:.6f} rad "
                f"({math.degrees(waist_pick_angle):.2f} deg)"
            )
            if not node.move_body_joint2(waist_pick_angle, speed_scale=0.5):
                log.error("腰部运动到抓取角度失败")
                return False
            waist_moved = True
            log.info("腰部运动完成，重新视觉识别")

            vision_pose = read_vision_object_pose(node, log)
            if vision_pose is None:
                return False
            first_return_modes, all_xyz_rpy = vision_pose
            if point_index >= len(all_xyz_rpy) or point_index >= len(first_return_modes):
                log.error(
                    f"重新视觉识别后点数量不足，无法继续使用原点 {point_index + 1}"
                )
                return False

            xyz_key = pick_side
            new_xyz_rpy = all_xyz_rpy[point_index]
            if xyz_key not in new_xyz_rpy:
                log.error(f"重新视觉识别结果缺少 {xyz_key!r} 位姿，无法改用纯臂规划")
                return False

            pick_group = f"{pick_side}_arm"
            pick_link = "l_tool" if pick_side == "left" else "r_tool"
            pick_frame = "l_base_link" if pick_side == "left" else "r_base_link"
            pick_joint_names = joint_names_for_group(pick_group)
            pick_target_pose = xyz_rpy_to_pose(new_xyz_rpy[xyz_key])
            first_return_mode = first_return_modes[point_index]
            pick_label = "左臂" if pick_side == "left" else "右臂"
            log.info(
                f"腰部到位后改用纯臂规划: 点 {point_index + 1}/{len(all_xyz_rpy)}, "
                f"group={pick_group}, link={pick_link}, frame={pick_frame}"
            )

        selected_msg = (
            f"可达性选中: 点 {point_index + 1}/{len(all_xyz_rpy)}, "
            f"mode={first_return_mode}, group={pick_group}, "
            f"link={pick_link}, frame={pick_frame}"
        )
        log.info(selected_msg)
        print(f"\033[32m{selected_msg}\033[0m")

        node.show_cylinder_at_pose(pick_target_pose, frame_id=pick_frame)
        node.show_z_axis_at_pose(pick_target_pose, frame_id=pick_frame)
        if prompt_exit("按回车继续，输入 q 回车退出并移除深框 …"):
            return None

        log.info(f"开始尝试{pick_label}抓取")
        if not node.pick_and_return(
            target_pose=pick_target_pose,
            speed_scale=0.2,
            group=pick_group,
            link=pick_link,
            plan_frame=pick_frame,
            joint_names=pick_joint_names,
            place_joints=place_joints_for_group(pick_group),
            place_speed_scale=0.2,
            cutoff_joint_names=joint_names,
            first_return_mode=first_return_mode,
            waist_moved=waist_moved,
        ):
            log.error(f"{pick_label}抓取失败")
            return False

        return True

    try:
        if ACTIVE_GROUP not in JOINT_TARGETS:
            log.error(f"未知 ACTIVE_GROUP={ACTIVE_GROUP}，可选: {list(JOINT_TARGETS)}")
            return 1

        # 上位机通信
        # grasp_cmd = node.wait_for_grasp_start()
        # if grasp_cmd is None:
        #     return 1
    
        # --- 1. 添加深框 ---
        log.info(f"添加碰撞体「{FRAME_ID}」到 {SCENE_FRAME} …")
        if not node.add_frame():
            publish_attempt_result(False, 0)
            return 1
        frame_added = True

        if prompt_exit("按回车开始第一次抓取，输入 q 回车退出并移除深框 …"):
            code = 0
        else:
            while True:
                try:
                    grasp_result = run_one_grasp()
                finally:
                    cleanup_grasp_display()

                if grasp_result is None:
                    code = 0
                    break

                publish_attempt_result(grasp_result, 1 if grasp_result else 0)
                if grasp_result:
                    code = 0

                if prompt_exit("按回车继续下一次抓取，输入 q 回车退出 …"):
                    code = 0
                    break

    finally:
        cleanup_grasp_display(remove_scene_objects=frame_added)
        if frame_added:
            log.info(f"移除「{FRAME_ID}」…")
            if not node.remove_frame():
                code = 1
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return code


if __name__ == "__main__":
    sys.exit(main())
