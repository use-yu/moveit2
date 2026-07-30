#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 关节空间规划不到1s，给末端位姿4s左右
"""
G01 MoveIt 演示脚本

功能（按顺序执行）：
输入 1：先关节空间运动到 框_Q1，再发送 p,4 识别深框，
        按识别坐标重建深框障碍物并持续上料，直到 SW1~SW4 没有空位：
    机器人先到初始位
    → 读取视觉点
    → 生成左臂、右臂、SJ、base_link 下的目标位姿
    → 按视觉点和 group 顺序做可达性验证
    → 选中“有 IK + 直线 approach 可行 + 加入临时碰撞体后 q_pre 无碰撞”的候选
    → OMPL 到 q_pre
    → Cartesian approach 到抓取点
    → 工具上电抓取
    → 反向 Cartesian 回 q_pre
    → 根据 first_return_mode：
    ├─ mode=1：双臂交换 → 接物臂放置
    └─ mode=0/2：抓取臂直接放置
输入 2：执行原下料：
    下料读取最新 SW 信号，0=有料。
    优先取右臂 SW1 + 左臂 SW3，其次右臂 SW4 + 左臂 SW2；必须整对有料。
    发送 p,2 识别。
    添加 0.6×1.5×1.0m 取料台和 1.5×0.2×3.0m 障碍物。
    拼接对应 SW 的 yubei，腰部、升降均设为 0。
    双臂分别计算末端 +Z 10cm 的 IK/Cartesian 路径，合成一条 12 关节轨迹，一次执行。
    两个末端上电后，反向同一轨迹直线复位。
前提：
  - 已启动 move_group： ros2 launch g01_moveit_config demo.launch.py
  - 本节点与 move_group 在同一 ROS 域
  ros2 launch g01_moveit_config demo.launch.py use_real_hardware:=true

运行（Python）：
  colcon build --packages-select hello_moveit && source install/setup.bash
  ros2 run hello_moveit g01.py

腰部30度/0.523598 放置和交换

交换放置相比直接放绕z轴顺时针转1度

实际测试遇到的问题：
1. moveit关节空间规划时由于时间参数化（TOTG）有时候规划成功，执行失败
可以通过修改 ompl_planning.yaml 中的 longest_valid_segment_fraction 参数来解决，值越小，规划时间越长，但是规划成功率越高
本代码采用设置一个较好的初始构型，再加上失败重新规划方法来解决这个问题，相比改参数这样求解速度更快

2. 做末端直线运动时由于机械臂构型不同，实际有解但有时候会规划失败，
本代码采用求解多个逆解，再从多个逆解中选择一个直线规划可以求解成功的逆解，保证构型合理

3. moveit先plan再execute时，耗时较长
本代码采用先plan+execute的方法

4. 臂直接抓不可达
加上腰部关节求解，先ik验证逆解可达，再只用臂验证直线退回，直线退回的ik初始值用带腰部关节求解的值

"""
# solution_rad: r_arm_joint1=-1.944364063, r_arm_joint2=-1.597215489, r_arm_joint3=-0.565983840, r_arm_joint4=-0.979311147, r_arm_joint5=0.851759446, r_arm_joint6=3.140000000
# solution_deg: r_arm_joint1=-111.403854650, r_arm_joint2=-91.513706512, r_arm_joint3=-32.428485315, r_arm_joint4=-56.110395575, r_arm_joint5=48.802221443, r_arm_joint6=179.908747671
# solution_rad: l_arm_joint1=0.070556798, l_arm_joint2=-1.414919441, l_arm_joint3=-1.053385853, l_arm_joint4=-0.663103266, l_arm_joint5=0.585734181, l_arm_joint6=-0.010876821
# solution_deg: l_arm_joint1=4.042606739, l_arm_joint2=-81.068912306, l_arm_joint3=-60.354563592, l_arm_joint4=-37.993018506, l_arm_joint5=33.560096466, l_arm_joint6=-0.623195926
from __future__ import annotations

import bisect
import copy
import json
import math
import multiprocessing
import random
import re
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import Point, Pose, PoseStamped, TransformStamped
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
from moveit_msgs.srv import (
    ApplyPlanningScene,
    GetCartesianPath,
    GetPositionFK,
    GetPositionIK,
    GetStateValidity,
)
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import ColorRGBA, String, UInt8
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from trajectory_msgs.msg import JointTrajectoryPoint
from visualization_msgs.msg import Marker
from dobot_msgs_v4.srv import SetToolPower

# =============================================================================
# 用户可调参数（改这里即可，无需动下面逻辑）
# =============================================================================
GREEN = '\033[32m'
RESET = '\033[0m'
# 第一步：关节空间规划使用哪个 SRDF 组


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

    # 预备位：底盘略前伸、躯干抬起、双臂零位（与 hello_go1.cpp dual_arm 一致）
    "robot": {
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
    "dual_arm": {
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
    "dual_arm_body": {
        "body_joint1": 0.0,
        "body_joint2": 1.313,
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
    "dual_arm_waist": {
        "body_joint2": 1.313,
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
        "body_joint1": 0.0,
        "body_joint2": 1.313,
        "l_arm_joint1": 1.8697,
        "l_arm_joint2": 0.2,
        "l_arm_joint3": 0.135997,
        "l_arm_joint4": 1.23459,
        "l_arm_joint5": 2.1201,
        "l_arm_joint6": -1.5702,
    },
    "left_waist": {
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
    "left_arm": {
        "l_arm_joint1": 1.8697,
        "l_arm_joint2": 0.2,
        "l_arm_joint3": 0.135997,
        "l_arm_joint4": 1.23459,
        "l_arm_joint5": 2.1201,
        "l_arm_joint6": -1.5702,
    },
    "right": {
        "base_joint1": 1.25,
        "base_joint2": 0.0,
        "body_joint1": 0.0,
        "body_joint2": 1.313,
        "r_arm_joint1": -1.8697,
        "r_arm_joint2": 0.2,
        "r_arm_joint3": 0.135997,
        "r_arm_joint4": 1.23459,
        "r_arm_joint5": 2.1201,
        "r_arm_joint6": -1.5702,
    },
    "right_body": {
        "body_joint1": 0.0,
        "body_joint2": -1.313,
        "r_arm_joint1": -1.8697,
        "r_arm_joint2": 0.2,
        "r_arm_joint3": 0.135997,
        "r_arm_joint4": 1.23459,
        "r_arm_joint5": 2.1201,
        "r_arm_joint6": -1.5702,
    },
    "right_waist": {
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
    
}

# 位姿规划时作为起始状态的关节（不含底盘，与 left_body 组一致）
POSE_START_JOINTS = list(JOINT_TARGETS.get(POSE_GROUP, {}).keys())
if not POSE_START_JOINTS:
    raise KeyError(f"JOINT_TARGETS 中未找到 POSE_GROUP={POSE_GROUP} 的关节列表")

# p,4 深框识别专用预备构型，顺序与 dual_arm_body 一致。
框_Q1 = [
    0.13,
    50 * math.pi / 180,
    -1.57,
    -0.15,
    -1.578090,
    -1.370549,
    -1.672852,
    -0.588477,
    1.57,
    0.15,
    1.578090,
    1.370549,
    1.672852,
    0.588477,
]

# 每轮抓取使用的预备构型。
GRASP_Q1 = [
    0.0,
    30 * math.pi / 180,
    -1.57,
    -0.15,
    -1.578090,
    -1.370549,
    -1.672852,
    -0.588477,
    1.57,
    0.15,
    1.578090,
    1.370549,
    1.672852,
    0.588477,
]
for q1_name, q1_values in (("框_Q1", 框_Q1), ("GRASP_Q1", GRASP_Q1)):
    if len(q1_values) != len(JOINT_TARGETS["dual_arm_body"]):
        raise ValueError(
            f"{q1_name} 长度 {len(q1_values)} 与 dual_arm_body 关节数 "
            f"{len(JOINT_TARGETS['dual_arm_body'])} 不一致"
        )


# 规划器
PLANNER_ID = "RRTConnect"
NUM_ATTEMPTS = 20
PLAN_TIME_SEC = 10.0
MOVE_MAX_RETRIES = 10  # move_action 失败后再试几次（应对 INVALID_MOTION_PLAN 等 OMPL 随机性失败）

# 执行速度缩放（同时用于 velocity / acceleration；范围 0~1，越大越快）
DEFAULT_SPEED_SCALE = 0.5

# 深框障碍物（相对于 base_link 发布）
SCENE_FRAME = "base_link"
FRAME_ID = "深框"
FRAME_SIZE = (0.795, 0.795, 0.565)  # 深框整体外尺寸：长×宽×高 [m]
WALL_T = 0.035  # 壁厚向内部收缩，外轮廓尺寸保持 FRAME_SIZE
FRAME_CENTER = (0.92, 0.01, 0.4545)  # 深框整体外轮廓中心，相对于 base_link [m]
FRAME_RPY_DEG = (0.0, -0.0, 0.0)  # 深框整体姿态，相对于 base_link [degree]
FRAME_COLOR = ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.5)

# p,4 识别位姿先绕自身 Z 轴 +90°，再沿旋转后的自身坐标平移，
# 得到深框顶部空心区域中心；该坐标点位于开口中，不落在框壁实体上。
FRAME_VISION_TRIGGER_COMMAND = "p,4"
FRAME_VISION_POSE_KEY = "right_body"
FRAME_VISION_TF_FRAME = "deep_frame_vision"
FRAME_TOP_CENTER_TF_FRAME = "deep_frame_top_center"
FRAME_CENTER_TF_FRAME = "deep_frame_center"
FRAME_RECOGNITION_LOCAL_YAW = math.pi / 2.0
FRAME_RECOGNITION_TO_TOP_CENTER_LOCAL = (
    0.385,
    0.0,
    0.065,
)
# 长方体上表面中心相对深框顶部空心区域中心的局部平移。
BOX_OBSTACLE_ID = "长方体障碍物"
BOX_OBSTACLE_SIZE = (0.9, 1.0, 0.965)  # X × Y × Z [m]
BOX_OBSTACLE_TOP_FROM_FRAME_TOP_LOCAL = (0.0, 1.65, 0.4)
BOX_OBSTACLE_TOP_TF_FRAME = "box_obstacle_top_center"
BOX_OBSTACLE_COLOR = ColorRGBA(r=0.55, g=0.55, b=0.55, a=1.0)

FRAME_CUTOFF_ID = "深框隔离面"
FRAME_CUTOFF_THICKNESS = 0.01  # 水平隔离面厚度 [m]，沿深框局部 z 轴
FRAME_CUTOFF_BELOW_COLLISION = 0.01  # 隔离面上表面比 L6/R6 collision 最低点低 [m]
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
SVC_CHECK_STATE_VALIDITY = "check_state_validity"
LEFT_TOOL_COMMAND_SERVICE = "/g01/left/tool_commands"
RIGHT_TOOL_COMMAND_SERVICE = "/g01/right/tool_commands"
GRASP_CMD_TOPIC = "/grasp_cmd"
GRASP_CMD_RESULT_TOPIC = "/grasp_cmd_result"
DRIVER_SIGNAL_TOPIC = "/driver_report/signal"
SIGNAL_SW1_MASK = 0x08
SIGNAL_SW2_MASK = 0x10
SIGNAL_SW3_MASK = 0x20
SIGNAL_SW4_MASK = 0x40
PLACE_SLOT_MASKS = {
    "sw1": SIGNAL_SW1_MASK,
    "sw2": SIGNAL_SW2_MASK,
    "sw3": SIGNAL_SW3_MASK,
    "sw4": SIGNAL_SW4_MASK,
}
ALL_PLACE_SLOTS_EMPTY_SIGNAL = (
    SIGNAL_SW1_MASK | SIGNAL_SW2_MASK | SIGNAL_SW3_MASK | SIGNAL_SW4_MASK
)
ALL_PLACE_SLOTS_MATERIAL_SIGNAL = 0x00
DRIVER_SIGNAL_WAIT_TIMEOUT_SEC = 2.0

# 下料流程：双臂成对取料，优先右 SW1 + 左 SW3，其次右 SW4 + 左 SW2。
UNLOAD_TRIGGER_COMMAND = "p,2"
UNLOAD_VISION_POSE_KEY = "right_body"
UNLOAD_VISION_TF_FRAME = "material_table_vision"
UNLOAD_TABLE_TOP_TF_FRAME = "material_table_top"
UNLOAD_SLOT_PAIRS = (("sw1", "sw3"), ("sw4", "sw2"))  # (右臂 SW, 左臂 SW)
UNLOAD_TABLE_ID = "unload_table"
UNLOAD_TABLE_SIZE = (0.6, 1.7, 1.0)
UNLOAD_RECOGNITION_ABOVE_TABLE = -0.045
UNLOAD_TABLE_LOCAL_RPY = (0.0, math.pi, math.pi / 2.0)  # 局部 Y=180°，Z=90°
UNLOAD_TABLE_COLOR = ColorRGBA(r=0.48, g=0.30, b=0.14, a=0.85)
UNLOAD_TABLE_TOP_BOX_ID = "unload_table_top_box"
UNLOAD_TABLE_TOP_BOX_SIZE = (
    UNLOAD_TABLE_SIZE[0],
    UNLOAD_TABLE_SIZE[1],
    0.035,
)
UNLOAD_TABLE_TOP_BOX_COLOR = ColorRGBA(
    r=0.72,
    g=0.52,
    b=0.20,
    a=0.90,
)
UNLOAD_OBSTACLE_ID = "unload_vision_y_obstacle"
UNLOAD_OBSTACLE_SIZE = (1.5, 0.2, 3.0)
UNLOAD_OBSTACLE_Y_OFFSET = 1.2
UNLOAD_OBSTACLE_COLOR = ColorRGBA(r=0.55, g=0.55, b=0.55, a=0.85)
UNLOAD_APPROACH_DISTANCE = 0.10
UNLOAD_PLACE_DESCENT_DISTANCE = 0.10
# 上料放置位抓取
UNLOAD_EXTRA_APPROACH_BY_SLOT = {
    "sw1": 0.003,  # 右臂：多降 4 mm
    "sw2": 0.001,  # 左臂：多降 1 mm
    "sw3": 0.003,  # 左臂：多降 4 mm
    "sw4": 0.005,  # 右臂：多降 8 mm
}
# 物料台放置点：在 material_table_top 桌子坐标系下的 xyz 偏移 [m]。
UNLOAD_PLACE_LOCAL_YAW = math.radians(204.5)
UNLOAD_PLACE_LOCAL_OFFSETS = {
    1: (0.1 + 0.002782, 0.325-0.338809+0.347293, -0.16 + 1.115998 - 1.070837-0.01),
    2: (0.1 + 0.653630-0.656605+0.005, 0.125-0.138855+0.146165-0.001, -0.16+1.206160-1.071223-0.1),
    3: (0.1 + 0.643439-0.655317+0.012, -0.125+0.110927-0.111151-0.001, -0.16+1.203874-1.073237-0.1),
    4: (0.1 + 0.644717-0.649813, -0.325+0.310772-0.311753, -0.16+1.109917-1.074118-0.01),
}
UNLOAD_PLACE_SEQUENCE = ((1, 4), (2, 3))  # 每轮 (左臂点位, 右臂点位)
UNLOAD_JOINT_SPEED = 0.2
UNLOAD_PLACE_JOINT_SPEED = 0.2
UNLOAD_PLACE_ARM_IK_ATTEMPTS = 200
UNLOAD_PLACE_ARM_IK_MAX_SOLUTIONS = 20
UNLOAD_PLACE_BODY_IK_ATTEMPTS = 200
UNLOAD_PLACE_BODY_IK_MAX_SOLUTIONS = 20
UNLOAD_PLACE_RIGHT_IK_ATTEMPTS_PER_BODY = 40
UNLOAD_PLACE_RIGHT_IK_MAX_SOLUTIONS_PER_BODY = 10
UNLOAD_PLACE_MAX_PAIR_PLANS = 200
UNLOAD_PLACE_IK_PERTURB = math.pi
UNLOAD_PLACE_IK_RANDOM_SEED = 20260724
# IK 随机种子使用的 URDF 位置限位。抓取与下料共用，避免生成越界 RobotState。
BODY_JOINT1_LIMITS = (0.0, 0.19)
BODY_JOINT2_LIMITS = (0.0, 1.0)
ARM_JOINT_LIMITS_BY_INDEX = {
    1: (-3.14, 3.8),
    2: (-3.14, 3.14),
    3: (-2.79, 2.79),
    4: (-3.14, 3.14),
    5: (-3.14, 3.14),
    6: (-3.14, 3.14),
}
UNLOAD_CARTESIAN_SPEED = 0.2
UNLOAD_CARTESIAN_AVOID_COLLISIONS = False
UNLOAD_SYNC_SAMPLE_PERIOD = 0.005
UNLOAD_TOOL_SETTLE_SEC = 1.0

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
DEFAULT_ROBOT_URDF = (
    Path(__file__).resolve().parents[2]
    / "moveit_resources/g01_description/urdf/G01-URDF888.urdf"
)

# 放置位关节目标 [rad]。fang 字典的顺序就是放置顺序，槽位按左右臂分别记录。
PLACE_JOINTS = {
    # "left_body": [
    #     -0.01292, 1.015203, -0.712975, -0.550402, 1.300752, 0.543868, -0.143126, -0.338787
    # ],
    # "left": [
    #     1.25, 0.0, -0.01292, 1.015203, -0.712975, -0.550402, 1.300752, 0.543868, -0.143126, -0.338787
    # ],
    # "right_body": [
    #     -0.01292, 1.015203, -0.712975, -0.550402, 1.300752, 0.543868, -0.143126, -0.338787
    # ],
    # "right": [
    #     1.25, 0.0, -0.01292, 1.015203, -0.712975, -0.550402, 1.300752, 0.543868, -0.143126, -0.338787
    # ],
    # "dual_arm": [
    #     1.25, 0.0, -0.01292, 1.015203, -0.712975, -0.550402, 1.300752, 0.543868, -0.143126, -0.338787
    #     ,3.13, -1.419584, 1.578090, 1.370549, 1.672852, 0.588477
    # ],
    "right_arm": {
        "yubei_j": {
            "sw1": [
                0.0, 0*math.pi/180,
                3.552814732,1.067102581,2.033169521,0.056248203,0.421488379,-2.317843286
            ],
            "sw2": [
                0.0, 0*math.pi/180,
                3.550977687,1.543371705,1.710112869,-0.092421272,0.416408188,-2.312775084
            ],
            "sw3": [
                0.0, 0*math.pi/180,
                3.413621317,1.587298827,1.201184044,0.372457647,0.278332061,-2.311537131
            ],
            "sw4": [
                0.0, 0*math.pi/180,
                3.414030102,1.160954858,1.517017010,0.478336433,0.281375128,-2.316729426
            ],

        },
        "fang_j": {
            "sw1": [
                3.779784309,1.104560904,1.744916198,0.298812597,0.647904917,-2.310057784
            ],
            "sw2": [
                3.778844881,1.547545233,1.437285734,0.165709989,0.644147624,-2.312850674
            ],
            "sw3": [
                3.597396272,1.670758464,0.888395467,0.590636173,0.460773818,-2.310696428
            ],
            "sw4": [
                3.597224496,1.248114698,1.239936572,0.660243743,0.463286113,-2.310028515
            ],
        },
        "yubei": {
            "sw1": [
                0.0, 0*math.pi/180,
                3.552814732,1.067102581,2.033169521,0.056248203,0.421488379,-2.337843286
            ],
            "sw2": [
                0.0, 0*math.pi/180,
                3.550977687,1.543371705,1.710112869,-0.092421272,0.416408188,-2.342775084
            ],
            "sw3": [
                0.0, 0*math.pi/180,
                3.413621317,1.587298827,1.201184044,0.372457647,0.278332061,-2.341537131
            ],
            "sw4": [
                0.0, 0*math.pi/180,
                3.414030102,1.160954858,1.517017010,0.478336433,0.281375128,-2.336729426
            ],

        },
        "fang": {
            "sw1": [
                3.779784309,1.104560904,1.744916198,0.298812597,0.647904917,-2.330057784
            ],
            "sw2": [
                3.778844881,1.547545233,1.437285734,0.165709989,0.644147624,-2.332850674
            ],
            "sw3": [
                3.597396272,1.670758464,0.888395467,0.590636173,0.460773818,-2.330696428
            ],
            "sw4": [
                3.597224496,1.248114698,1.239936572,0.660243743,0.463286113,-2.329028515
            ],
        },
    },
    "left_arm": {
        "yubei_j": {
            "sw1": [
                0.0, 0*math.pi/180,
                2.742146020,-1.534454859,-1.704132965,0.066555922,-0.408860117,0.887351503
            ],
            "sw2": [
                0.0, 0*math.pi/180,
                2.740184794,-1.060367820,-2.021640662,-0.083783500,-0.414819447,0.88060267
            ],
            "sw3": [
                0.0, 0*math.pi/180,
                2.874734517,-1.158993236,-1.498075078,-0.509897863,-0.278424737,0.898361450
            ],
            "sw4": [
                0.0, 0*math.pi/180,
                2.874999193,-1.583481445,-1.185773517,-0.403933549,-0.274935549,0.887360005
            ],

        },
        "fang_j": {
            "sw1": [
                2.514280207,-1.540982362,-1.433430928,-0.182283060,-0.636504865,0.872310002
            ],
            "sw2": [
                2.513366944,-1.100374679,-1.736173236,-0.317114736,-0.640877592,0.86856996
            ],
            "sw3": [
                2.692263525,-1.247297530,-1.222955195,-0.684074186,-0.459222663,0.885742410
            ],
            "sw4": [
                2.691832494,-1.668514988,-0.874431034,-0.613384141,-0.456350585,0.870520441
            ],
        },
        "yubei": {
            "sw1": [
                0.0, 0*math.pi/180,
                2.742154533,-1.538445015,-1.700329092,0.066732123,-0.408816014,0.835002272
            ],
            "sw2": [
                0.0, 0*math.pi/180,
                2.740184794,-1.060367820,-2.021640662,-0.083783500,-0.414819447,0.828242796
            ],
            "sw3": [
                0.0, 0*math.pi/180,
                2.874734517,-1.158993236,-1.498075078,-0.509897863,-0.278424737,0.837274926
            ],
            "sw4": [
                0.0, 0*math.pi/180,
                2.874990700,-1.588803345,-1.180363673,-0.404020694,-0.274896344,0.834999291
            ],

        },
        "fang": {
            "sw1": [
                2.514280179,-1.544936097,-1.429629445,-0.182134808,-0.636469956,0.819954936
            ],
            "sw2": [
                2.513366937,-1.100374674,-1.736173231,-0.317114734,-0.640877599,0.816210074
            ],
            "sw3": [
                2.692263525,-1.247297537,-1.222955166,-0.684074206,-0.459222663,0.824655886
            ],
            "sw4": [
                2.691811998,-1.674210877,-0.867914402,-0.614182914,-0.456316350,0.818136293
            ],
        },
    },
}

# 抓取流程默认参数
PRE_GRASP_OFFSET = -0.1  # 预备抓取点沿末端坐标系 z 轴外移的距离 [m]
PLACE_SPEED_SCALE = 0.5     # 5/9~9/9 放置与返回的速度缩放
FIRST_RETURN_MODE = 2       # 1: 只反向直线回到 q_pre；2: 直线+OMPL 回到 1/9 初始位置
EXCHANGE_Q1 = [
    -1.57, -0.15, -1.578090, -1.370549, -1.672852, -0.588477,
    1.57, 0.15, 1.578090, 1.370549, 1.672852, 0.588477,
]
# 交换放置
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
# 交换
EXCHANGE_Q2 = {
    "right": [
        0.0, 30*math.pi/180,
        0.162671941,-1.605106236,-0.766205809,-0.762374542,0.680110627,-0.532089222,
        -1.850076372,-1.576109233,-0.618642353,-0.947968147,0.771840083,0.000991018,
    ],
    "left": [
        0.0, 30*math.pi/180,
        1.850076372,1.576109233,0.618642353,0.947968147,-0.771840083,-0.000991018,
        -0.162671941,1.605106236,0.766205809,0.762374542,-0.680110627,-0.532089222,
        
    ],
}

# 视觉 TCP 配置
VISION_IP = "192.168.1.110"
VISION_PORT = 50000
VISION_TRIGGER_COMMAND = "p,1"
VISION_CONNECT_TIMEOUT = 3.0
VISION_RECV_TIMEOUT = 30.0
# 正则表达式，用来匹配字符串里的数字：
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
# 标定输入：x, y, z 单位米，四元数顺序为 w, x, y, z。
# 下面会转成平移单位为毫米的 4x4 矩阵，与 viewer pose 的毫米单位保持一致。
VISION_RIGHT_TRANSFORM_XYZ_WXYZ = [
    -0.150665, -0.147699, -0.137334, 0.654304, -0.254175, -0.659826, -0.268161
]
VISION_LEFT_TRANSFORM_XYZ_WXYZ = [
    0.152736, -0.146261, -0.214731, 0.657772, -0.272723, 0.653477, 0.256760
]
# 左臂抓
SIM_VISION_RESULT = (
    0,
    [-195.0305, 43.2781, 889.8122, -0.481, 0.0739, -0.1165, -0.8658],
)
# 右臂抓
# SIM_VISION_RESULT = (
#     1,
#     [-25.0305, 43.2781, 789.8122, -0.481, 0.0739, -0.1165, -0.8658],
# )
# 物料台 p,2 仿真视觉数据：
# 原始协议为 1,x,y,z,qw,qx,qy,qz,mode；首个 1 是点数量，末尾 1 是模式码。
SIM_UNLOAD_VISION_RESULT = (
    1,
    [-47.2306, -52.774, 621.5524, 0.3712, 0.9282, 0.0238, -0.0043],
)
# p,4 仿真时沿用同一套视觉协议；框位姿仍按旋转和平移链计算。
SIM_FRAME_VISION_RESULT = SIM_UNLOAD_VISION_RESULT


def sim_vision_result_for_trigger(
    trigger_command: str,
) -> tuple[int, list[float]]:
    """按视觉触发命令选择对应的仿真数据。"""
    if trigger_command == UNLOAD_TRIGGER_COMMAND:
        return SIM_UNLOAD_VISION_RESULT
    if trigger_command == FRAME_VISION_TRIGGER_COMMAND:
        return SIM_FRAME_VISION_RESULT
    return SIM_VISION_RESULT

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


def read_vision_pose(
    sock: socket.socket,
    log,
    trigger_command: str = VISION_TRIGGER_COMMAND,
) -> list[tuple[int, list[float]]] | None:
    """解析视觉数据：第 1 个数忽略，后面每 8 个数为 xyz(mm)+quat(wxyz)+模式码。"""
    try:
        t0 = time.monotonic()
        sock.sendall(trigger_command.encode("utf-8"))
        log.info(f"viewer 已发送触发命令: {trigger_command}")
        raw_text = sock.recv(4096).decode("utf-8", errors="ignore").strip()
        used_ms = (time.monotonic() - t0) * 1000.0
        message = f"viewer 发送命令到接收到数字耗时: {used_ms:.3f} ms"
        log.info(message)
        print(f"{GREEN}{message}{RESET}")
        print(f"{GREEN}viewer 接收到的数据: {raw_text}{RESET}")

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


def read_vision_object_pose(
    node,
    log,
    sim_mode: bool = False,
    trigger_command: str = VISION_TRIGGER_COMMAND,
):
    """视觉识别封装：返回所有点的模式码和 xyz_rpy。"""
    vision_sock = None
    if sim_mode:
        first_return_mode, pose = sim_vision_result_for_trigger(trigger_command)
        vision_results = [(first_return_mode, list(pose))]
        if trigger_command == UNLOAD_TRIGGER_COMMAND:
            sim_source = "物料台 p,2"
        elif trigger_command == FRAME_VISION_TRIGGER_COMMAND:
            sim_source = "深框 p,4"
        else:
            sim_source = "普通抓取"
        log.info(f"[sim] viewer 点个数: 1，使用{sim_source}固定 pose")
        print(
            f"{GREEN}[sim] viewer point 1: "
            f"first_return_mode = {first_return_mode}, pose = {pose}{RESET}"
        )
    else:
        vision_sock = connect_vision(log)
        if vision_sock is None:
            return None

    try:
        if not sim_mode:
            vision_results = read_vision_pose(
                vision_sock,
                log,
                trigger_command=trigger_command,
            )
            if vision_results is None:
                return None

        # 读取实际身体关节：
        #   body_joint2 用于把左/右臂基坐标下的物体位姿转到 SJ；
        #   body_joint1 用于继续把 SJ 下的位姿转到 base_link。
        body_joints = node._get_joints(["body_joint1", "body_joint2"], wait_new=True)
        if body_joints is None:
            log.error("读取实际身体关节失败，无法转换视觉位姿")
            return None
        print(
            f"实际身体关节: body_joint1={body_joints['body_joint1']:.6f} m, "
            f"body_joint2={body_joints['body_joint2']:.6f} rad"
        )

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
        sj_in_base = node._get_link_pose_fk(
            "SJ",
            joints=body_joints,
            plan_frame="base_link",
        )
        if sj_in_base is None:
            log.error("计算 SJ → base_link 变换失败")
            return None

        t_sj_r_base = pose_to_matrix(r_base_in_sj)
        t_sj_l_base = pose_to_matrix(l_base_in_sj)
        t_base_sj = pose_to_matrix(sj_in_base)
        first_return_modes: list[int] = []
        all_xyz_rpy: list[dict[str, tuple[float, float, float, float, float, float]]] = []
        raw_poses: list[list[float]] = []

        for point_index, (first_return_mode, pose) in enumerate(vision_results, start=1):
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
            right_body_matrix = matmul4(t_base_sj, right_sj_matrix)
            left_body_matrix = matmul4(t_base_sj, left_sj_matrix)
            xyz_rpy = {
                "right": matrix_to_xyz_rpy(right_matrix),
                "left": matrix_to_xyz_rpy(left_matrix),
                "right_sj": matrix_to_xyz_rpy(right_sj_matrix),
                "left_sj": matrix_to_xyz_rpy(left_sj_matrix),
                "right_body": matrix_to_xyz_rpy(right_body_matrix),
                "left_body": matrix_to_xyz_rpy(left_body_matrix),
                "sj": matrix_to_xyz_rpy(right_sj_matrix),
            }

            first_return_modes.append(first_return_mode)
            all_xyz_rpy.append(xyz_rpy)
            raw_poses.append(pose)

        sorted_results = sorted(
            zip(first_return_modes, all_xyz_rpy, raw_poses),
            key=lambda item: item[1]["sj"][2],
            reverse=True,
        )
        first_return_modes = [mode for mode, _, _ in sorted_results]
        all_xyz_rpy = [xyz_rpy for _, xyz_rpy, _ in sorted_results]

        for point_index, (first_return_mode, xyz_rpy, pose) in enumerate(sorted_results, start=1):
            print(
                f"\033[32mviewer point {point_index}: "
                f"first_return_mode = {first_return_mode}, pose = {pose}\033[0m"
            )
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


def ik_seed_limits_for_joint(joint_name: str) -> tuple[float, float]:
    """返回 IK 随机种子使用的 URDF 位置限位。"""
    if joint_name == "body_joint1":
        return BODY_JOINT1_LIMITS
    if joint_name == "body_joint2":
        return BODY_JOINT2_LIMITS

    arm_match = re.fullmatch(r"[lr]_arm_joint([1-6])", joint_name)
    if arm_match:
        return ARM_JOINT_LIMITS_BY_INDEX[int(arm_match.group(1))]

    raise KeyError(f"没有配置关节 {joint_name!r} 的 IK 随机种子限位")


def make_unload_yubei_joint_target(
    right_slot: str,
    left_slot: str,
) -> tuple[list[str], list[float]]:
    """把左右 SW 的 yubei 纯臂关节拼成 body_joint1/2 均为 0 的 dual_arm_body 目标。"""
    right_config = PLACE_JOINTS.get("right_arm", {}).get("yubei")
    left_config = PLACE_JOINTS.get("left_arm", {}).get("yubei")
    if not isinstance(right_config, dict) or right_slot not in right_config:
        raise KeyError(f"右臂 yubei 未配置 {right_slot}")
    if not isinstance(left_config, dict) or left_slot not in left_config:
        raise KeyError(f"左臂 yubei 未配置 {left_slot}")

    right_values = list(right_config[right_slot])
    left_values = list(left_config[left_slot])
    if len(right_values) != 8 or len(left_values) != 8:
        raise ValueError(
            f"下料 yubei 配置长度必须为 8: "
            f"right.{right_slot}={len(right_values)}, "
            f"left.{left_slot}={len(left_values)}"
        )

    left_joint_names = joint_names_for_group("left_arm")
    right_joint_names = joint_names_for_group("right_arm")
    dual_body_joint_names = joint_names_for_group("dual_arm_body")
    target = {
        "body_joint1": 0.0,
        "body_joint2": 0.0,
        **dict(zip(left_joint_names, left_values[-6:])),
        **dict(zip(right_joint_names, right_values[-6:])),
    }
    missing = [name for name in dual_body_joint_names if name not in target]
    if missing:
        raise ValueError(f"dual_arm_body 下料目标缺少关节: {missing}")
    return dual_body_joint_names, [target[name] for name in dual_body_joint_names]


def _side_order_from_xyz_rpy(xyz_rpy: dict) -> tuple[str, str]:
    """根据物体在 SJ 下的 y 值决定左右优先级：y > 0 时左臂优先。"""
    return ("left", "right") if xyz_rpy["sj"][1] > 0.0 else ("right", "left")


def _reachability_attempts_for_point(xyz_rpy: dict) -> list[dict[str, str]]:
    """依次验证纯臂、腰+臂、身体+臂，每层均按物体侧向决定左右顺序。"""
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
            "group": f"{side}_waist",
            "link": "l_tool" if side == "left" else "r_tool",
            "plan_frame": "SJ",
            "xyz_key": f"{side}_sj",
        })
    for side in side_order:
        attempts.append({
            "side": side,
            "group": f"{side}_body",
            "link": "l_tool" if side == "left" else "r_tool",
            "plan_frame": "base_link",
            "xyz_key": f"{side}_body",
        })
    return attempts


def arm_context_for_group(group: str) -> tuple[str, str] | None:
    """抓取 group 对应的纯臂 group 和纯臂规划坐标系。"""
    if group in ("left_arm", "left_waist", "left_body"):
        return "left_arm", "l_base_link"
    if group in ("right_arm", "right_waist", "right_body"):
        return "right_arm", "r_base_link"
    return None


# 按视觉点顺序遍历：
#   点1 → 点2 → 点3 ...

# 每个点内按 side_value = xyz_rpy["sj"][1] 判断左右顺序：

# side_value > 0:
#   left_arm → right_arm → left_waist → right_waist → left_body → right_body
# side_value <= 0:
#   right_arm → left_arm → right_waist → left_waist → right_body → left_body
def validate_reachable_grasp(
    node,
    all_xyz_rpy: list[dict],
    speed_scale: float = 0.2,
    cutoff_joint_names: Sequence[str] | None = None,
):
    """验证 IK、Cartesian approach 以及加入两个临时碰撞体后的 q_pre。"""
    log = node.get_logger()
    if not all_xyz_rpy:
        log.error("[reach] 没有可验证的视觉点")
        return None

    for point_index, xyz_rpy in enumerate(all_xyz_rpy):
        attempts = _reachability_attempts_for_point(xyz_rpy)
        order_text = " → ".join(item["group"] for item in attempts)
        log.info(
            f"[reach] 点 {point_index + 1}/{len(all_xyz_rpy)}: "
            f"SJ.y={xyz_rpy['sj'][1]:.6f}, 验证顺序 {order_text}"
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
            picked = node._select_feasible_grasp_pair(
                pick_group,
                pick_link,
                pick_target_pose,
                pre_pose,
                joint_names=pick_joint_names,
                speed_scale=speed_scale,
                plan_frame=pick_frame,
                cutoff_joint_names=cutoff_joint_names,
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


def pose_offset_local(pose: Pose, dx: float, dy: float, dz: float) -> Pose:
    """按 pose 自身坐标系平移 xyz，姿态保持不变。"""
    qx, qy, qz, qw = (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    offset_x, offset_y, offset_z = rotate_xyz_by_quat(
        dx, dy, dz, qx, qy, qz, qw
    )
    out = copy.deepcopy(pose)
    out.position.x += offset_x
    out.position.y += offset_y
    out.position.z += offset_z
    return out


def deep_frame_poses_from_recognition(
    recognition_pose: Pose,
) -> tuple[Pose, Pose]:
    """返回（深框顶部空心中心，深框实体中心）两个位姿。"""
    rotated_pose = pose_rotate_local_rpy(
        recognition_pose,
        0.0,
        0.0,
        FRAME_RECOGNITION_LOCAL_YAW,
    )
    top_center_pose = pose_offset_local(
        rotated_pose,
        *FRAME_RECOGNITION_TO_TOP_CENTER_LOCAL,
    )
    frame_center_pose = pose_offset_local(
        top_center_pose,
        0.0,
        0.0,
        -FRAME_SIZE[2] / 2.0,
    )
    return top_center_pose, frame_center_pose


def box_obstacle_top_pose_from_frame_top(frame_top_pose: Pose) -> Pose:
    """由深框顶部空心中心计算长方体障碍物的上表面中心。"""
    return pose_offset_local(
        frame_top_pose,
        *BOX_OBSTACLE_TOP_FROM_FRAME_TOP_LOCAL,
    )


def pose_rotate_local_rpy(
    pose: Pose,
    roll: float,
    pitch: float,
    yaw: float,
) -> Pose:
    """姿态右乘局部 RPY 旋转，位置保持不变。"""
    ox, oy, oz, ow = quat_from_rpy(roll, pitch, yaw)
    q = pose.orientation
    out = copy.deepcopy(pose)
    out.orientation.x = q.w * ox + q.x * ow + q.y * oz - q.z * oy
    out.orientation.y = q.w * oy - q.x * oz + q.y * ow + q.z * ox
    out.orientation.z = q.w * oz + q.x * oy - q.y * ox + q.z * ow
    out.orientation.w = q.w * ow - q.x * ox - q.y * oy - q.z * oz
    norm = math.sqrt(
        out.orientation.x * out.orientation.x
        + out.orientation.y * out.orientation.y
        + out.orientation.z * out.orientation.z
        + out.orientation.w * out.orientation.w
    )
    if norm <= 1e-12:
        raise ValueError("局部旋转后的四元数长度为 0")
    out.orientation.x /= norm
    out.orientation.y /= norm
    out.orientation.z /= norm
    out.orientation.w /= norm
    return out


def pose_relative_to_frame(
    pose_in_reference: Pose,
    frame_in_reference: Pose,
) -> Pose:
    """把 reference 下的目标位姿转换成指定 frame 下的表达。"""
    relative_matrix = matmul4(
        invert_transform4(pose_to_matrix(frame_in_reference)),
        pose_to_matrix(pose_in_reference),
    )
    return xyz_rpy_to_pose(matrix_to_xyz_rpy(relative_matrix))


def joint_distance_squared(
    candidate: dict[str, float],
    reference: dict[str, float],
    joint_names: Sequence[str],
) -> float:
    """用于按接近当前构型的程度排序 IK 解。"""
    return sum(
        (candidate[name] - reference[name]) ** 2
        for name in joint_names
    )


def spread_sorted_candidates(
    candidates: Sequence,
    max_count: int,
) -> list:
    """从已排序候选中均匀取样，兼顾近、中、远构型。"""
    candidates = list(candidates)
    max_count = max(0, int(max_count))
    if max_count == 0:
        return []
    if len(candidates) <= max_count:
        return candidates
    if max_count == 1:
        return [candidates[0]]
    last_index = len(candidates) - 1
    return [
        candidates[round(index * last_index / (max_count - 1))]
        for index in range(max_count)
    ]


def pose_offset_local_z(pose: Pose, dz: float) -> Pose:
    """沿 pose 自身坐标系 z 轴平移 dz 米，姿态保持不变。"""
    return pose_offset_local(pose, 0.0, 0.0, dz)


def make_deep_frame(frame_pose: Pose | None = None) -> CollisionObject:
    """
    深框 = 1 块底板 + 4 块侧墙（BOX  primitive），顶部无盖。
    FRAME_SIZE 为深框整体外尺寸；frame_pose 为外轮廓中心位姿。
    frame_pose=None 时使用静态 FRAME_CENTER / FRAME_RPY_DEG。
    WALL_T 只向内部收缩，外轮廓保持 L × W × H。
    """
    L, W, H = FRAME_SIZE
    t = WALL_T
    if frame_pose is None:
        roll, pitch, yaw = (math.radians(v) for v in FRAME_RPY_DEG)
        frame_pose = make_pose(*FRAME_CENTER, roll, pitch, yaw)

    obj = CollisionObject()
    obj.header.frame_id = SCENE_FRAME
    obj.id = FRAME_ID
    obj.operation = CollisionObject.ADD

    def add_box(dx, dy, dz, ox, oy, oz):
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [dx, dy, dz]
        obj.primitives.append(prim)
        obj.primitive_poses.append(
            pose_offset_local(frame_pose, ox, oy, oz)
        )

    wall_h = H - t
    wall_z = -H / 2 + t + wall_h / 2
    add_box(L, W, t, 0, 0, -H / 2 + t / 2)                    # 底板
    add_box(t, W, wall_h, L / 2 - t / 2, 0, wall_z)           # +X 侧墙
    add_box(t, W, wall_h, -(L / 2 - t / 2), 0, wall_z)        # -X 侧墙
    add_box(L - 2 * t, t, wall_h, 0, W / 2 - t / 2, wall_z)   # +Y 侧墙
    add_box(L - 2 * t, t, wall_h, 0, -(W / 2 - t / 2), wall_z)  # -Y 侧墙
    return obj


def make_box_obstacle(box_top_pose: Pose) -> CollisionObject:
    """由长方体上表面中心向自身 -Z 下移半高，创建实体碰撞体。"""
    obj = CollisionObject()
    obj.header.frame_id = SCENE_FRAME
    obj.id = BOX_OBSTACLE_ID
    obj.operation = CollisionObject.ADD

    prim = SolidPrimitive()
    prim.type = SolidPrimitive.BOX
    prim.dimensions = list(BOX_OBSTACLE_SIZE)
    obj.primitives.append(prim)
    obj.primitive_poses.append(
        pose_offset_local(
            box_top_pose,
            0.0,
            0.0,
            -BOX_OBSTACLE_SIZE[2] / 2.0,
        )
    )
    return obj


def make_unload_table_top_pose(recognition_pose: Pose) -> Pose:
    """桌面中心沿视觉 -Z 偏移，姿态再相对视觉绕局部 Y=180°、Z=90°。"""
    translated = pose_offset_local_z(
        recognition_pose,
        -UNLOAD_RECOGNITION_ABOVE_TABLE,
    )
    return pose_rotate_local_rpy(translated, *UNLOAD_TABLE_LOCAL_RPY)


def make_unload_table(recognition_pose: Pose) -> CollisionObject:
    """生成相对 p,2 视觉姿态绕局部 Y=180°、Z=90° 的取料台。"""
    table_top_pose = make_unload_table_top_pose(recognition_pose)
    center_pose = pose_offset_local_z(
        table_top_pose,
        UNLOAD_TABLE_SIZE[2] / 2.0,
    )

    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = list(UNLOAD_TABLE_SIZE)

    obj = CollisionObject()
    obj.header.frame_id = SCENE_FRAME
    obj.id = UNLOAD_TABLE_ID
    obj.operation = CollisionObject.ADD
    obj.primitives.append(primitive)
    obj.primitive_poses.append(center_pose)
    return obj


def make_unload_table_top_box(recognition_pose: Pose) -> CollisionObject:
    """生成薄长方体；下表面中心与视觉识别位置重合，姿态与料台一致。"""
    bottom_pose = pose_rotate_local_rpy(
        recognition_pose,
        *UNLOAD_TABLE_LOCAL_RPY,
    )
    center_pose = pose_offset_local_z(
        bottom_pose,
        -UNLOAD_TABLE_TOP_BOX_SIZE[2] / 2.0,
    )

    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = list(UNLOAD_TABLE_TOP_BOX_SIZE)

    obj = CollisionObject()
    obj.header.frame_id = SCENE_FRAME
    obj.id = UNLOAD_TABLE_TOP_BOX_ID
    obj.operation = CollisionObject.ADD
    obj.primitives.append(primitive)
    obj.primitive_poses.append(center_pose)
    return obj


def make_unload_obstacle(recognition_pose: Pose) -> CollisionObject:
    """在识别点的 base_link +Y 方向 1.2 m 处生成 3 m 高墙状障碍物。"""
    center_pose = make_pose(
        recognition_pose.position.x,
        recognition_pose.position.y + UNLOAD_OBSTACLE_Y_OFFSET,
        UNLOAD_OBSTACLE_SIZE[2] / 2.0,
    )

    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = list(UNLOAD_OBSTACLE_SIZE)

    obj = CollisionObject()
    obj.header.frame_id = SCENE_FRAME
    obj.id = UNLOAD_OBSTACLE_ID
    obj.operation = CollisionObject.ADD
    obj.primitives.append(primitive)
    obj.primitive_poses.append(center_pose)
    return obj


def make_unload_place_poses(recognition_pose: Pose) -> dict[int, Pose]:
    """以桌面为基准生成四个点位，姿态相对桌子绕局部 Z 轴 206°。"""
    table_top_pose = make_unload_table_top_pose(recognition_pose)
    return {
        point_index: pose_rotate_local_rpy(
            pose_offset_local(table_top_pose, *local_offset),
            0.0,
            0.0,
            UNLOAD_PLACE_LOCAL_YAW,
        )
        for point_index, local_offset in UNLOAD_PLACE_LOCAL_OFFSETS.items()
    }


def make_deep_frame_cutoff(
    scene_z: float,
    frame_pose: Pose | None = None,
) -> CollisionObject:
    """在深框内生成水平隔离面，其上表面位于 SCENE_FRAME 的 scene_z。"""
    L, W, _ = FRAME_SIZE
    t = WALL_T
    if frame_pose is None:
        roll, pitch, yaw = (math.radians(v) for v in FRAME_RPY_DEG)
        frame_pose = make_pose(*FRAME_CENTER, roll, pitch, yaw)
    q = frame_pose.orientation

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
    # 隔离面沿深框局部 X/Y 铺满内腔，局部 Z 为厚度方向。
    # scene_z 是隔离面上表面高度，实体中心向下偏半个厚度。
    ox, oy, oz = rotate_xyz_by_quat(
        0.0,
        0.0,
        -FRAME_CUTOFF_THICKNESS / 2.0,
        q.x,
        q.y,
        q.z,
        q.w,
    )
    obj.primitives.append(prim)
    cutoff_pose = copy.deepcopy(frame_pose)
    cutoff_pose.position.x += ox
    cutoff_pose.position.y += oy
    cutoff_pose.position.z = scene_z + oz
    obj.primitive_poses.append(cutoff_pose)
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


def cylinder_half_extent_z(pose: Pose) -> float:
    """圆柱按当前姿态投影到 Z 轴后的半尺寸，用于让水平隔板与圆柱相切。"""
    return cylinder_half_extent_z_from_shape(pose, CYLINDER_DIAMETER / 2.0, CYLINDER_HEIGHT)


def cylinder_half_extent_z_from_shape(pose: Pose, radius: float, length: float) -> float:
    """指定半径/长度的圆柱按当前姿态投影到 Z 轴后的半尺寸。"""
    q = pose.orientation
    _, _, axis_z = rotate_xyz_by_quat(0.0, 0.0, 1.0, q.x, q.y, q.z, q.w)
    axis_z = max(-1.0, min(1.0, axis_z))
    half_height = float(length) / 2.0
    radial_projection = math.sqrt(max(0.0, 1.0 - axis_z * axis_z))
    return abs(axis_z) * half_height + radial_projection * float(radius)


def matrix_from_xyz_rpy(xyz: Sequence[float], rpy: Sequence[float]) -> list[list[float]]:
    x, y, z = [float(value) for value in xyz]
    rot = rpy_to_rot(*[float(value) for value in rpy])
    return [
        [rot[0][0], rot[0][1], rot[0][2], x],
        [rot[1][0], rot[1][1], rot[1][2], y],
        [rot[2][0], rot[2][1], rot[2][2], z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _xml_numbers(value: str | None, default: Sequence[float]) -> list[float]:
    if not value:
        return list(default)
    return [float(part) for part in value.split()]


@lru_cache(maxsize=1)
def _expanded_robot_urdf_root(urdf_path: str = str(DEFAULT_ROBOT_URDF)) -> ET.Element:
    path = Path(urdf_path).resolve()
    try:
        result = subprocess.run(
            ["xacro", str(path)],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(path.parent),
        )
        return ET.fromstring(result.stdout)
    except FileNotFoundError as exc:
        raise RuntimeError("xacro command not found; source ROS environment first") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.strip() or exc.stdout.strip()) from exc


@lru_cache(maxsize=1)
def _urdf_collision_data(urdf_path: str = str(DEFAULT_ROBOT_URDF)):
    root = _expanded_robot_urdf_root(urdf_path)
    collisions_by_link: dict[str, list[dict[str, object]]] = {}
    parent_by_child: dict[str, str] = {}

    for joint in root.iter("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is not None and child is not None:
            parent_by_child[child.attrib["link"]] = parent.attrib["link"]

    for link in root.iter("link"):
        link_name = link.attrib.get("name")
        if not link_name:
            continue
        for collision in link.findall("collision"):
            cylinder = collision.find("./geometry/cylinder")
            if cylinder is None:
                continue
            origin = collision.find("origin")
            xyz = _xml_numbers(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
            rpy = _xml_numbers(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
            spec = {
                "radius": float(cylinder.attrib["radius"]),
                "length": float(cylinder.attrib["length"]),
                "origin_matrix": matrix_from_xyz_rpy(xyz, rpy),
            }
            collisions_by_link.setdefault(link_name, []).append(spec)

    return collisions_by_link, parent_by_child


def collision_link_for_link(link: str) -> str | None:
    """若 link 本身无 collision，则向父 link 查找最近的圆柱 collision link。"""
    collisions_by_link, parent_by_child = _urdf_collision_data()
    current = link
    visited: set[str] = set()
    while current and current not in visited:
        visited.add(current)
        if collisions_by_link.get(current):
            return current
        current = parent_by_child.get(current)
    return None


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


def _trajectory_duration_seconds(point: JointTrajectoryPoint) -> float:
    return point.time_from_start.sec + point.time_from_start.nanosec * 1e-9


def _trajectory_positions_at_phase(
    trajectory: RobotTrajectory,
    phase: float,
) -> list[float]:
    """按归一化路径进度插值轨迹关节位置。"""
    points = trajectory.joint_trajectory.points
    if not points:
        raise ValueError("轨迹没有路径点")
    if len(points) == 1:
        return list(points[0].positions)

    times = [_trajectory_duration_seconds(point) for point in points]
    source_time = max(0.0, min(1.0, phase)) * times[-1]
    if source_time <= times[0]:
        return list(points[0].positions)
    if source_time >= times[-1]:
        return list(points[-1].positions)

    upper = bisect.bisect_right(times, source_time)
    lower = upper - 1
    span = times[upper] - times[lower]
    ratio = 0.0 if span <= 1e-12 else (source_time - times[lower]) / span
    return [
        start + ratio * (end - start)
        for start, end in zip(
            points[lower].positions,
            points[upper].positions,
        )
    ]


def merge_dual_arm_cartesian_trajectories(
    left: RobotTrajectory,
    right: RobotTrajectory,
) -> RobotTrajectory:
    """把左右臂 Cartesian 路径同步重采样为一条可一次执行的 dual_arm 轨迹。"""
    left_names = list(left.joint_trajectory.joint_names)
    right_names = list(right.joint_trajectory.joint_names)
    overlap = set(left_names).intersection(right_names)
    if overlap:
        raise ValueError(f"左右轨迹包含重复关节: {sorted(overlap)}")
    if not left.joint_trajectory.points or not right.joint_trajectory.points:
        raise ValueError("左右 Cartesian 轨迹不能为空")

    left_duration = _trajectory_duration_seconds(left.joint_trajectory.points[-1])
    right_duration = _trajectory_duration_seconds(right.joint_trajectory.points[-1])
    duration = max(left_duration, right_duration)
    if duration <= 0.0:
        raise ValueError(
            f"Cartesian 轨迹时长无效: left={left_duration}, right={right_duration}"
        )

    sample_count = max(
        1,
        int(math.ceil(duration / UNLOAD_SYNC_SAMPLE_PERIOD)),
    )
    merged = RobotTrajectory()
    merged.joint_trajectory.header.frame_id = SCENE_FRAME
    merged.joint_trajectory.joint_names = left_names + right_names

    for index in range(sample_count + 1):
        phase = index / sample_count
        timestamp = duration * phase
        point = JointTrajectoryPoint()
        point.positions = (
            _trajectory_positions_at_phase(left, phase)
            + _trajectory_positions_at_phase(right, phase)
        )
        point.time_from_start.sec = int(timestamp)
        point.time_from_start.nanosec = int(
            round((timestamp - int(timestamp)) * 1e9)
        )
        if point.time_from_start.nanosec >= 1_000_000_000:
            point.time_from_start.sec += 1
            point.time_from_start.nanosec -= 1_000_000_000
        merged.joint_trajectory.points.append(point)
    return merged


# =============================================================================
# ROS 节点：场景管理 + move_action 调用
# =============================================================================


class G01Demo(Node):
    """封装 apply_planning_scene 与 move_action，对外提供少量高层接口。"""

    def __init__(self, sim_mode: bool = False):
        super().__init__("g01_demo")
        self.sim_mode = bool(sim_mode)
        self._scene_cli = self.create_client(ApplyPlanningScene, SVC_APPLY_SCENE)
        self._cart_cli = self.create_client(GetCartesianPath, SVC_CARTESIAN_PATH)
        self._ik_cli = self.create_client(GetPositionIK, SVC_COMPUTE_IK)
        self._fk_cli = self.create_client(GetPositionFK, SVC_COMPUTE_FK)
        self._state_validity_cli = self.create_client(
            GetStateValidity, SVC_CHECK_STATE_VALIDITY
        )
        self._left_tool_cli = self.create_client(SetToolPower, LEFT_TOOL_COMMAND_SERVICE)
        self._right_tool_cli = self.create_client(SetToolPower, RIGHT_TOOL_COMMAND_SERVICE)
        self._move_cli = ActionClient(self, MoveGroup, ACT_MOVE_GROUP)
        self._exec_cli = ActionClient(self, ExecuteTrajectory, ACT_EXEC_TRAJ)
        self._vision_tf_broadcaster = StaticTransformBroadcaster(self)
        frame_roll, frame_pitch, frame_yaw = (
            math.radians(value) for value in FRAME_RPY_DEG
        )
        self._frame_pose = make_pose(
            *FRAME_CENTER,
            frame_roll,
            frame_pitch,
            frame_yaw,
        )
        self._frame_top_pose = pose_offset_local(
            self._frame_pose,
            0.0,
            0.0,
            FRAME_SIZE[2] / 2.0,
        )
        self._box_obstacle_top_pose = (
            box_obstacle_top_pose_from_frame_top(self._frame_top_pose)
        )
        # 缓存最新 joint_states，供规划起点使用
        self._joints: dict[str, float] = {}
        self._js_count = 0
        self._latest_grasp_cmd: dict | None = None
        self._start_grasp_cmd: dict | None = None
        self.last_pick_failure_reason: str | None = None
        # UInt8 会在回调中显式限制到 uint8 范围。真实模式在收到首帧前为未知，
        # 未知状态按“没有安全空位”处理；仿真模式默认四个 SW 均有料，
        # 便于直接测试双臂下料流程。
        self._driver_signal: int | None = (
            ALL_PLACE_SLOTS_MATERIAL_SIGNAL if self.sim_mode else None
        )
        self._driver_signal_count = 0
        self.create_subscription(JointState, "/g01/joint_states", self._on_js, 10)
        self.create_subscription(String, GRASP_CMD_TOPIC, self._on_grasp_cmd, 10)
        # 只保留最新一帧，避免循环间 input() 阻塞时积压旧的开关状态。
        self.create_subscription(UInt8, DRIVER_SIGNAL_TOPIC, self._on_driver_signal, 1)
        self._grasp_result_pub = self.create_publisher(String, GRASP_CMD_RESULT_TOPIC, 10)
        self._cylinder_marker_pub = self.create_publisher(Marker, CYLINDER_MARKER_TOPIC, 1)
        self._cylinder_marker_ids: dict[str, int] = {}
        self._next_cylinder_marker_id = 0

    def _on_driver_signal(self, msg: UInt8):
        """缓存驱动板 UInt8 信号；SW 位非 0=空，SW 位为 0=已有物体。"""
        signal = int(msg.data) & 0xFF
        self._driver_signal_count += 1
        changed = signal != self._driver_signal
        self._driver_signal = signal

        if changed:
            states = ", ".join(
                f"{slot_name.upper()}={'空' if signal & mask else '有物体'}"
                for slot_name, mask in PLACE_SLOT_MASKS.items()
            )
            self.get_logger().info(
                f"收到 {DRIVER_SIGNAL_TOPIC}: uint8=0x{signal:02X}, {states}"
            )

    def _wait_for_driver_signal(
        self,
        timeout_sec: float = DRIVER_SIGNAL_WAIT_TIMEOUT_SEC,
        require_new: bool = False,
    ) -> bool:
        """等待可靠的放置位信号；require_new=True 时必须收到调用后处理的新帧。"""
        if self.sim_mode:
            return True
        signal_count_before_wait = self._driver_signal_count
        if self._driver_signal is not None and not require_new:
            return True
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(
                self,
                timeout_sec=min(0.1, max(0.0, deadline - time.monotonic())),
            )
            if self._driver_signal is not None and (
                not require_new or self._driver_signal_count > signal_count_before_wait
            ):
                return True
        if self._driver_signal is None or (
            require_new and self._driver_signal_count <= signal_count_before_wait
        ):
            wait_desc = "新一帧" if require_new else "首帧"
            self.get_logger().error(
                f"[pick] {timeout_sec:.1f}s 内未收到 {DRIVER_SIGNAL_TOPIC} {wait_desc}，"
                "无法确认放置位，禁止抓取"
            )
            return False
        return True

    def _place_slot_is_empty(self, slot_name: str) -> bool:
        """按 SW 掩码判断指定槽位是否为空；未知名称/未知信号均返回 False。"""
        mask = PLACE_SLOT_MASKS.get(str(slot_name).lower())
        if mask is None or self._driver_signal is None:
            return False
        return (self._driver_signal & mask) != 0

    def _place_slot_has_material(self, slot_name: str) -> bool:
        """按实时 SW 掩码判断指定位置是否有料；信号未知时不视为有料。"""
        mask = PLACE_SLOT_MASKS.get(str(slot_name).lower())
        if mask is None or self._driver_signal is None:
            return False
        return (self._driver_signal & mask) == 0

    def _has_any_empty_place_slot(self) -> bool:
        return any(self._place_slot_is_empty(name) for name in PLACE_SLOT_MASKS)

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
        if self.sim_mode:
            log.info(f"[sim] 跳过 {side} SetToolPower({status})")
            return True

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

    def _apply_scene(
        self,
        objects: list[CollisionObject],
        colors: list[ObjectColor] | None = None,
        report_error: bool = True,
    ) -> bool:
        """向 move_group 提交规划场景 diff（添加/删除障碍物）。"""
        if not self._scene_cli.wait_for_service(timeout_sec=10.0):
            if report_error:
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
            if report_error:
                self.get_logger().error("apply_planning_scene 失败")
            return False
        return True

    def add_frame(self) -> bool:
        """按当前动态深框姿态添加深框和长方体障碍物。"""
        colors = [
            ObjectColor(id=FRAME_ID, color=FRAME_COLOR),
            ObjectColor(id=BOX_OBSTACLE_ID, color=BOX_OBSTACLE_COLOR),
        ]
        return self._apply_scene(
            [
                make_deep_frame(self._frame_pose),
                make_box_obstacle(self._box_obstacle_top_pose),
            ],
            colors,
        )

    def remove_frame(self) -> bool:
        """同时从场景中删除深框和长方体障碍物。"""
        objects = [
            CollisionObject(id=FRAME_ID, operation=CollisionObject.REMOVE),
            CollisionObject(id=BOX_OBSTACLE_ID, operation=CollisionObject.REMOVE),
        ]
        return self._apply_scene(objects)

    def clear_managed_scene_objects(self) -> None:
        """逐个清除本程序的碰撞体；不存在或删除失败时直接继续。"""
        object_ids = (
            FRAME_ID,
            BOX_OBSTACLE_ID,
            FRAME_CUTOFF_ID,
            GRASP_OBJECT_COLLISION_ID,
            UNLOAD_TABLE_ID,
            UNLOAD_TABLE_TOP_BOX_ID,
            UNLOAD_OBSTACLE_ID,
        )
        if not self._scene_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().info(
                f"[startup] 服务 {SVC_APPLY_SCENE} 暂不可用，跳过清理并继续"
            )
            return

        removed_ids = []
        skipped_ids = []
        for object_id in object_ids:
            removal = CollisionObject(
                id=object_id,
                operation=CollisionObject.REMOVE,
            )
            if self._apply_scene([removal], report_error=False):
                removed_ids.append(object_id)
            else:
                skipped_ids.append(object_id)
        self.get_logger().info(
            f"[startup] 场景障碍物清理完成："
            f"{len(removed_ids)} 个删除成功，{len(skipped_ids)} 个不存在或未确认；继续"
        )
        if skipped_ids:
            self.get_logger().info(
                "[startup] 以下障碍物不存在或删除未确认，已忽略: "
                + ", ".join(skipped_ids)
            )

    def configure_deep_frame_from_recognition(
        self,
        recognition_pose: Pose,
    ) -> Pose:
        """保存由 p,4 识别位姿计算出的深框和长方体动态基准。"""
        self._frame_top_pose, self._frame_pose = (
            deep_frame_poses_from_recognition(recognition_pose)
        )
        self._box_obstacle_top_pose = (
            box_obstacle_top_pose_from_frame_top(self._frame_top_pose)
        )
        return copy.deepcopy(self._frame_pose)

    def publish_deep_frame_vision_tf(
        self,
        recognition_pose: Pose,
        frame_top_pose: Pose,
        frame_pose: Pose,
        box_top_pose: Pose,
    ) -> None:
        """发布 p,4、深框顶部/实体中心和长方体上表面中心坐标系。"""
        transforms = []
        for child_frame, pose in (
            (FRAME_VISION_TF_FRAME, recognition_pose),
            (FRAME_TOP_CENTER_TF_FRAME, frame_top_pose),
            (FRAME_CENTER_TF_FRAME, frame_pose),
            (BOX_OBSTACLE_TOP_TF_FRAME, box_top_pose),
        ):
            transform = TransformStamped()
            transform.header.frame_id = SCENE_FRAME
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.child_frame_id = child_frame
            transform.transform.translation.x = pose.position.x
            transform.transform.translation.y = pose.position.y
            transform.transform.translation.z = pose.position.z
            transform.transform.rotation = copy.deepcopy(pose.orientation)
            transforms.append(transform)
        self._vision_tf_broadcaster.sendTransform(transforms)
        self.get_logger().info(
            f"[frame-vision] 已发布 TF: {SCENE_FRAME} → "
            f"{FRAME_VISION_TF_FRAME}、{FRAME_TOP_CENTER_TF_FRAME}、"
            f"{FRAME_CENTER_TF_FRAME}、{BOX_OBSTACLE_TOP_TF_FRAME}"
        )

    def add_unload_scene(self, recognition_pose: Pose) -> bool:
        """添加 p,2 识别得到的料台、台面薄长方体及墙状障碍物。"""
        colors = [
            ObjectColor(id=UNLOAD_TABLE_ID, color=UNLOAD_TABLE_COLOR),
            ObjectColor(
                id=UNLOAD_TABLE_TOP_BOX_ID,
                color=UNLOAD_TABLE_TOP_BOX_COLOR,
            ),
            ObjectColor(id=UNLOAD_OBSTACLE_ID, color=UNLOAD_OBSTACLE_COLOR),
        ]
        return self._apply_scene(
            [
                make_unload_table(recognition_pose),
                make_unload_table_top_box(recognition_pose),
                make_unload_obstacle(recognition_pose),
            ],
            colors,
        )

    def publish_unload_vision_tf(self, recognition_pose: Pose) -> None:
        """发布 base_link → 上料台视觉识别坐标系的固定 TF。"""
        transform = TransformStamped()
        transform.header.frame_id = SCENE_FRAME
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.child_frame_id = UNLOAD_VISION_TF_FRAME
        transform.transform.translation.x = recognition_pose.position.x
        transform.transform.translation.y = recognition_pose.position.y
        transform.transform.translation.z = recognition_pose.position.z
        transform.transform.rotation = copy.deepcopy(
            recognition_pose.orientation
        )
        self._vision_tf_broadcaster.sendTransform(transform)
        self.get_logger().info(
            f"[unload] 已发布视觉坐标系 TF: "
            f"{SCENE_FRAME} → {UNLOAD_VISION_TF_FRAME}"
        )

    def publish_unload_table_top_tf(self, recognition_pose: Pose) -> None:
        """发布相对视觉局部 Y=180°、Z=90° 的桌面中心固定 TF。"""
        table_top_pose = make_unload_table_top_pose(recognition_pose)
        transform = TransformStamped()
        transform.header.frame_id = SCENE_FRAME
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.child_frame_id = UNLOAD_TABLE_TOP_TF_FRAME
        transform.transform.translation.x = table_top_pose.position.x
        transform.transform.translation.y = table_top_pose.position.y
        transform.transform.translation.z = table_top_pose.position.z
        transform.transform.rotation = copy.deepcopy(table_top_pose.orientation)
        self._vision_tf_broadcaster.sendTransform(transform)
        mode_prefix = "[sim] " if self.sim_mode else ""
        self.get_logger().info(
            f"[unload] {mode_prefix}已发布上料桌上表面中心 TF: "
            f"{SCENE_FRAME} → {UNLOAD_TABLE_TOP_TF_FRAME}"
        )

    def remove_unload_scene(self) -> bool:
        """移除下料流程使用的料台、台面薄长方体和墙状障碍物。"""
        objects = [
            CollisionObject(
                id=UNLOAD_TABLE_ID,
                operation=CollisionObject.REMOVE,
            ),
            CollisionObject(
                id=UNLOAD_TABLE_TOP_BOX_ID,
                operation=CollisionObject.REMOVE,
            ),
            CollisionObject(
                id=UNLOAD_OBSTACLE_ID,
                operation=CollisionObject.REMOVE,
            ),
        ]
        return self._apply_scene(objects)

    def add_frame_cutoff_for_pose(
        self,
        pose: Pose | dict,
        source_frame: str = PLAN_FRAME,
        target_frame: str = SCENE_FRAME,
        joint_names: Sequence[str] | None = None,
        tangent_link: str | None = None,
        tangent_joints: dict[str, float] | None = None,
    ) -> bool:
        """添加隔板和临时物体；传入 link 时，隔板低于其 collision 最低面 1 cm。"""
        scene_pose = self._pose_in_frame(
            pose,
            source_frame=source_frame,
            target_frame=target_frame,
            joint_names=joint_names,
        )
        if scene_pose is None:
            self.get_logger().error(f"无法把目标位姿从 {source_frame} 转到 {target_frame}")
            return False

        if tangent_link:
            link_surface = self._link_collision_min_z(
                tangent_link,
                target_frame=target_frame,
                joints=tangent_joints,
                joint_names=joint_names,
            )
            if link_surface is None:
                return False
            collision_min_z, collision_link = link_surface
            scene_z = collision_min_z - FRAME_CUTOFF_BELOW_COLLISION
            tangent_desc = (
                f"q_pre 状态下 collision link {collision_link} 的最低面 "
                f"z={collision_min_z:.3f} m，并向下偏移 "
                f"{FRAME_CUTOFF_BELOW_COLLISION:.3f} m"
            )
        else:
            half_extent_z = cylinder_half_extent_z(scene_pose)
            scene_z = scene_pose.position.z - half_extent_z
            tangent_desc = (
                f"目标圆柱，物体中心 z={scene_pose.position.z:.3f} m, "
                f"Z 半尺寸={half_extent_z:.3f} m"
            )
        colors = [
            ObjectColor(id=FRAME_CUTOFF_ID, color=FRAME_CUTOFF_COLOR),
            ObjectColor(id=GRASP_OBJECT_COLLISION_ID, color=GRASP_OBJECT_COLLISION_COLOR),
        ]
        self.get_logger().info(
            f"添加碰撞体「{FRAME_CUTOFF_ID}」+「{GRASP_OBJECT_COLLISION_ID}」到 {target_frame}: "
            f"隔板以{tangent_desc}为基准, "
            f"隔板上表面 z={scene_z:.3f} m"
        )
        return self._apply_scene(
            [
                make_deep_frame_cutoff(scene_z, self._frame_pose),
                make_grasp_object_collision(scene_pose),
            ],
            # [make_deep_frame_cutoff(scene_z)],
            colors,
        )

    def _link_collision_min_z(
        self,
        link: str,
        *,
        target_frame: str = SCENE_FRAME,
        joints: dict[str, float] | None = None,
        joint_names: Sequence[str] | None = None,
    ) -> tuple[float, str] | None:
        """读取 URDF cylinder collision，返回 link 在 target_frame 下的最低 Z 表面。"""
        log = self.get_logger()
        try:
            collision_link = collision_link_for_link(link)
            collisions_by_link, _ = _urdf_collision_data()
        except RuntimeError as exc:
            log.error(f"读取 URDF collision 失败：{exc}")
            return None

        if collision_link is None:
            log.error(f"未在 {link} 或其父 link 上找到 cylinder collision")
            return None

        fk_joints = joints
        if joints is not None and joint_names is not None:
            current = self._get_joints(list(joint_names), timeout=2.0)
            if current is not None:
                fk_joints = dict(current)
                fk_joints.update(joints)

        link_pose = self._get_link_pose_fk(
            collision_link,
            joints=fk_joints,
            joint_names=joint_names,
            plan_frame=target_frame,
        )
        if link_pose is None:
            log.error(f"无法计算 {collision_link} 在 {target_frame} 下的当前 FK")
            return None

        t_target_link = pose_to_matrix(link_pose)
        min_z: float | None = None
        for spec in collisions_by_link.get(collision_link, []):
            collision_matrix = matmul4(t_target_link, spec["origin_matrix"])
            collision_pose = xyz_rpy_to_pose(matrix_to_xyz_rpy(collision_matrix))
            half_extent_z = cylinder_half_extent_z_from_shape(
                collision_pose,
                radius=float(spec["radius"]),
                length=float(spec["length"]),
            )
            surface_z = collision_pose.position.z - half_extent_z
            min_z = surface_z if min_z is None else min(min_z, surface_z)

        if min_z is None:
            log.error(f"{collision_link} 没有可用 cylinder collision")
            return None
        return min_z, collision_link

    def add_frame_cutoff_only_for_pose(
        self,
        pose: Pose | dict,
        source_frame: str = PLAN_FRAME,
        target_frame: str = SCENE_FRAME,
        joint_names: Sequence[str] | None = None,
        tangent_link: str | None = None,
        tangent_joints: dict[str, float] | None = None,
    ) -> bool:
        """只添加隔板；传入 link 时，隔板低于其 collision 最低面 1 cm。"""
        if tangent_link:
            link_surface = self._link_collision_min_z(
                tangent_link,
                target_frame=target_frame,
                joints=tangent_joints,
                joint_names=joint_names,
            )
            if link_surface is None:
                return False
            collision_min_z, collision_link = link_surface
            scene_z = collision_min_z - FRAME_CUTOFF_BELOW_COLLISION
            tangent_desc = (
                f"q_pre 状态下 collision link {collision_link} 的最低面 "
                f"z={collision_min_z:.3f} m，并向下偏移 "
                f"{FRAME_CUTOFF_BELOW_COLLISION:.3f} m"
            )
        else:
            scene_pose = self._pose_in_frame(
                pose,
                source_frame=source_frame,
                target_frame=target_frame,
                joint_names=joint_names,
            )
            if scene_pose is None:
                self.get_logger().error(f"无法把目标位姿从 {source_frame} 转到 {target_frame}")
                return False
            half_extent_z = cylinder_half_extent_z(scene_pose)
            scene_z = scene_pose.position.z - half_extent_z
            tangent_desc = (
                f"目标圆柱，物体中心 z={scene_pose.position.z:.3f} m, "
                f"Z 半尺寸={half_extent_z:.3f} m"
            )
        self.get_logger().info(
            f"添加碰撞体「{FRAME_CUTOFF_ID}」到 {target_frame}: "
            f"隔板以{tangent_desc}为基准, "
            f"隔板上表面 z={scene_z:.3f} m"
        )
        return self._apply_scene(
            [make_deep_frame_cutoff(scene_z, self._frame_pose)],
            [ObjectColor(id=FRAME_CUTOFF_ID, color=FRAME_CUTOFF_COLOR)],
        )

    def remove_frame_cutoff(self) -> bool:
        """从场景中同时删除深框隔离面和临时抓取物体。"""
        removals = [
            CollisionObject(id=FRAME_CUTOFF_ID, operation=CollisionObject.REMOVE),
            CollisionObject(id=GRASP_OBJECT_COLLISION_ID, operation=CollisionObject.REMOVE),
        ]
        return self._apply_scene(removals)

    def _check_state_collision_free(
        self,
        group: str,
        joints: dict[str, float],
        state_joint_names: Sequence[str],
    ) -> bool:
        """用当前完整状态补齐 joints，并检查该 group 是否无碰撞。"""
        log = self.get_logger()
        state_joint_names = list(state_joint_names)
        current = self._get_joints(state_joint_names, wait_new=True)
        if current is None:
            log.error("[q-pre-collision] 读取完整机器人状态失败")
            return False

        state = dict(current)
        state.update(joints)
        if not self._state_validity_cli.wait_for_service(timeout_sec=10.0):
            log.error(f"服务 {SVC_CHECK_STATE_VALIDITY} 不可用")
            return False

        req = GetStateValidity.Request()
        req.group_name = group
        req.robot_state.is_diff = True
        req.robot_state.joint_state.name = list(state.keys())
        req.robot_state.joint_state.position = list(state.values())
        fut = self._state_validity_cli.call_async(req)
        if not self._spin_until(fut, 10.0):
            log.error(f"服务 {SVC_CHECK_STATE_VALIDITY} 调用超时")
            return False

        res = fut.result()
        if res.valid:
            return True

        contact_pairs = []
        for contact in res.contacts[:6]:
            body_1 = getattr(contact, "contact_body_1", "")
            body_2 = getattr(contact, "contact_body_2", "")
            pair = f"{body_1}<->{body_2}" if body_1 or body_2 else "未知碰撞对"
            if pair not in contact_pairs:
                contact_pairs.append(pair)
        contacts_text = ", ".join(contact_pairs) if contact_pairs else "服务未返回碰撞对"
        log.info(f"[q-pre-collision] q_pre 有碰撞：{contacts_text}")
        return False

    def _q_pre_valid_with_temporary_collisions(
        self,
        group: str,
        link: str,
        target_pose: Pose,
        plan_frame: str,
        q_pre: dict[str, float],
        state_joint_names: Sequence[str],
    ) -> bool:
        """临时加入隔离面和待抓物体，仅检查 q_pre 状态，然后恢复场景。"""
        log = self.get_logger()
        if not self.add_frame_cutoff_for_pose(
            target_pose,
            source_frame=plan_frame,
            target_frame=SCENE_FRAME,
            joint_names=state_joint_names,
            tangent_link=link,
            tangent_joints=q_pre,
        ):
            log.error("[q-pre-collision] 添加两个临时碰撞体失败")
            return False

        valid = False
        removed = False
        try:
            valid = self._check_state_collision_free(
                group,
                q_pre,
                state_joint_names,
            )
        finally:
            removed = self.remove_frame_cutoff()
            if not removed:
                log.error("[q-pre-collision] 移除两个临时碰撞体失败")
        return valid and removed

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

    def _solve_ik_candidates_from_seed(
        self,
        group: str,
        link: str,
        pose: Pose,
        joint_names: Sequence[str],
        seed_state: dict[str, float],
        *,
        plan_frame: str,
        n_attempts: int,
        max_solutions: int,
        random_seed: int,
        perturb_joint_names: Sequence[str] | None = None,
        avoid_collisions: bool = False,
        perturb: float = UNLOAD_PLACE_IK_PERTURB,
        dedup_tol: float = 1e-2,
    ) -> list[dict[str, float]]:
        """从显式完整种子枚举多组 IK；允许固定额外关节参与 RobotState。

        joint_names 是需要从 IK 结果提取的规划组关节。seed_state 可以额外包含
        body_joint1/2 和另一只手臂，使 IK 服务在指定身体构型下求解。已知的
        双臂、升降和腰部关节按 URDF 完整限位采样，其他关节使用 perturb 扰动。
        """
        log = self.get_logger()
        joint_names = list(joint_names)
        perturb_joint_names = list(
            joint_names if perturb_joint_names is None else perturb_joint_names
        )
        missing_seed = [name for name in joint_names if name not in seed_state]
        if missing_seed:
            log.error(
                f"[unload-ik] group={group} 种子缺少关节: {missing_seed}"
            )
            return []

        rng = random.Random(int(random_seed))
        solutions: list[dict[str, float]] = []
        failure_codes: dict[int, int] = {}

        for attempt in range(max(1, int(n_attempts))):
            seed = dict(seed_state)
            if attempt > 0:
                for name in perturb_joint_names:
                    if name not in seed:
                        continue
                    if name == "body_joint1":
                        seed[name] = rng.uniform(*BODY_JOINT1_LIMITS)
                    elif name == "body_joint2":
                        seed[name] = rng.uniform(*BODY_JOINT2_LIMITS)
                    else:
                        arm_match = re.fullmatch(
                            r"[lr]_arm_joint([1-6])",
                            name,
                        )
                        if arm_match:
                            joint_index = int(arm_match.group(1))
                            seed[name] = rng.uniform(
                                *ARM_JOINT_LIMITS_BY_INDEX[joint_index]
                            )
                        else:
                            perturbed = (
                                seed[name] + rng.uniform(-perturb, perturb)
                            )
                            seed[name] = math.atan2(
                                math.sin(perturbed),
                                math.cos(perturbed),
                            )

            solution, code = self._solve_ik(
                group,
                link,
                pose,
                seed,
                avoid_collisions=avoid_collisions,
                return_code=True,
                plan_frame=plan_frame,
            )
            if solution is None:
                if code is not None:
                    failure_codes[int(code)] = (
                        failure_codes.get(int(code), 0) + 1
                    )
                continue

            candidate = {
                name: solution[name]
                for name in joint_names
                if name in solution
            }
            if len(candidate) != len(joint_names):
                continue
            duplicate = any(
                all(
                    abs(candidate[name] - old[name]) < dedup_tol
                    for name in joint_names
                )
                for old in solutions
            )
            if duplicate:
                continue
            solutions.append(candidate)
            if len(solutions) >= max(1, int(max_solutions)):
                break

        solutions.sort(
            key=lambda item: joint_distance_squared(
                item,
                seed_state,
                joint_names,
            )
        )
        failure_text = ""
        if failure_codes:
            failure_text = ", failures=" + ", ".join(
                f"{_moveit_error_name(code)}:{count}"
                for code, count in sorted(failure_codes.items())
            )
        log.info(
            f"[unload-ik] group={group}, frame={plan_frame}, "
            f"solutions={len(solutions)}/{n_attempts}, "
            f"avoid_collisions={avoid_collisions}{failure_text}"
        )
        return solutions

    def _solve_ik_multi(
        self,
        group: str,
        link: str,
        pose: Pose,
        joint_names: Sequence[str],
        n_candidates: int = IK_N_CANDIDATES,
        dedup_tol: float = 1e-2,
        avoid_collisions: bool = True,
        plan_frame: str = PLAN_FRAME,
    ) -> list[dict]:
        """通过随机种子枚举 pose 在 group 上的多个不同 IK 解。

        - 第 0 次以当前 joint_states 为种子，能拿到「最自然」的解。
        - 之后每次按 URDF 位置限位，对 `joint_names` 中的关节做全范围均匀采样。
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

        try:
            joint_limits = {
                name: ik_seed_limits_for_joint(name)
                for name in joint_names
            }
        except KeyError as exc:
            log.error(f"[ik-multi] {exc}")
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
                seed = {
                    name: rng.uniform(*joint_limits[name])
                    for name in joint_names
                }
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
            f"(按 URDF 关节限位采样, avoid_collisions={avoid_collisions})"
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
                    n_candidates=n_candidates, dedup_tol=dedup_tol,
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
                        "  对策：抬高 target z、增加 IK_N_CANDIDATES / IK_TIMEOUT_SEC、或在 RViz 拖动 IK marker 直接验证可达性"
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
        cutoff_joint_names: Sequence[str] | None = None,
    ):
        """从 target_pose 的多个 IK 解里挑可用的一组。

        target_seeds 不为 None 时，只用这些 seed 求 IK，不做当前关节/随机 seed 枚举。
        cutoff_joint_names 不为 None 时，还会临时加入隔离面和待抓物体，并要求
        q_pre 通过 check_state_validity；这里只做状态碰撞检查，不做 OMPL 规划。
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
        feasible_pairs = []

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
            feasible_pairs.append(
                (
                    score,
                    q_pre,
                    q_target,
                    approach_traj,
                    idx + 1,
                    point_count,
                    total_motion,
                    max_step,
                )
            )

        feasible_pairs.sort(key=lambda item: item[0])
        for (
            _score,
            q_pre,
            q_target,
            approach_traj,
            best_idx,
            point_count,
            total_motion,
            max_step,
        ) in feasible_pairs:
            if cutoff_joint_names is not None:
                log.info(
                    f"[grasp-select] 检查候选 IK {best_idx}/{len(candidates)}："
                    f"加入「{FRAME_CUTOFF_ID}」+「{GRASP_OBJECT_COLLISION_ID}」后的 q_pre"
                )
                if not self._q_pre_valid_with_temporary_collisions(
                    group,
                    link,
                    target_pose,
                    plan_frame,
                    q_pre,
                    cutoff_joint_names,
                ):
                    log.info(
                        f"[grasp-select] 候选 IK {best_idx}/{len(candidates)}："
                        "q_pre 在两个新碰撞体加入后未通过碰撞检查 → 淘汰"
                    )
                    continue
            log.info(
                f"[grasp-select] 选用候选 IK {best_idx}/{len(candidates)}："
                f"{point_count} 点, joint_motion={total_motion:.3f}, "
                f"max_step={max_step:.3f}"
            )
            return q_pre, q_target, approach_traj

        if feasible_pairs and cutoff_joint_names is not None:
            log.error(
                f"[grasp-select] {len(feasible_pairs)} 个直线可行候选中，没有一个 q_pre "
                f"能在加入「{FRAME_CUTOFF_ID}」+「{GRASP_OBJECT_COLLISION_ID}」后保持无碰撞"
            )
            return None

        log.error(
            f"[grasp-select] 共 {len(candidates)} 个候选 IK 解，没有一个能直线退回 pre_pose"
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

    def _has_available_place_target(
        self,
        arm_group: str,
        place_joints: dict[str, object],
        first_return_mode: int,
    ) -> bool:
        """检查硬件信号所示的空位中，是否存在已配置的放置目标。"""
        available, _ = self._select_available_place_target(
            arm_group, place_joints, first_return_mode
        )
        return available

    def _select_available_place_target(
        self,
        arm_group: str,
        place_joints: dict[str, object],
        first_return_mode: int,
    ) -> tuple[bool, str | None]:
        """返回 (是否可放, SW 名称)；字典配置会选择第一个真实空位。"""
        suffix = "_j" if first_return_mode == 1 else ""
        yubei_config = place_joints.get(f"yubei{suffix}")
        fang_config = place_joints.get(f"fang{suffix}")
        if isinstance(fang_config, dict):
            if not isinstance(yubei_config, dict):
                return False, None
            slot_name = next(
                (
                    str(name).lower()
                    for name in fang_config
                    if name in yubei_config and self._place_slot_is_empty(str(name))
                ),
                None,
            )
            return slot_name is not None, slot_name

        # *_j 是没有 SW 名称的单一旧配置，无法把某个掩码映射到不同关节目标；
        # 仍要求四个硬件检测位中至少存在一个空位，确保“无空位不抓取”。
        configured = yubei_config is not None and fang_config is not None
        return configured and self._has_any_empty_place_slot(), None

    def _place_and_return(
        self,
        speed: float,
        place_joints: dict[str, object],
        *,
        group: str,
        link: str,
        first_return_mode: int,
        selected_slot_name: str | None = None,
    ) -> bool:
        """body+手臂到 yubei，再用纯臂放置、原路返回并运动到 Q1。"""
        log = self.get_logger()
        arm_context = arm_context_for_group(group)
        if arm_context is None:
            log.error(f"[pick] group={group} 无法确定放置使用的纯臂 group")
            return False
        arm_group, arm_plan_frame = arm_context
        arm_joint_names = joint_names_for_group(arm_group)
        body_group = "left_body" if arm_group == "left_arm" else "right_body"
        body_joint_names = joint_names_for_group(body_group)
        suffix = "_j" if first_return_mode == 1 else ""
        yubei_key = f"yubei{suffix}"
        fang_key = f"fang{suffix}"
        if yubei_key not in place_joints or fang_key not in place_joints:
            log.error(f"[pick] group={group} 的放置配置缺少 {yubei_key}/{fang_key}")
            return False

        yubei_config = place_joints[yubei_key]
        fang_config = place_joints[fang_key]
        slot_name: str | None = None
        if isinstance(fang_config, dict):
            if not isinstance(yubei_config, dict):
                log.error(f"[pick] {yubei_key} 不是按 SW 命名的放置配置")
                return False
            if (
                selected_slot_name is not None
                and selected_slot_name in fang_config
                and selected_slot_name in yubei_config
                and self._place_slot_is_empty(selected_slot_name)
            ):
                slot_name = selected_slot_name
            else:
                # 抓取过程中原目标可能被占用；放置动作开始前允许改选另一个空位。
                available, slot_name = self._select_available_place_target(
                    arm_group, place_joints, first_return_mode
                )
                if not available or slot_name is None:
                    log.error(f"[pick] {fang_key} 没有硬件信号为“空”的可用放置位")
                    return False
                if selected_slot_name is not None and slot_name != selected_slot_name:
                    log.warning(
                        f"[pick] 原定放置位 {selected_slot_name} 已不可用，改用 {slot_name}"
                    )
            fang_values = fang_config[slot_name]
        else:
            if not self._has_any_empty_place_slot():
                log.error("[pick] SW1~SW4 均有物体，禁止执行放置")
                return False
            fang_values = fang_config

        if isinstance(yubei_config, dict):
            if slot_name is None or slot_name not in yubei_config:
                log.error(f"[pick] {yubei_key} 缺少与 {fang_key} 对应的槽位 {slot_name}")
                return False
            yubei_values = yubei_config[slot_name]
        else:
            yubei_values = yubei_config

        try:
            yubei_joints = list(yubei_values)
            fang_joints = list(fang_values)
        except TypeError:
            log.error(f"[pick] {yubei_key}/{fang_key} 关节配置不是序列")
            return False
        if len(yubei_joints) != len(body_joint_names):
            log.error(
                f"[pick] {yubei_key} 配置长度错误："
                f"{len(yubei_joints)} != {len(body_joint_names)}"
            )
            return False
        if len(fang_joints) != len(arm_joint_names):
            log.error(
                f"[pick] {fang_key} 放置配置长度错误："
                f"{len(fang_joints)} != {len(arm_joint_names)}"
            )
            return False
        yubei_name = f"{yubei_key}.{slot_name}" if slot_name else yubei_key
        place_name = f"{fang_key}.{slot_name}" if slot_name else fang_key

        log.info(f"[pick] 5/9  {body_group} OMPL → {yubei_name}")
        current = self._get_joints(body_joint_names, wait_new=True)
        if current is None:
            log.error("[pick] 读取当前 body+手臂关节失败")
            return False
        goal = [make_joint_constraints_from_vector(body_group, body_joint_names, yubei_joints)]
        ok, used_ms, _to_yubei_traj = self.move(
            body_group, goal, start=current, plan_only=False, speed_scale=speed
        )
        log.info(f"[pick] OMPL → {yubei_name}: {used_ms:.3f} ms ({'success' if ok else 'failed'})")
        if not ok:
            log.error(f"[pick] 运动到 {yubei_name} 失败")
            return False

        log.info("按回车继续 …")
        try:
            input()
        except EOFError:
            pass

        log.info(f"[pick] 6/9  {arm_group} 笛卡尔直线 → {place_name}（不考虑碰撞）")
        current = self._get_joints(arm_joint_names, wait_new=True)
        if current is None:
            log.error("[pick] 读取当前手臂关节失败")
            return False
        fang_pose = self._get_link_pose_fk(
            link,
            joints=dict(zip(arm_joint_names, fang_joints)),
            plan_frame=arm_plan_frame,
        )
        if fang_pose is None:
            log.error(f"[pick] 无法由 {place_name} 关节角计算末端位姿")
            return False
        to_fang_traj = self._cartesian_plan(
            arm_group,
            link,
            fang_pose,
            speed_scale=speed,
            avoid_collisions=False,
            start_joints=current,
            joint_names=arm_joint_names,
            plan_frame=arm_plan_frame,
        )
        if to_fang_traj is None:
            log.error(f"[pick] 直线运动到 {place_name} 规划失败")
            return False
        if not self._execute_traj(to_fang_traj):
            log.error(f"[pick] 直线运动到 {place_name} 执行失败")
            return False
        log.info(
            f"[pick] {place_name} 关节: "
            + ", ".join(f"{n}={v:.3f}" for n, v in zip(arm_joint_names, fang_joints))
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
        if slot_name is not None and not self._place_slot_is_empty(slot_name):
            log.error(
                f"[pick] 放置前复查发现 {slot_name} 已有物体，保持工具上电并中止放置"
            )
            return False
        if slot_name is None and not self._has_any_empty_place_slot():
            log.error("[pick] 放置前复查发现 SW1~SW4 均有物体，保持工具上电并中止放置")
            return False
        log.info(f"[pick] 7/9  {tool_label}工具下电，放置到 {place_name}")
        if not self.set_tool_power(tool_side, 0):
            log.error(f"[pick] {tool_label}工具下电失败")
            return False
        print(f"\033[32m{tool_label}下电成功\033[0m")

        log.info(
            f"[pick] 8/9  反向播放到 {place_name} 的轨迹原路返回 "
            f"（{len(to_fang_traj.joint_trajectory.points)} 点）"
        )
        if not self._execute_traj(self._reverse_trajectory(to_fang_traj)):
            log.error("[pick] 8/9 原路返回失败")
            return False

        q1_count = len(arm_joint_names)
        if arm_group == "left_arm":
            q1_source = list(EXCHANGE_Q1[:6])
            q1_joints = q1_source[:q1_count]
            q1_slice_text = "前6"
        elif arm_group == "right_arm":
            q1_source = list(EXCHANGE_Q1[-6:])
            q1_joints = q1_source[-q1_count:]
            q1_slice_text = "后6"
        else:
            log.error(f"[pick] 9/9 group={arm_group} 不支持按 Q1 固定点返回")
            return False
        if len(q1_joints) != q1_count:
            log.error(
                f"[pick] 9/9 Q1 固定点长度错误：Q1={len(q1_joints)}, "
                f"关节数={q1_count}"
            )
            return False

        log.info(
            f"[pick] 9/9  从当前位置运动到 Q1 固定点 "
            f"（{q1_slice_text} 个数，使用 {q1_count} 个关节）"
        )
        current = self._get_joints(arm_joint_names, wait_new=True)
        if current is None:
            log.error("[pick] 9/9 读取当前关节失败")
            return False
        goal = [make_joint_constraints_from_vector(arm_group, arm_joint_names, q1_joints)]
        ok, used_ms, _to_q1_traj = self.move(
            arm_group, goal, start=current, plan_only=False, speed_scale=speed
        )
        log.info(f"[pick] OMPL → Q1 固定点: {used_ms:.3f} ms ({'success' if ok else 'failed'})")
        if not ok:
            log.error("[pick] 9/9 运动到 Q1 固定点失败")
            return False
        log.info(
            "[pick] Q1 固定点返回完成: "
            + ", ".join(f"{n}={v:.3f}" for n, v in zip(arm_joint_names, q1_joints))
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
            joint_names       : group 内关节名顺序
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

        log.info(
            f"[pick] group={group}, link={link}, plan_frame={plan_frame}"
        )
        # 抓取前先下电
        tool_side = tool_side_for_link(link)
        if tool_side is None:
            log.error(f"[pick] link={link} 不是 l_tool 或 r_tool，无法判断抓取工具侧")
            return False
        tool_label = "左臂" if tool_side == "left" else "右臂"

        if first_return_mode == 1:
            place_arm_group = "left_arm" if tool_side == "right" else "right_arm"
        else:
            arm_context = arm_context_for_group(group)
            if arm_context is None:
                log.error(f"[pick] group={group} 无法确定放置手臂")
                return False
            place_arm_group, _ = arm_context
        place_joint_config = PLACE_JOINTS.get(place_arm_group)
        if not isinstance(place_joint_config, dict):
            log.error(f"[pick] PLACE_JOINTS 未配置 {place_arm_group}")
            return False
        if not self._wait_for_driver_signal():
            self.last_pick_failure_reason = "no_place_signal"
            return False
        place_available, selected_place_slot = self._select_available_place_target(
            place_arm_group, place_joint_config, first_return_mode
        )
        if not place_available:
            signal_text = (
                "unknown"
                if self._driver_signal is None
                else f"0x{self._driver_signal:02X}"
            )
            log.error(
                f"[pick] 没有信号为“空”且已配置的放置位"
                f"（driver_signal={signal_text}），取消本次抓取"
            )
            self.last_pick_failure_reason = "no_place"
            return False
        if selected_place_slot is not None:
            log.info(
                f"[pick] 抓取前确认放置位 {selected_place_slot.upper()} 为空，"
                "本次抓取将使用该位置"
            )

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

        log.info("[pick] 0/9  IK 多解枚举 + approach 预检 …")
        picked = self._select_feasible_grasp_pair(
            group, link, target_pose, pre_pose,
            joint_names=joint_names,
            speed_scale=speed_scale,
            plan_frame=plan_frame,
            cutoff_joint_names=cutoff_joint_names,
        )
        if picked is None:
            log.error(
                "[pick] 未找到「IK 可解 + cartesian approach 可行 + "
                "加入两个临时碰撞体后 q_pre 无碰撞」的 IK 解"
            )
            self.last_pick_failure_reason = "no_ik"
            return False
        q_pre, q_target, approach_traj = picked
        log.info(
            "[pick] 选定 q_pre: "
            + ", ".join(f"{n}={q_pre[n]:.3f}" for n in joint_names)
        )

        log.info("[pick] 1/9  OMPL  → q_pre（关节目标，IK 解已确定）")
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
                tangent_link=link,
                tangent_joints=q_pre,
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
            log.error("[pick] 1/9 未返回 OMPL 轨迹，无法原路退回初始位置")
            return False

        log.info(
            f"[pick] 2/9  执行已缓存的 approach 轨迹 → q_target "
            f"（{len(approach_traj.joint_trajectory.points)} 点，免重规划）"
        )

        try:
            input()
        except EOFError:
            pass

        if not self._execute_traj(approach_traj):
            log.error("[pick] 直线接近执行失败")
            return False

        self._log_actual_fk_error(
            "[pick] 3/9",
            link=link,
            target_pose=target_pose,
            plan_frame=plan_frame,
            joint_names=joint_names,
        )
        log.info("[pick] 3/9  到达抓取位置，按回车继续 …")
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
            else "反向 approach + 1/9 OMPL → 1/9 初始位置"
        )
        log.info(f"[pick] 4/9  第一段复位：{return_desc}")
        retreat_approach = self._reverse_trajectory(approach_traj)
        if not self._execute_traj(retreat_approach):
            log.error("[pick] 反向播放 approach 失败")
            return False

        try:
            input()
        except EOFError:
            pass

        if not self.add_frame_cutoff_only_for_pose(
            target_pose,
            source_frame=plan_frame,
            target_frame=SCENE_FRAME,
            joint_names=cutoff_joint_names,
            tangent_link=link,
            tangent_joints=q_pre,
        ):
            log.error("[pick] 第一段复位后添加隔板失败")
            return False

        if first_return_mode == 1:
            log.info(
                "[pick] 第一段复位完成，已反向直线回到 q_pre: "
                + ", ".join(f"{n}={q_pre[n]:.3f}" for n in joint_names)
            )
            
            log.info("[pick] 3/9  到达抓取位置，按回车继续 …")
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
                pick_group=group,
                dual_speed=place_speed_scale,
                cartesian_speed=speed_scale,
            ):
                return False

            if not self._place_and_return(
                place_speed_scale,
                place_joint_config,
                group=place_arm_group,
                link="l_tool" if place_arm_group == "left_arm" else "r_tool",
                first_return_mode=first_return_mode,
                selected_slot_name=selected_place_slot,
            ):
                return False

        else:
            if not self._place_and_return(
                place_speed_scale,
                place_joint_config,
                group=group,
                link=link,
                first_return_mode=first_return_mode,
                selected_slot_name=selected_place_slot,
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
        pick_group: str,
        dual_speed: float = 0.2,
        cartesian_speed: float = 0.2,
        z_down_distance: float | None = None,
    ) -> bool:
        """双臂交换：Q2 根据抓取 group 选择双臂/+腰/+升降规划组。"""
        log = self.get_logger()
        arm_dual_group = "dual_arm"
        arm_dual_joint_names = joint_names_for_group(arm_dual_group)

        if pick_group in ("left_arm", "right_arm"):
            exchange_group = "dual_arm"
            exchange_joint_names = arm_dual_joint_names
            q2_slice_text = "后12个"
        elif pick_group in ("left_waist", "right_waist"):
            exchange_group = "dual_arm_waist"
            exchange_joint_names = ["body_joint2", *arm_dual_joint_names]
            q2_slice_text = "后13个"
        elif pick_group in ("left_body", "right_body"):
            exchange_group = "dual_arm_body"
            exchange_joint_names = ["body_joint1", "body_joint2", *arm_dual_joint_names]
            q2_slice_text = "全部14个"
        else:
            log.error(f"[exchange] 不支持 pick_group={pick_group}")
            return False

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
        q2_source = list(q2_by_side[source_side])
        if len(q2_source) != 14:
            log.error(f"[exchange] EXCHANGE_Q2[{source_side!r}] 长度错误: {len(q2_source)} != 14")
            return False
        q2 = q2_source[-len(exchange_joint_names):]

        if not self.set_tool_power(receiver_side, 0):
            log.error(f"[exchange] {receiver_label}工具下电失败")
            return False

        if len(q1) != len(arm_dual_joint_names):
            log.error(f"[exchange] q1 长度错误: {len(q1)} != {len(arm_dual_joint_names)}")
            return False
        if len(q2) != len(exchange_joint_names):
            log.error(f"[exchange] q2 长度错误: {len(q2)} != {len(exchange_joint_names)}")
            return False

        log.info(
            f"[exchange] 1/4  {exchange_group} 使用 EXCHANGE_Q2 {q2_slice_text} "
            f"规划执行到 q2"
        )
        if not self.plan_execute_joint_waypoints(
            exchange_group,
            dual_speed,
            exchange_joint_names,
            [q2],
            num_attempts=20,
        ):
            log.error(f"[exchange] {exchange_group} 到 q2 失败")
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

        # log.info(f"[exchange] 4/4  {arm_dual_group} 规划执行到 q1")
        # if not self.plan_execute_joint_waypoints(
        #     arm_dual_group,
        #     dual_speed,
        #     arm_dual_joint_names,
        #     [q1],
        # ):
        #     log.error(f"[exchange] {arm_dual_group} 到 q1 失败")
        #     return False

        log.info("[exchange] 双臂交换流程完成")
        return True


# =============================================================================
# 主流程
# =============================================================================


def split_sim_args(argv: list[str] | None = None) -> tuple[bool, list[str]]:
    """取出脚本自己的 --sim，其余参数继续交给 rclpy。"""
    raw_args = list(sys.argv if argv is None else argv)
    sim_mode = "--sim" in raw_args
    ros_args = [arg for arg in raw_args if arg != "--sim"]
    return sim_mode, ros_args


def main(argv: list[str] | None = None) -> int:
    sim_mode, ros_args = split_sim_args(argv)
    rclpy.init(args=ros_args)
    node = G01Demo(sim_mode=sim_mode)
    log = node.get_logger()
    if sim_mode:
        log.info(
            "[sim] 仿真模式：跳过工具电源服务，视觉使用固定 viewer pose，"
            "SW1~SW4 默认一直有料（driver_signal=0x00）"
        )
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

    def prompt_workflow(message: str) -> str:
        """1 选择深框识别并连续抓取，2 选择原下料流程，q 退出。"""
        log.info(message)
        try:
            return input().strip().lower()
        except EOFError:
            return "q"

    def publish_attempt_result(result: bool, success_grasp_number: int) -> None:
        node.publish_grasp_cmd_result(result, success_grasp_number)

    def recognize_and_add_deep_frame() -> bool:
        """先运动到 框_Q1，再发送 p,4 并用识别结果重建深框碰撞场景。"""
        nonlocal frame_added
        q1_joint_names = list(JOINT_TARGETS["dual_arm_body"].keys())
        log.info(
            "[frame-vision] p,4 识别前，dual_arm_body 先关节空间运动到 框_Q1"
        )
        if not node.plan_execute_joint_waypoints(
            "dual_arm_body",
            0.2,
            q1_joint_names,
            [框_Q1],
        ):
            log.error("[frame-vision] 运动到 框_Q1 失败，不发送 p,4")
            return False

        vision_result = read_vision_object_pose(
            node,
            log,
            sim_mode=sim_mode,
            trigger_command=FRAME_VISION_TRIGGER_COMMAND,
        )
        if vision_result is None:
            return False

        _, all_xyz_rpy = vision_result
        if not all_xyz_rpy or FRAME_VISION_POSE_KEY not in all_xyz_rpy[0]:
            log.error(
                f"[frame-vision] p,4 视觉结果缺少 "
                f"{FRAME_VISION_POSE_KEY!r}"
            )
            return False

        recognition_pose = make_pose(*all_xyz_rpy[0][FRAME_VISION_POSE_KEY])
        frame_top_pose, frame_pose = deep_frame_poses_from_recognition(
            recognition_pose
        )
        box_top_pose = box_obstacle_top_pose_from_frame_top(frame_top_pose)
        box_center_pose = pose_offset_local(
            box_top_pose,
            0.0,
            0.0,
            -BOX_OBSTACLE_SIZE[2] / 2.0,
        )
        node.publish_deep_frame_vision_tf(
            recognition_pose,
            frame_top_pose,
            frame_pose,
            box_top_pose,
        )
        recognition_rpy = pose_to_xyz_rpy(recognition_pose)[3:]
        frame_top_rpy = pose_to_xyz_rpy(frame_top_pose)[3:]
        frame_rpy = pose_to_xyz_rpy(frame_pose)[3:]
        log.info(
            f"[frame-vision] p,4 识别点 @ {SCENE_FRAME}: "
            f"pos=({recognition_pose.position.x:.4f}, "
            f"{recognition_pose.position.y:.4f}, "
            f"{recognition_pose.position.z:.4f}), "
            f"rpy=({recognition_rpy[0]:.4f}, "
            f"{recognition_rpy[1]:.4f}, {recognition_rpy[2]:.4f})"
        )
        log.info(
            f"[frame-vision] 识别点先绕自身 Z +90°，再按 local offset="
            f"{FRAME_RECOGNITION_TO_TOP_CENTER_LOCAL} m 得到深框顶部空心中心: "
            f"pos=({frame_top_pose.position.x:.4f}, "
            f"{frame_top_pose.position.y:.4f}, "
            f"{frame_top_pose.position.z:.4f}), "
            f"rpy=({frame_top_rpy[0]:.4f}, "
            f"{frame_top_rpy[1]:.4f}, {frame_top_rpy[2]:.4f})"
        )
        log.info(
            f"[frame-vision] 深框实体中心（顶部中心沿自身 -Z "
            f"{FRAME_SIZE[2] / 2.0:.4f} m）: "
            f"pos=({frame_pose.position.x:.4f}, "
            f"{frame_pose.position.y:.4f}, "
            f"{frame_pose.position.z:.4f}), "
            f"rpy=({frame_rpy[0]:.4f}, "
            f"{frame_rpy[1]:.4f}, {frame_rpy[2]:.4f})"
        )
        log.info(
            f"[frame-vision] 长方体上表面中心（相对深框顶部 local offset="
            f"{BOX_OBSTACLE_TOP_FROM_FRAME_TOP_LOCAL} m）: "
            f"pos=({box_top_pose.position.x:.4f}, "
            f"{box_top_pose.position.y:.4f}, "
            f"{box_top_pose.position.z:.4f}); "
            f"碰撞体中心=({box_center_pose.position.x:.4f}, "
            f"{box_center_pose.position.y:.4f}, "
            f"{box_center_pose.position.z:.4f})"
        )

        if frame_added:
            log.info(f"[frame-vision] 移除旧碰撞体「{FRAME_ID}」后更新场景 …")
            if not node.remove_frame():
                log.error("[frame-vision] 移除旧深框场景失败")
                return False
            frame_added = False

        node.configure_deep_frame_from_recognition(recognition_pose)
        log.info(f"[frame-vision] 按 p,4 识别位姿添加「{FRAME_ID}」 …")
        if not node.add_frame():
            log.error("[frame-vision] 添加动态深框场景失败")
            return False
        frame_added = True
        return True

    def run_one_grasp() -> bool | None:
        """执行一轮抓取；True=成功，False=失败，None=用户选择退出。"""
        node.last_pick_failure_reason = None
        # 在机器人运动和视觉识别之前先确认放置区，避免无空位时仍触发识别。
        if not node._wait_for_driver_signal(require_new=True):
            node.last_pick_failure_reason = "no_place_signal"
            return False
        empty_place_slots = [
            slot_name.upper()
            for slot_name in PLACE_SLOT_MASKS
            if node._place_slot_is_empty(slot_name)
        ]
        empty_slots_text = ", ".join(empty_place_slots) if empty_place_slots else "无"
        print(f"{GREEN}[pick] 当前空位: {empty_slots_text}{RESET}")
        if not empty_place_slots:
            node.last_pick_failure_reason = "no_place"
            log.info("[pick] SW1~SW4 均无空位，结束连续抓取，不执行视觉识别")
            return False

        # --- 2. 关节空间运动 ---
        targets = JOINT_TARGETS["dual_arm_body"]
        joint_names = list(targets.keys())
        waypoints = [GRASP_Q1]  # 需要多点时：继续 waypoints.append(q3) ...

        # log.info(f"关节规划组: {"dual_arm"}")
        current = node._get_joints(joint_names)
        if current is None:
            log.error("读取当前关节位置失败，无法规划")
            return False
        log.info("规划前当前关节位置 [rad]:")
        log.info("  " + ", ".join(joint_names))
        log.info("  " + ", ".join(f"{current[name]:.6f}" for name in joint_names))

        if not node.plan_execute_joint_waypoints("dual_arm_body", 0.2, joint_names, waypoints):
            log.error("关节规划/执行失败")
            return False

        time.sleep(1)

        # 视觉识别
        vision_pose = read_vision_object_pose(node, log, sim_mode=sim_mode)
        if vision_pose is None:
            return False
        first_return_modes, all_xyz_rpy = vision_pose

        # 可达性验证：按视觉点顺序，依次验证 arm、waist、body。
        reachable = validate_reachable_grasp(
            node,
            all_xyz_rpy,
            speed_scale=0.2,
            cutoff_joint_names=joint_names,
        )
        if reachable is None:
            log.error(
                "没有找到 IK 可解 + cartesian approach 可行 + "
                "加入两个临时碰撞体后 q_pre 无碰撞的视觉点"
            )
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
            log.info(
                f"可达性选中 {pick_group}，后续直接使用该 body 组规划抓取，"
                f"不再单独动腰和重新视觉识别；"
                f"body_joint2={waist_pick_angle:.6f} rad "
                f"({math.degrees(waist_pick_angle):.2f} deg)"
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
            speed_scale=0.4,
            group=pick_group,
            link=pick_link,
            plan_frame=pick_frame,
            joint_names=pick_joint_names,
            place_speed_scale=0.4,
            cutoff_joint_names=joint_names,
            first_return_mode=first_return_mode,
            waist_moved=waist_moved,
        ):
            log.error(f"{pick_label}抓取失败")
            return False

        return True

    def run_grasps_until_no_empty_slot() -> bool | None:
        """持续执行抓取；检测到 SW1~SW4 均无空位时正常结束。"""
        success_count = 0
        grasp_index = 0
        while rclpy.ok():
            grasp_index += 1
            log.info(
                f"[frame-batch] 开始第 {grasp_index} 次抓取，"
                "抓取前读取最新 SW 空位状态"
            )
            try:
                grasp_result = run_one_grasp()
            finally:
                cleanup_grasp_display()

            if grasp_result is None:
                return None
            if (
                not grasp_result
                and node.last_pick_failure_reason == "no_place"
            ):
                log.info(
                    f"[frame-batch] SW1~SW4 已无空位，连续抓取正常结束；"
                    f"本批次成功 {success_count} 次"
                )
                publish_attempt_result(True, success_count)
                return True

            if grasp_result:
                success_count += 1
            publish_attempt_result(grasp_result, success_count)
            if not grasp_result:
                log.error(
                    f"[frame-batch] 第 {grasp_index} 次抓取失败，停止本批次"
                )
                return False

            log.info(
                f"[frame-batch] 第 {grasp_index} 次抓取完成；"
                f"累计成功 {success_count} 次"
            )

        return None

    def unload_material_slots() -> list[str]:
        """返回当前实时信号中有料的 SW 名称。"""
        return [
            slot_name
            for slot_name in PLACE_SLOT_MASKS
            if node._place_slot_has_material(slot_name)
        ]

    def select_unload_pair(start_index: int = 0) -> tuple[int, str, str] | None:
        """从指定优先级位置开始选择完整料对，返回 (索引, 右SW, 左SW)。"""
        for pair_index in range(max(0, start_index), len(UNLOAD_SLOT_PAIRS)):
            right_slot, left_slot = UNLOAD_SLOT_PAIRS[pair_index]
            if (
                node._place_slot_has_material(right_slot)
                and node._place_slot_has_material(left_slot)
            ):
                return pair_index, right_slot, left_slot
        return None

    def set_unload_tool_power(status: int, stage: str) -> bool:
        """下料流程中依次设置左右末端电源。"""
        action = "上电" if status else "下电"
        if not node.set_tool_power("left", status):
            log.error(f"[unload] {stage}：左臂工具{action}失败")
            return False
        if not node.set_tool_power("right", status):
            log.error(f"[unload] {stage}：右臂工具{action}失败")
            return False
        print(f"{GREEN}[unload] {stage}：左右末端{action}成功{RESET}")
        return True

    def execute_unload_pick_pair(right_slot: str, left_slot: str) -> bool:
        """从一对 SW 同步取料，并反向直线返回对应 yubei。"""
        pair_message = (
            f"[unload] 选择右臂 {right_slot.upper()} + "
            f"左臂 {left_slot.upper()}"
        )
        log.info(pair_message)
        print(f"{GREEN}{pair_message}{RESET}")

        try:
            dual_body_joint_names, yubei_target = (
                make_unload_yubei_joint_target(right_slot, left_slot)
            )
        except (KeyError, ValueError) as exc:
            log.error(f"[unload] 拼接双臂 yubei 目标失败: {exc}")
            return False

        log.info(
            f"[unload] dual_arm_body → yubei："
            f"body_joint1=0, body_joint2=0, "
            f"右={right_slot.upper()}, 左={left_slot.upper()}"
        )
        if not node.plan_execute_joint_waypoints(
            "dual_arm_body",
            UNLOAD_JOINT_SPEED,
            dual_body_joint_names,
            [yubei_target],
        ):
            log.error("[unload] dual_arm_body 到 yubei 失败")
            return False

        left_joint_names = joint_names_for_group("left_arm")
        right_joint_names = joint_names_for_group("right_arm")
        dual_arm_joint_names = joint_names_for_group("dual_arm")
        current = node._get_joints(dual_arm_joint_names, wait_new=True)
        if current is None:
            log.error("[unload] 读取 yubei 处双臂关节失败")
            return False

        left_pose = node._get_link_pose_fk(
            "l_tool",
            joints=current,
            joint_names=left_joint_names,
            plan_frame="l_base_link",
        )
        right_pose = node._get_link_pose_fk(
            "r_tool",
            joints=current,
            joint_names=right_joint_names,
            plan_frame="r_base_link",
        )
        if left_pose is None or right_pose is None:
            log.error("[unload] 计算 yubei 处左右末端 FK 失败")
            return False

        left_approach_distance = (
            UNLOAD_APPROACH_DISTANCE
            + UNLOAD_EXTRA_APPROACH_BY_SLOT[left_slot]
        )
        right_approach_distance = (
            UNLOAD_APPROACH_DISTANCE
            + UNLOAD_EXTRA_APPROACH_BY_SLOT[right_slot]
        )
        left_target = pose_offset_local_z(left_pose, left_approach_distance)
        right_target = pose_offset_local_z(right_pose, right_approach_distance)
        log.info(
            f"[unload] 直线下降距离：左臂 {left_slot.upper()}="
            f"{left_approach_distance * 1000.0:.1f} mm，"
            f"右臂 {right_slot.upper()}="
            f"{right_approach_distance * 1000.0:.1f} mm"
        )

        left_seed = {name: current[name] for name in left_joint_names}
        right_seed = {name: current[name] for name in right_joint_names}
        left_ik = node._solve_ik(
            "left_arm",
            "l_tool",
            left_target,
            left_seed,
            avoid_collisions=UNLOAD_CARTESIAN_AVOID_COLLISIONS,
            plan_frame="l_base_link",
        )
        right_ik = node._solve_ik(
            "right_arm",
            "r_tool",
            right_target,
            right_seed,
            avoid_collisions=UNLOAD_CARTESIAN_AVOID_COLLISIONS,
            plan_frame="r_base_link",
        )
        if left_ik is None or right_ik is None:
            log.error("[unload] 至少一只手臂的直线终点没有 IK")
            return False
        missing_left = [
            name for name in left_joint_names if name not in left_ik
        ]
        missing_right = [
            name for name in right_joint_names if name not in right_ik
        ]
        if missing_left or missing_right:
            log.error(
                f"[unload] IK 返回的关节不完整："
                f"left_missing={missing_left}, right_missing={missing_right}"
            )
            return False
        dual_arm_target = {
            **{name: left_ik[name] for name in left_joint_names},
            **{name: right_ik[name] for name in right_joint_names},
        }
        log.info(
            "[unload] 左右终点 IK 已合成 dual_arm 目标: "
            + ", ".join(
                f"{name}={dual_arm_target[name]:.3f}"
                for name in dual_arm_joint_names
            )
        )

        left_trajectory = node._cartesian_plan(
            "left_arm",
            "l_tool",
            left_target,
            speed_scale=UNLOAD_CARTESIAN_SPEED,
            avoid_collisions=UNLOAD_CARTESIAN_AVOID_COLLISIONS,
            start_joints=current,
            joint_names=left_joint_names,
            plan_frame="l_base_link",
        )
        right_trajectory = node._cartesian_plan(
            "right_arm",
            "r_tool",
            right_target,
            speed_scale=UNLOAD_CARTESIAN_SPEED,
            avoid_collisions=UNLOAD_CARTESIAN_AVOID_COLLISIONS,
            start_joints=current,
            joint_names=right_joint_names,
            plan_frame="r_base_link",
        )
        if left_trajectory is None or right_trajectory is None:
            log.error("[unload] 至少一只手臂的 Cartesian 规划失败")
            return False

        try:
            approach = merge_dual_arm_cartesian_trajectories(
                left_trajectory,
                right_trajectory,
            )
        except ValueError as exc:
            log.error(f"[unload] 合并双臂 Cartesian 轨迹失败: {exc}")
            return False

        try:
            input("按回车执行双臂同步直线取料: ")
        except EOFError:
            pass

        log.info(
            f"[unload] 一次执行双臂沿末端 +Z 直线："
            f"左臂 {left_approach_distance:.3f} m，"
            f"右臂 {right_approach_distance:.3f} m（不考虑碰撞）"
        )
        if not node._execute_traj(approach):
            log.error("[unload] 双臂同步直线取料失败")
            return False

        try:
            input("按回车给左右末端上电: ")
        except EOFError:
            pass

        if not set_unload_tool_power(1, "取料"):
            return False
        time.sleep(UNLOAD_TOOL_SETTLE_SEC)

        log.info("[unload] 反向执行同一条双臂直线轨迹，原路复位到 yubei")
        if not node._execute_traj(node._reverse_trajectory(approach)):
            log.error("[unload] 双臂同步直线复位失败")
            return False

        log.info(
            f"[unload] {right_slot.upper()} + {left_slot.upper()} "
            "双臂取料并复位完成"
        )
        return True

    def move_unload_pair_to_place(
        place_poses: dict[int, Pose],
        left_point: int,
        right_point: int,
    ) -> bool:
        """多构型求解放置目标，并验证/执行双臂末端 +Z 直线放置。"""
        left_joint_names = joint_names_for_group("left_arm")
        right_joint_names = joint_names_for_group("right_arm")
        dual_arm_joint_names = joint_names_for_group("dual_arm")
        left_body_joint_names = joint_names_for_group("left_body")
        dual_body_joint_names = joint_names_for_group("dual_arm_body")
        current_full = node._get_joints(
            dual_body_joint_names,
            wait_new=True,
        )
        if current_full is None:
            log.error("[unload] 读取物料台放置规划起点失败")
            return False

        left_pose = place_poses[left_point]
        right_pose = place_poses[right_point]
        log.info(
            f"[unload] 物料台目标 @ {SCENE_FRAME}："
            f"左点{left_point}=({left_pose.position.x:.4f}, "
            f"{left_pose.position.y:.4f}, {left_pose.position.z:.4f})，"
            f"右点{right_point}=({right_pose.position.x:.4f}, "
            f"{right_pose.position.y:.4f}, {right_pose.position.z:.4f})"
        )

        def plan_place_descent(
            endpoint_state: dict[str, float],
            label: str,
        ) -> RobotTrajectory | None:
            """从候选构型 FK 出发，预检并合并双臂末端局部 +Z 10 cm 路径。"""
            left_start_pose = node._get_link_pose_fk(
                "l_tool",
                joints=endpoint_state,
                plan_frame="l_base_link",
            )
            right_start_pose = node._get_link_pose_fk(
                "r_tool",
                joints=endpoint_state,
                plan_frame="r_base_link",
            )
            if left_start_pose is None or right_start_pose is None:
                log.info(f"[unload] {label}：放置直线起点 FK 失败，换下一构型")
                return None

            left_descent_pose = pose_offset_local_z(
                left_start_pose,
                UNLOAD_PLACE_DESCENT_DISTANCE,
            )
            right_descent_pose = pose_offset_local_z(
                right_start_pose,
                UNLOAD_PLACE_DESCENT_DISTANCE,
            )
            left_descent = node._cartesian_plan(
                "left_arm",
                "l_tool",
                left_descent_pose,
                speed_scale=UNLOAD_CARTESIAN_SPEED,
                avoid_collisions=UNLOAD_CARTESIAN_AVOID_COLLISIONS,
                start_joints=endpoint_state,
                joint_names=left_joint_names,
                plan_frame="l_base_link",
                verbose=False,
            )
            if left_descent is None:
                log.info(
                    f"[unload] {label}：左臂末端局部 +Z "
                    f"{UNLOAD_PLACE_DESCENT_DISTANCE:.3f} m 不可达，换下一构型"
                )
                return None

            right_descent = node._cartesian_plan(
                "right_arm",
                "r_tool",
                right_descent_pose,
                speed_scale=UNLOAD_CARTESIAN_SPEED,
                avoid_collisions=UNLOAD_CARTESIAN_AVOID_COLLISIONS,
                start_joints=endpoint_state,
                joint_names=right_joint_names,
                plan_frame="r_base_link",
                verbose=False,
            )
            if right_descent is None:
                log.info(
                    f"[unload] {label}：右臂末端局部 +Z "
                    f"{UNLOAD_PLACE_DESCENT_DISTANCE:.3f} m 不可达，换下一构型"
                )
                return None

            try:
                merged = merge_dual_arm_cartesian_trajectories(
                    left_descent,
                    right_descent,
                )
            except ValueError as exc:
                log.info(
                    f"[unload] {label}：合并双臂放置直线失败 ({exc})，"
                    "换下一构型"
                )
                return None

            log.info(
                f"[unload] {label}：双臂末端局部 +Z "
                f"{UNLOAD_PLACE_DESCENT_DISTANCE:.3f} m 直线预检通过"
            )
            return merged

        def try_plan_validate_and_execute(
            group: str,
            target: dict[str, float],
            start: dict[str, float],
            label: str,
        ) -> bool | None:
            """候选未通过返回 None；开始执行后返回最终成功与否。"""
            ok, used_ms, trajectory = node.move(
                group,
                [make_joint_constraints(group, target)],
                start=start,
                plan_only=True,
                speed_scale=UNLOAD_PLACE_JOINT_SPEED,
                max_retries=1,
                num_attempts=3,
            )
            log.info(
                f"[unload] {label}: {used_ms:.1f} ms "
                f"({'planned' if ok else 'failed'})"
            )
            if not ok or trajectory is None:
                return None
            if not trajectory.joint_trajectory.points:
                log.error(f"[unload] {label} 成功但没有返回轨迹")
                return None

            # Cartesian 预检必须从联合关节规划的实际末点开始，而不是直接使用
            # IK 目标值，避免规划容差造成直线起点不一致。
            endpoint_state = dict(current_full)
            endpoint_state.update(target)
            final_point = trajectory.joint_trajectory.points[-1]
            endpoint_state.update(
                zip(
                    trajectory.joint_trajectory.joint_names,
                    final_point.positions,
                )
            )
            descent = plan_place_descent(endpoint_state, label)
            if descent is None:
                return None

            log.info(
                f"[unload] 已选中 {label}，执行 {group} 联合轨迹到放置预备位"
            )
            if not node._execute_traj(trajectory):
                log.error(f"[unload] {label} 联合轨迹执行失败")
                return False

            try:
                input(
                    f"已到物料台左点{left_point}/右点{right_point}，"
                    "按回车执行双臂末端局部 +Z 10 cm 同步下降: "
                )
            except EOFError:
                pass

            log.info(
                "[unload] 执行已预检的双臂同步放置直线："
                f"左右末端局部 +Z {UNLOAD_PLACE_DESCENT_DISTANCE:.3f} m"
                f"（avoid_collisions={UNLOAD_CARTESIAN_AVOID_COLLISIONS}）"
            )
            if not node._execute_traj(descent):
                log.error("[unload] 双臂同步直线下降放置失败")
                return False

            try:
                input(
                    f"已下降到物料台左点{left_point}/右点{right_point}，"
                    "按回车给双臂下电: "
                )
            except EOFError:
                pass
            if not set_unload_tool_power(
                0,
                f"物料台左点{left_point}/右点{right_point}放置",
            ):
                return False
            time.sleep(UNLOAD_TOOL_SETTLE_SEC)

            log.info("[unload] 双臂已下电，反向执行同一条同步直线原路返回")
            if not node._execute_traj(node._reverse_trajectory(descent)):
                log.error("[unload] 放置后双臂同步直线复位失败")
                return False
            return True

        # ------------------------------------------------------------------
        # 第一级：身体保持当前位置，只求左右纯臂多组 IK。
        # 目标原始表达在 base_link；纯臂 IK 必须分别转换到 l/r_base_link。
        # ------------------------------------------------------------------
        left_base_pose = node._get_link_pose_fk(
            "l_base_link",
            joints=current_full,
            plan_frame=SCENE_FRAME,
        )
        right_base_pose = node._get_link_pose_fk(
            "r_base_link",
            joints=current_full,
            plan_frame=SCENE_FRAME,
        )
        if left_base_pose is None or right_base_pose is None:
            log.error("[unload] 计算当前左右纯臂基座位姿失败")
            return False
        left_pose_in_arm_base = pose_relative_to_frame(
            left_pose,
            left_base_pose,
        )
        right_pose_in_arm_base = pose_relative_to_frame(
            right_pose,
            right_base_pose,
        )
        log.info(
            "[unload] 纯臂 IK 基座：左目标转换到 l_base_link，"
            "右目标转换到 r_base_link"
        )

        left_arm_solutions = node._solve_ik_candidates_from_seed(
            "left_arm",
            "l_tool",
            left_pose_in_arm_base,
            left_joint_names,
            current_full,
            plan_frame="l_base_link",
            n_attempts=UNLOAD_PLACE_ARM_IK_ATTEMPTS,
            max_solutions=UNLOAD_PLACE_ARM_IK_MAX_SOLUTIONS,
            random_seed=(
                UNLOAD_PLACE_IK_RANDOM_SEED + left_point * 100 + 1
            ),
            perturb_joint_names=left_joint_names,
            avoid_collisions=False,
        )
        right_arm_solutions = node._solve_ik_candidates_from_seed(
            "right_arm",
            "r_tool",
            right_pose_in_arm_base,
            right_joint_names,
            current_full,
            plan_frame="r_base_link",
            n_attempts=UNLOAD_PLACE_ARM_IK_ATTEMPTS,
            max_solutions=UNLOAD_PLACE_ARM_IK_MAX_SOLUTIONS,
            random_seed=(
                UNLOAD_PLACE_IK_RANDOM_SEED + right_point * 100 + 2
            ),
            perturb_joint_names=right_joint_names,
            avoid_collisions=False,
        )
        log.info(
            f"[unload] 纯臂多构型：左点{left_point}="
            f"{len(left_arm_solutions)} 解，右点{right_point}="
            f"{len(right_arm_solutions)} 解"
        )

        pure_pairs = [
            (left_solution, right_solution)
            for left_solution in left_arm_solutions
            for right_solution in right_arm_solutions
        ]
        pure_pairs.sort(
            key=lambda pair: (
                joint_distance_squared(
                    pair[0],
                    current_full,
                    left_joint_names,
                )
                + joint_distance_squared(
                    pair[1],
                    current_full,
                    right_joint_names,
                )
            )
        )
        pure_pairs_to_validate = spread_sorted_candidates(
            pure_pairs,
            UNLOAD_PLACE_MAX_PAIR_PLANS,
        )
        log.info(
            f"[unload] 纯臂共生成 {len(pure_pairs)} 个左右组合，"
            f"按关节距离全范围分布验证 {len(pure_pairs_to_validate)} 组"
        )
        pure_start = {
            name: current_full[name]
            for name in dual_arm_joint_names
        }
        for pair_index, (left_solution, right_solution) in enumerate(
            pure_pairs_to_validate,
            start=1,
        ):
            target = {
                **left_solution,
                **right_solution,
            }
            result = try_plan_validate_and_execute(
                "dual_arm",
                target,
                pure_start,
                (
                    f"dual_arm 纯臂构型 {pair_index}/"
                    f"{len(pure_pairs_to_validate)}"
                ),
            )
            if result is not None:
                return result

        log.warning(
            "[unload] 当前腰部/升降位置下没有可执行的纯臂组合，"
            "开始枚举 left_body 构型"
        )

        # ------------------------------------------------------------------
        # 第二级：左臂先用 left_body 枚举升降+腰部+左臂解。
        # 对每个左侧解固定 body_joint1/2，重新计算 r_base_link，再求右纯臂。
        # 最终目标包含 body+两臂，因此改用 dual_arm_body 联合规划。
        # ------------------------------------------------------------------
        left_body_solutions = node._solve_ik_candidates_from_seed(
            "left_body",
            "l_tool",
            left_pose,
            left_body_joint_names,
            current_full,
            plan_frame=SCENE_FRAME,
            n_attempts=UNLOAD_PLACE_BODY_IK_ATTEMPTS,
            max_solutions=UNLOAD_PLACE_BODY_IK_MAX_SOLUTIONS,
            random_seed=(
                UNLOAD_PLACE_IK_RANDOM_SEED + left_point * 1000 + 3
            ),
            perturb_joint_names=left_body_joint_names,
            avoid_collisions=False,
        )
        if not left_body_solutions:
            log.error(
                f"[unload] 左点{left_point} 即使加入腰部和升降仍无 IK"
            )
            return False

        body_plan_count = 0
        for body_index, left_body_solution in enumerate(
            left_body_solutions,
            start=1,
        ):
            candidate_state = dict(current_full)
            candidate_state.update(left_body_solution)
            lift = left_body_solution["body_joint1"]
            waist = left_body_solution["body_joint2"]
            log.info(
                f"[unload] 左侧 body 构型 {body_index}/"
                f"{len(left_body_solutions)}："
                f"body_joint1={lift:.4f} m，"
                f"body_joint2={waist:.4f} rad"
            )

            candidate_right_base_pose = node._get_link_pose_fk(
                "r_base_link",
                joints=candidate_state,
                plan_frame=SCENE_FRAME,
            )
            if candidate_right_base_pose is None:
                log.warning(
                    f"[unload] body 构型 {body_index} 无法计算 r_base_link，"
                    "尝试下一构型"
                )
                continue
            right_pose_for_body = pose_relative_to_frame(
                right_pose,
                candidate_right_base_pose,
            )
            right_solutions = node._solve_ik_candidates_from_seed(
                "right_arm",
                "r_tool",
                right_pose_for_body,
                right_joint_names,
                candidate_state,
                plan_frame="r_base_link",
                n_attempts=UNLOAD_PLACE_RIGHT_IK_ATTEMPTS_PER_BODY,
                max_solutions=UNLOAD_PLACE_RIGHT_IK_MAX_SOLUTIONS_PER_BODY,
                random_seed=(
                    UNLOAD_PLACE_IK_RANDOM_SEED
                    + right_point * 1000
                    + body_index
                    + 100
                ),
                perturb_joint_names=right_joint_names,
                avoid_collisions=False,
            )
            if not right_solutions:
                log.info(
                    f"[unload] body 构型 {body_index} 下右点{right_point}"
                    "无纯臂 IK，切换下一组腰部/升降"
                )
                continue

            for right_index, right_solution in enumerate(
                right_solutions,
                start=1,
            ):
                if body_plan_count >= UNLOAD_PLACE_MAX_PAIR_PLANS:
                    break
                body_plan_count += 1
                target = {
                    name: (
                        right_solution[name]
                        if name in right_solution
                        else left_body_solution[name]
                    )
                    for name in dual_body_joint_names
                }
                result = try_plan_validate_and_execute(
                    "dual_arm_body",
                    target,
                    current_full,
                    (
                        f"dual_arm_body 左body构型{body_index}/"
                        f"{len(left_body_solutions)}，"
                        f"右解{right_index}/{len(right_solutions)}"
                    ),
                )
                if result is not None:
                    return result
            if body_plan_count >= UNLOAD_PLACE_MAX_PAIR_PLANS:
                break

        log.error(
            f"[unload] 左点{left_point}/右点{right_point} 的纯臂及 "
            "dual_arm_body 多构型全部失败"
        )
        return False

    def run_one_unload() -> bool:
        """最多取两对料：首对放 1/4，实时复查后次对放 2/3。"""
        log.info("[unload] 开始下料前先给左右末端下电")
        if not set_unload_tool_power(0, "下料开始"):
            log.error("[unload] 双臂未全部下电，禁止执行后续动作")
            return False

        if not node._wait_for_driver_signal(require_new=True):
            return False
        material_slots = unload_material_slots()
        material_text = (
            ", ".join(slot.upper() for slot in material_slots)
            if material_slots
            else "无"
        )
        print(f"{GREEN}[unload] 当前有料位置: {material_text}{RESET}")

        selected = select_unload_pair()
        if selected is None:
            log.warning(
                "[unload] 没有可供双臂成对取料的组合："
                "需要 SW1+SW3 或 SW2+SW4 同时有料，本轮不取"
            )
            return True
        pair_cursor, right_slot, left_slot = selected

        # p,2 的识别姿态同时用于构造场景和四个物料台局部放置点。
        vision_result = read_vision_object_pose(
            node,
            log,
            sim_mode=sim_mode,
            trigger_command=UNLOAD_TRIGGER_COMMAND,
        )
        if vision_result is None:
            return False
        _, all_xyz_rpy = vision_result
        if not all_xyz_rpy or UNLOAD_VISION_POSE_KEY not in all_xyz_rpy[0]:
            log.error(
                f"[unload] p,2 视觉结果缺少 {UNLOAD_VISION_POSE_KEY!r}"
            )
            return False
        recognition_values = all_xyz_rpy[0][UNLOAD_VISION_POSE_KEY]
        recognition_pose = make_pose(*recognition_values)
        node.publish_unload_vision_tf(recognition_pose)
        node.publish_unload_table_top_tf(recognition_pose)
        place_poses = make_unload_place_poses(recognition_pose)
        log.info(
            f"[unload] p,2 识别点 @ {SCENE_FRAME}: "
            f"({recognition_pose.position.x:.4f}, "
            f"{recognition_pose.position.y:.4f}, "
            f"{recognition_pose.position.z:.4f})"
        )
        for point_index, point_pose in place_poses.items():
            local_offset = UNLOAD_PLACE_LOCAL_OFFSETS[point_index]
            log.info(
                f"[unload] 物料台点{point_index} table_local={local_offset}, "
                f"base=({point_pose.position.x:.4f}, "
                f"{point_pose.position.y:.4f}, {point_pose.position.z:.4f})"
            )

        unload_scene_added = False
        try:
            if not node.add_unload_scene(recognition_pose):
                log.error("[unload] 添加料台/台面薄长方体/墙状障碍物失败")
                return False
            unload_scene_added = True
            table_top_pose = make_unload_table_top_pose(recognition_pose)
            log.info(
                f"[unload] 已添加取料台 size={UNLOAD_TABLE_SIZE} m, "
                f"top_center=({table_top_pose.position.x:.4f}, "
                f"{table_top_pose.position.y:.4f}, "
                f"{table_top_pose.position.z:.4f})，"
                "姿态相对视觉绕局部 Y=180°、Z=90°；"
                f"台面薄长方体 size={UNLOAD_TABLE_TOP_BOX_SIZE} m，"
                "下表面中心=视觉识别位置；"
                f"障碍物 size={UNLOAD_OBSTACLE_SIZE} m, "
                f"center_y={recognition_pose.position.y + UNLOAD_OBSTACLE_Y_OFFSET:.4f}"
            )

            for batch_index, (left_point, right_point) in enumerate(
                UNLOAD_PLACE_SEQUENCE
            ):
                if batch_index > 0:
                    if not node._wait_for_driver_signal(require_new=True):
                        return False
                    material_slots = unload_material_slots()
                    material_text = (
                        ", ".join(slot.upper() for slot in material_slots)
                        if material_slots
                        else "无"
                    )
                    print(
                        f"{GREEN}[unload] 第一轮放置后仍有料位置: "
                        f"{material_text}{RESET}"
                    )
                    selected = select_unload_pair(pair_cursor + 1)
                    if selected is None:
                        log.info(
                            "[unload] 第一轮完成后没有下一组完整料对，"
                            "不执行第二轮取料"
                        )
                        return True
                    pair_cursor, right_slot, left_slot = selected

                log.info(
                    f"[unload] 第 {batch_index + 1} 轮："
                    f"右臂 {right_slot.upper()}、左臂 {left_slot.upper()} → "
                    f"物料台左点{left_point}/右点{right_point}"
                )
                if not execute_unload_pick_pair(right_slot, left_slot):
                    return False

                if not move_unload_pair_to_place(
                    place_poses,
                    left_point,
                    right_point,
                ):
                    return False

                log.info(
                    f"[unload] 第 {batch_index + 1} 轮放置完成："
                    f"左臂点{left_point}、右臂点{right_point}，"
                    "双臂直线返回放置预备位"
                )

            return True
        finally:
            if unload_scene_added and not node.remove_unload_scene():
                log.error("[unload] 移除料台/台面薄长方体/墙状障碍物失败")

    try:

        # 上位机通信
        # grasp_cmd = node.wait_for_grasp_start()
        # if grasp_cmd is None:
        #     return 1

        log.info("[startup] 键盘输入前先清除本程序管理的场景障碍物 …")
        node.clear_managed_scene_objects()
        frame_added = False

        workflow = prompt_workflow(
            "输入 1：p,4 识别深框并抓取至 SW 无空位；"
            "输入 2：执行原输入 1 的下料流程；输入 q 退出 …"
        )
        while workflow != "q":
            if workflow == "1":
                if not recognize_and_add_deep_frame():
                    publish_attempt_result(False, 0)
                    workflow = prompt_workflow(
                        "深框识别或场景添加失败；输入 1 重试，"
                        "输入 2 执行原下料流程，输入 q 退出 …"
                    )
                    continue

                log.info("按回车继续 …")
                try:
                    input()
                except EOFError:
                    pass

                batch_result = run_grasps_until_no_empty_slot()
                if batch_result is None:
                    code = 0
                    break
                if batch_result:
                    code = 0

            elif workflow == "2":
                # 下料使用 p,2 识别得到的取料台场景，不保留上料深框。
                if frame_added:
                    log.info(f"切换下料，移除上料碰撞体「{FRAME_ID}」…")
                    if not node.remove_frame():
                        log.error("切换下料时移除上料碰撞体失败")
                        workflow = prompt_workflow(
                            "场景切换失败；输入 1 重新识别深框，"
                            "输入 2 重试原下料流程，输入 q 退出 …"
                        )
                        continue
                    frame_added = False

                unload_result = run_one_unload()
                if unload_result:
                    code = 0
            else:
                log.warning(
                    f"未知输入 {workflow!r}：1=深框识别并抓取至SW无空位，"
                    "2=原下料流程，q=退出"
                )

            workflow = prompt_workflow(
                "输入 1：重新识别深框并抓取至 SW 无空位；"
                "输入 2：执行原下料流程；输入 q 退出 …"
            )

        if workflow == "q":
            code = 0

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
