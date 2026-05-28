#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 关节空间规划不到1s，给末端位姿4s左右

# @hello_moveit/scripts/reach.py:1508-1532 修改代码，写个循环随机生成位置和姿态，相对于base_link, EE_POSE2的位置在深框里面，姿态是@reach.py (85-87) roll和pitch可以加上正负0.5，yaw不变，抓取流程如果IK 可解 + cartesian approach EE_POSE2的位姿存到txt文件里面，其他不行的存到另一个txt文件，不需要执行，相当于循环一直检测，每次循环的时候需要在rviz中刷新显示圆柱体
"""
G01 MoveIt 演示脚本

功能（按顺序执行）：
  1. 向规划场景添加半透明「深框」碰撞体（底板 + 四面墙，顶部敞开）
  2. 对 ACTIVE_GROUP 做关节空间规划并执行（例如 dual_arm）
  3. 对 left_body 组做末端位姿（L6）规划并执行
  4. 程序退出时在 finally 中自动移除深框

前提：
  - 已启动 move_group：ros2 launch g01_moveit_config demo.launch.py
  - 本节点与 move_group 在同一 ROS 域

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
"""

from __future__ import annotations

import copy
import math
import multiprocessing
import random
import sys
import time
from typing import Iterable, Sequence

import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration as DurationMsg
from rclpy.duration import Duration
from geometry_msgs.msg import Pose, PoseStamped
from tf2_ros import Buffer, TransformException, TransformListener
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
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

# =============================================================================
# 用户可调参数（改这里即可，无需动下面逻辑）
# =============================================================================

# 第一步：关节空间规划使用哪个 SRDF 组
ACTIVE_GROUP = "dual_arm"

# 第二步：末端位姿规划组与目标（连杆 L6，坐标系 base_link）
POSE_GROUP = "left"

# EE_LINK 必须在 末端linkL6 下游、用 fixed joint 连上去的子 link（例如 l_tool）
EE_LINK = "l_tool"
EE_POSE2 = dict(
    x=-0.6,
    y=-0.35,
    z=0.0,
    roll=-math.pi/2,
    pitch=-math.pi/2,
    yaw=-math.pi,
)

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
        "base_joint1": 1.0,
        "base_joint2": 0.0,
        "body_joint1": 0.0,
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
        "body_joint1": 0.0,
        "body_joint2": -1.313,
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


# 规划器
PLANNER_ID = "RRTConnect"
NUM_ATTEMPTS = 20
PLAN_TIME_SEC = 10.0
MOVE_MAX_RETRIES = 5  # move_action 失败后再试几次（应对 INVALID_MOTION_PLAN 等 OMPL 随机性失败）

# 执行速度缩放（同时用于 velocity / acceleration；范围 0~1，越大越快）
DEFAULT_SPEED_SCALE = 0.5

# 深框障碍物（固定在 world，与机器人 base_link 无关）
SCENE_FRAME = "world"
PLAN_FRAME = "base_link"  # 末端位姿约束坐标系
FRAME_ID = "深框"
FRAME_SIZE = (0.9, 0.9, 0.6)  # 长×宽×高 [m]
WALL_T = 0.02
FRAME_CENTER = (2.0, 0.0, 0.0)  # 底面中心 [m]
FRAME_COLOR = ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.5)

# 位姿标记圆柱（仅 RViz Marker 显示，不进入规划场景碰撞）
CYLINDER_MARKER_ID = "ee_pose_cylinder"
CYLINDER_MARKER_NS = "g01_pose_cylinder"
CYLINDER_MARKER_TOPIC = "g01_pose_cylinder"
CYLINDER_DIAMETER = 0.15   # 直径 [m]
CYLINDER_HEIGHT = 0.06     # 高度 [m]（沿位姿局部 z 轴）
CYLINDER_COLOR = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.45)

# 随机采样时的“物体尺寸”（用于保证物体整体在深框内）
# 默认用 RViz 圆柱体尺寸；如果你的抓取物体更大，请在这里改。
OBJECT_DIAMETER = CYLINDER_DIAMETER
OBJECT_HEIGHT = CYLINDER_HEIGHT
OBJECT_CLEARANCE = 0.01  # 额外留边 [m]

# ROS 接口名（move_group 默认）
SVC_APPLY_SCENE = "apply_planning_scene"
SVC_CARTESIAN_PATH = "compute_cartesian_path"
SVC_COMPUTE_IK = "compute_ik"
SVC_COMPUTE_FK = "compute_fk"
ACT_MOVE_GROUP = "move_action"
ACT_EXEC_TRAJ = "execute_trajectory"

# 末端目标容差（与 MoveGroupInterface 默认一致）
_POS_TOL = 1e-4
_ORI_TOL = 1e-3

# 笛卡尔直线运动参数
CART_EEF_STEP = 0.005     # 服务端 IK 离散步长（m）
CART_MIN_FRACTION = 0.99  # 接受的最小成功比例（<1 表示直线被截断）

# left_body 放置位关节目标 [rad]（键顺序与 JOINT_TARGETS[group] 一致）
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
}

# 抓取流程默认参数
PRE_GRASP_OFFSET = -0.1  # 预备抓取点沿末端坐标系 z 轴外移的距离 [m]
POST_RETURN_Z_OFFSET = 0.1  # 复位后沿末端 +z 轴直线移动距离 [m]
PLACE_SPEED_SCALE = 0.5     # 5/8 OMPL 运动到放置位的速度缩放

# IK 多解枚举参数（抓取流程选 IK 解 + approach 预检用）
IK_N_CANDIDATES = 200          # 总共尝试的 IK 种子数（含 1 次以当前关节为种子）
IK_SEED_PERTURB = math.pi/2     # 随机种子各关节的最大扰动幅度 [rad]，越大解越分散
IK_TIMEOUT_SEC = 0.2          # 每次 /compute_ik 超时（KDL 对边界姿态需更长收敛时间）
IK_RANDOM_SEED = 42           # 让 IK 多解枚举可复现；改成 None 则每次随机


# =============================================================================
# 几何与消息构造（纯函数，无 ROS 通信）
# =============================================================================


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


def make_pose(x: float, y: float, z: float, roll=0.0, pitch=0.0, yaw=0.0) -> Pose:
    """构造 geometry_msgs/Pose。"""
    p = Pose()
    p.position.x, p.position.y, p.position.z = x, y, z
    qx, qy, qz, qw = quat_from_rpy(roll, pitch, yaw)
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = qx, qy, qz, qw
    return p


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
    所有墙板中心相对 FRAME_CENTER 偏移，在 SCENE_FRAME 下发布。
    """
    L, W, H = FRAME_SIZE
    t = WALL_T
    bx, by, bz = FRAME_CENTER

    obj = CollisionObject()
    obj.header.frame_id = SCENE_FRAME
    obj.id = FRAME_ID
    obj.operation = CollisionObject.ADD

    def add_box(dx, dy, dz, ox, oy, oz):
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [dx, dy, dz]
        pose = make_pose(bx + ox, by + oy, bz + oz)
        obj.primitives.append(prim)
        obj.primitive_poses.append(pose)

    add_box(L, W, t, 0, 0, t / 2)                      # 底板
    add_box(t, W, H, L / 2 - t / 2, 0, H / 2)          # +X 侧墙
    add_box(t, W, H, -(L / 2 - t / 2), 0, H / 2)      # -X 侧墙
    add_box(L - 2 * t, t, H, 0, W / 2 - t / 2, H / 2)  # +Y 侧墙
    add_box(L - 2 * t, t, H, 0, -(W / 2 - t / 2), H / 2)  # -Y 侧墙
    return obj


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


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


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


def _quat_conj(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, z, w = q
    return (-x, -y, -z, w)


def _quat_mul(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _rotate_vec_by_quat(
    v: tuple[float, float, float],
    q: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """用四元数 q 旋转向量 v（假设 q 为单位四元数）。"""
    vx, vy, vz = v
    vq = (vx, vy, vz, 0.0)
    rq = _quat_mul(_quat_mul(q, vq), _quat_conj(q))
    return (rq[0], rq[1], rq[2])


def _transform_point(
    x: float,
    y: float,
    z: float,
    tx: float,
    ty: float,
    tz: float,
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> tuple[float, float, float]:
    """对点 p 执行刚体变换：p' = t + R(q) * p。"""
    rx, ry, rz = _rotate_vec_by_quat((x, y, z), (qx, qy, qz, qw))
    return (tx + rx, ty + ry, tz + rz)


def make_joint_constraints(group: str, joints: dict[str, float]) -> Constraints:
    """关节目标（dict）→ MoveIt goal_constraints（每个关节一个 JointConstraint）。"""
    c = Constraints()
    c.name = f"{group}_joint_goal"
    for name, pos in joints.items():
        jc = JointConstraint()
        jc.joint_name = name
        jc.position = pos
        jc.tolerance_above = jc.tolerance_below = 1e-3
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
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._scene_cli = self.create_client(ApplyPlanningScene, SVC_APPLY_SCENE)
        self._cart_cli = self.create_client(GetCartesianPath, SVC_CARTESIAN_PATH)
        self._ik_cli = self.create_client(GetPositionIK, SVC_COMPUTE_IK)
        self._fk_cli = self.create_client(GetPositionFK, SVC_COMPUTE_FK)
        self._move_cli = ActionClient(self, MoveGroup, ACT_MOVE_GROUP)
        self._exec_cli = ActionClient(self, ExecuteTrajectory, ACT_EXEC_TRAJ)
        # 缓存最新 joint_states，供规划起点使用
        self._joints: dict[str, float] = {}
        self._js_count = 0
        self.create_subscription(JointState, "joint_states", self._on_js, 10)
        self._cylinder_marker_pub = self.create_publisher(Marker, CYLINDER_MARKER_TOPIC, 10)
        self._cylinder_marker_ids: dict[str, int] = {}
        self._next_cylinder_marker_id = 0

    def _lookup_transform(self, target_frame: str, source_frame: str, timeout_sec: float = 1.0):
        """获取 source->target 的 TF（用于把 source 下的数据变到 target）。"""
        try:
            # 不同 ROS2 发行版的 can_transform/lookup_transform 参数略有差异；
            # 这里用 lookup_transform + timeout（Duration）作为最兼容写法。
            return self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=float(timeout_sec)),
            )
        except TransformException:
            return None

    def _transform_point_via_tf(
        self, x: float, y: float, z: float, target_frame: str, source_frame: str
    ) -> tuple[float, float, float] | None:
        tf = self._lookup_transform(target_frame, source_frame)
        if tf is None:
            return None
        t = tf.transform.translation
        r = tf.transform.rotation
        return _transform_point(x, y, z, t.x, t.y, t.z, r.x, r.y, r.z, r.w)

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
    ) -> tuple[bool, float, RobotTrajectory | None, int | None]:
        """单次 move_action 调用。返回 (ok, 耗时 ms, trajectory, error_code)。"""
        log = self.get_logger()
        t0 = time.monotonic()
        elapsed_ms = lambda: (time.monotonic() - t0) * 1000.0

        g = MoveGroup.Goal()
        g.request.group_name = group
        g.request.planner_id = PLANNER_ID
        g.request.num_planning_attempts = NUM_ATTEMPTS
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
                group, goal_constraints, start, plan_only, speed_scale
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

        log.info(
            f"[{group}] 关节多点路径: {len(waypoints)} waypoints, speed_scale={_clamp01(speed_scale):.2f}, "
            f"planner={PLANNER_ID}, attempts={NUM_ATTEMPTS}, time={PLAN_TIME_SEC:.1f}s"
        )

        for idx, q in enumerate(waypoints):
            goal = [make_joint_constraints_from_vector(group, joint_names, q)]

            ok, used_ms, _ = self.move(
                group, goal, start=start, plan_only=False, speed_scale=speed_scale
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
        req.jump_threshold = 0.0
        req.avoid_collisions = avoid_collisions

        p = end_pose.position
        if verbose:
            log.info(
                f"[{group}] 笛卡尔直线 {link} @ {plan_frame}: "
                f"end pos({p.x:.3f}, {p.y:.3f}, {p.z:.3f}), "
                f"max_step={eef_step:.4f}, avoid_collisions={avoid_collisions}, "
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
    ):
        """从 target_pose 的多个 IK 解里挑「能直线退回 pre_pose」的一组。

        返回 (q_pre, q_target, approach_traj) 三元组；找不到返回 None。
        approach_traj = reverse( cartesian(start=q_target, end=pre_pose) )
            即真正用来「从 pre_pose 直线接近 target_pose」的轨迹。
        """
        log = self.get_logger()
        candidates = self._solve_ik_multi(
            group, link, target_pose, joint_names, n_candidates, plan_frame=plan_frame
        )
        if not candidates:
            log.error("[grasp-select] target_pose 在该 group 上没有任何 IK 解")
            return None

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

            approach_traj = self._reverse_trajectory(retreat_traj)
            last_pt = retreat_traj.joint_trajectory.points[-1]
            names = list(retreat_traj.joint_trajectory.joint_names)
            q_pre = {n: p for n, p in zip(names, last_pt.positions)}
            log.info(
                f"[grasp-select] 候选 IK {idx + 1}/{len(candidates)}：retreat 可行 ✓ "
                f"(轨迹 {len(retreat_traj.joint_trajectory.points)} 点)"
            )
            return q_pre, q_target, approach_traj

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

    def pick_and_return(
        self,
        target_pose: Pose,
        speed_scale: float,
        group: str,
        link: str,
        plan_frame: str,
        joint_names: Sequence[str],
        place_joints: Sequence[float],
        place_speed_scale: float = PLACE_SPEED_SCALE,
    ) -> bool:
        """抓取流程（IK 多解 + approach 预检 + 放置 + 原路返回）。

        参数：
            target_pose       : 末端抓取位姿（geometry_msgs/Pose，在 plan_frame 下表达）
            speed_scale       : 抓取段速度缩放（0~1）
            group             : SRDF 规划组（如 left_body）
            link              : 末端连杆名（如 L6）
            plan_frame        : 位姿/IK/笛卡尔规划使用的坐标系（如 base_link、world）
            joint_names       : group 内关节名顺序（与 place_joints 一一对应）
            place_joints      : 放置位关节目标 [rad]（向量，顺序同 joint_names）
            place_speed_scale : 5/8 OMPL 到放置位的速度缩放（0~1）
        """
        log = self.get_logger()
        joint_names = list(joint_names)
        if len(place_joints) != len(joint_names):
            log.error(
                f"[pick] place_joints 长度 {len(place_joints)} 与 joint_names 长度 "
                f"{len(joint_names)} 不一致"
            )
            return False

        log.info(
            f"[pick] group={group}, link={link}, plan_frame={plan_frame}"
        )

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
        ok, used_ms, to_pre_traj = self.move(
            group, goal, start=current, plan_only=False, speed_scale=speed_scale
        )
        log.info(f"[pick] OMPL → q_pre: {used_ms:.3f} ms ({'success' if ok else 'failed'})")
        if not ok:
            log.error("[pick] OMPL 到 q_pre 失败")
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

        log.info("[pick] 3/8  到达抓取位置，按回车继续 …")
        try:
            input()
        except EOFError:
            pass

        log.info(
            "[pick] 4/8  第一段复位：反向 approach + 1/8 OMPL → 1/8 初始位置"
        )
        retreat_approach = self._reverse_trajectory(approach_traj)
        if not self._execute_traj(retreat_approach):
            log.error("[pick] 反向播放 approach 失败")
            return False
        retreat_to_start = self._reverse_trajectory(to_pre_traj)
        if not self._execute_traj(retreat_to_start):
            log.error("[pick] 反向播放 1/8 OMPL 回到初始位置失败")
            return False
        log.info(
            "[pick] 第一段复位完成，已回到 1/8 初始位置: "
            + ", ".join(f"{n}={pick_start_joints[n]:.3f}" for n in joint_names)
        )

        log.info("[pick] 5/8  OMPL  → 运动到放置位置")
        current = self._get_joints(joint_names, wait_new=True)
        if current is None:
            log.error("[pick] 读取当前关节失败")
            return False
        goal = [make_joint_constraints_from_vector(group, joint_names, place_joints)]
        ok, used_ms, to_place_traj = self.move(
            group, goal, start=current, plan_only=False, speed_scale=place_speed_scale
        )
        log.info(f"[pick] OMPL → 放置位置: {used_ms:.3f} ms ({'success' if ok else 'failed'})")
        if not ok:
            log.error("[pick] 运动到放置位置失败")
            return False
        if to_place_traj is None or not to_place_traj.joint_trajectory.points:
            log.error("[pick] 5/8 未返回 OMPL 轨迹，无法原路返回")
            return False
        log.info(
            "[pick] 放置位关节: "
            + ", ".join(f"{n}={v:.3f}" for n, v in zip(joint_names, place_joints))
        )

        log.info(
            f"[pick] 6/8  沿末端 +z 轴直线移动 {POST_RETURN_Z_OFFSET:.3f} m "
            f"（笛卡尔，从当前位姿 FK 偏移）"
        )
        current = self._get_joints(joint_names, wait_new=True)
        if current is None:
            log.error("[pick] 读取当前关节失败")
            return False
        ee_pose = self._get_link_pose_fk(
            link, current, joint_names=joint_names, plan_frame=plan_frame
        )
        if ee_pose is None:
            log.error("[pick] FK 读取当前末端位姿失败")
            return False
        offset_pose = pose_offset_local_z(ee_pose, POST_RETURN_Z_OFFSET)
        z_offset_traj = self._cartesian_plan(
            group, link, offset_pose,
            speed_scale=speed_scale,
            joint_names=joint_names,
            plan_frame=plan_frame,
        )
        if z_offset_traj is None:
            log.error("[pick] 6/8 笛卡尔直线规划失败")
            return False
        if not self._execute_traj(z_offset_traj):
            log.error("[pick] 6/8 沿末端 z 轴直线移动执行失败")
            return False

        log.info(
            f"[pick] 7/8  原路返回 ①：反向播放 6/8 → 放置位置 "
            f"（{len(z_offset_traj.joint_trajectory.points)} 点）"
        )
        if not self._execute_traj(self._reverse_trajectory(z_offset_traj)):
            log.error("[pick] 7/8 反向播放 6/8 失败")
            return False

        log.info(
            f"[pick] 8/8  原路返回 ②：反向播放 5/8 → 1/8 初始位置 "
            f"（{len(to_place_traj.joint_trajectory.points)} 点）"
        )
        if not self._execute_traj(self._reverse_trajectory(to_place_traj)):
            log.error("[pick] 8/8 反向播放 5/8 失败")
            return False
        log.info(
            "[pick] 原路返回完成，当前应在 1/8 初始位置: "
            + ", ".join(f"{n}={pick_start_joints[n]:.3f}" for n in joint_names)
        )

        log.info("[pick] 抓取流程完成。")
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

    try:
        if ACTIVE_GROUP not in JOINT_TARGETS:
            log.error(f"未知 ACTIVE_GROUP={ACTIVE_GROUP}，可选: {list(JOINT_TARGETS)}")
            return 1

        # --- 1. 添加深框 ---
        log.info(f"添加碰撞体「{FRAME_ID}」到 {SCENE_FRAME} …")
        if not node.add_frame():
            return 1
        frame_added = True

        # --- 2. 关节空间运动 ---   
        targets = JOINT_TARGETS[ACTIVE_GROUP]
        joint_names = list(targets.keys())
        q1 = [1.25, 0.0, -0.25, 1.1, 
        3.13, -1.419584, 1.578090, 1.370549, 1.672852, 0.588477, 
        3.13, -1.419584, 1.578090, 1.370549, 1.672852, 0.588477,]
        # q2 = [1.25, 0.0, -0.25, 1.1, 
        # 80 * math.pi / 180, -102 * math.pi / 180, -92 * math.pi / 180, 137 * math.pi / 180, -0 * math.pi, -0 * math.pi / 180, 
        # 120 * math.pi / 180, -102 * math.pi / 180, -92 * math.pi / 180, 137 * math.pi / 180, -0 * math.pi, -0 * math.pi / 180]

        waypoints = [q1]  # 需要多点时：继续 waypoints.append(q3) ...

        log.info(f"关节规划组: {ACTIVE_GROUP}")
        current = node._get_joints(joint_names)
        if current is None:
            log.error("读取当前关节位置失败，无法规划")
            return 1
        log.info("规划前当前关节位置 [rad]:")
        log.info("  " + ", ".join(joint_names))
        log.info("  " + ", ".join(f"{current[name]:.6f}" for name in joint_names))

        # --- 3. 循环检测：随机生成 EE_POSE2（相对 base_link），只判断可解并保存 ---
        # 说明：
        # - 深框以 SCENE_FRAME=world 发布；EE_POSE2 需要以 PLAN_FRAME=base_link 表达。
        # - 因此：先在 world 下深框“内腔”采样点，再用 TF(world->base_link 的逆，即 lookup_transform(base_link <- world))
        #   把点变换到 base_link，得到 EE_POSE2 的 xyz。
        trials = 200
        ok_path = "ee_pose2_ok.txt"
        bad_path = "ee_pose2_bad.txt"

        pick_group = POSE_GROUP
        pick_joint_names = joint_names_for_group(pick_group)

        L, W, H = FRAME_SIZE
        t = WALL_T
        cx_w, cy_w, cz_w = FRAME_CENTER
        # 物体整体在深框内：在“内腔”基础上再扣掉物体半径/半高 + 额外 clearance
        r_obj = OBJECT_DIAMETER * 0.5 + OBJECT_CLEARANCE
        hz_obj = OBJECT_HEIGHT * 0.5 + OBJECT_CLEARANCE
        inner_x_min_w = cx_w - (L / 2 - t) + r_obj
        inner_x_max_w = cx_w + (L / 2 - t) - r_obj
        inner_y_min_w = cy_w - (W / 2 - t) + r_obj
        inner_y_max_w = cy_w + (W / 2 - t) - r_obj
        inner_z_min_w = cz_w + t + hz_obj
        inner_z_max_w = cz_w + H - hz_obj

        if inner_x_min_w >= inner_x_max_w or inner_y_min_w >= inner_y_max_w or inner_z_min_w >= inner_z_max_w:
            log.error(
                "深框内腔不足以容纳物体尺寸：请减小 OBJECT_* 或增大 FRAME_SIZE / 调整 WALL_T/FRAME_CENTER"
            )
            return 1

        base_roll = EE_POSE2["roll"]
        base_pitch = EE_POSE2["pitch"]
        base_yaw = EE_POSE2["yaw"]

        log.info(
            f"开始循环检测 {trials} 次：随机生成 EE_POSE2 @ {PLAN_FRAME}（落在深框内部）"
        )
        log.info(f"可行写入: {ok_path}；不可行写入: {bad_path}")

        with open(ok_path, "a", encoding="utf-8") as f_ok, open(bad_path, "a", encoding="utf-8") as f_bad:
            for i in range(trials):
                ep = dict(EE_POSE2)
                # 先在 world 下采样深框内点，再变换到 base_link
                xw = random.uniform(inner_x_min_w, inner_x_max_w)
                yw = random.uniform(inner_y_min_w, inner_y_max_w)
                zw = random.uniform(inner_z_min_w, inner_z_max_w)
                xb_yb_zb = node._transform_point_via_tf(xw, yw, zw, target_frame=PLAN_FRAME, source_frame=SCENE_FRAME)
                if xb_yb_zb is None:
                    log.warning(
                        f"TF 不可用：无法将点从 {SCENE_FRAME} 变换到 {PLAN_FRAME}，本轮记为不可行"
                    )
                    line = (
                        f"{i}\t"
                        f"x={math.nan:.4f}\t"
                        f"y={math.nan:.4f}\t"
                        f"z={math.nan:.4f}\t"
                        f"roll={math.nan:.4f}\t"
                        f"pitch={math.nan:.4f}\t"
                        f"yaw={math.nan:.4f}\n"
                    )
                    f_bad.write(line)
                    f_bad.flush()
                    continue
                ep["x"], ep["y"], ep["z"] = xb_yb_zb
                ep["roll"] = base_roll + random.uniform(-0.5, 0.5)
                ep["pitch"] = base_pitch + random.uniform(-0.5, 0.5)
                ep["yaw"] = base_yaw

                # 每轮刷新 RViz 圆柱体显示
                node.show_cylinder_at_pose(ep, object_id=CYLINDER_MARKER_ID, frame_id=PLAN_FRAME)

                target_pose = pose_from_dict(ep)
                pre_pose = pose_offset_local_z(target_pose, PRE_GRASP_OFFSET)

                # 抓取可行性判断：IK 可解 + cartesian approach 可行（通过 retreat 预检反推 approach）
                feasible = (
                    node._select_feasible_grasp_pair(
                        group=pick_group,
                        link=EE_LINK,
                        target_pose=target_pose,
                        pre_pose=pre_pose,
                        joint_names=pick_joint_names,
                        speed_scale=0.2,
                        plan_frame=PLAN_FRAME,
                    )
                    is not None
                )

                line = (
                    f"{i}\t"
                    f"x={ep['x']:.4f}\t"
                    f"y={ep['y']:.4f}\t"
                    f"z={ep['z']:.4f}\t"
                    f"roll={ep['roll']:.4f}\t"
                    f"pitch={ep['pitch']:.4f}\t"
                    f"yaw={ep['yaw']:.4f}\n"
                )
                if feasible:
                    f_ok.write(line)
                    f_ok.flush()
                else:
                    f_bad.write(line)
                    f_bad.flush()

        code = 0

    finally:
        node.remove_cylinder_at_pose()
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
