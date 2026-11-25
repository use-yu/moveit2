# 测试箱子姿态：
# ros2 run tf2_ros tf2_echo moveit_base_link aruco_33
# [moveit_base_link] 标签 33 xyz (m) (0.6488, 0.0067, 0.7318)，
# 箱中心 xyz (m) (0.8303, 0.0077, 0.7340)，
# 箱体四元数 xyzw (-0.4970, 0.4978, 0.4997, -0.5054)

# [moveit_base_link] 标签 33 xyz (m) (0.5158, -0.0323, 0.6835)，箱中心 xyz (m) (0.6939, -0.0314, 0.6486)，箱体四元数 xyzw (-0.4437, 0.4428, 0.5487, -0.5532)
# [SJ] 标签 33 xyz (m) (0.6118, -0.0323, 0.4295)，箱中心 xyz (m) (0.7899, -0.0314, 0.3946)，箱体四元数 xyzw (-0.4437, 0.4428, 0.5487, -0.5532)
# [YB] 标签 33 xyz (m) (-0.6436, -0.0658, 0.0850)，箱中心 xyz (m) (-0.8251, -0.0593, 0.0859)，箱体四元数 xyzw (-0.0063, -0.7089, 0.0034, -0.7053)
# head 0.75

#!/usr/bin/env python3
"""Detect a box, publish grasp TFs, and plan a staged dual-arm pick with MoveIt."""

import argparse
import bisect
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import math
from math import ceil, isfinite
from pathlib import Path
from queue import Empty, SimpleQueue
import random
import re
import sys
from threading import Thread
import time
from time import monotonic

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, TransformStamped, Twist, Vector3
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    CollisionObject, Constraints, JointConstraint, MoveItErrorCodes, PlanningScene,
    RobotState, RobotTrajectory,
)
from moveit_msgs.srv import (
    ApplyPlanningScene, GetCartesianPath, GetPositionFK, GetPositionIK,
    GetStateValidity,
)
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Float32, Float64MultiArray
from tf2_ros import TransformBroadcaster
from trajectory_msgs.msg import JointTrajectoryPoint


CAMERA_FRAME = "camera_color_optical_frame"
HEAD_LINK = "head"
HEAD_JOINT = "head_joint"
SCENE_FRAME = "moveit_base_link"
TARGET_FRAMES = (SCENE_FRAME, "SJ", "YB")
# --sim：固定视觉结果，坐标均位于 moveit_base_link。
SIM_MARKER_XYZ = (0.5397, -0.0025, 0.7278)
SIM_BOX_XYZ = (0.7208, 0.0063, 0.7375)
SIM_BOX_XYZW = (-0.5173, 0.4981, 0.4773, -0.5064)
BOX_ID = "aruco_box"
BOX_TABLE_ID = "aruco_box_table"
REAR_WALL_ID = "rear_wall"
REAR_WALL_SIZE_M = (0.10, 3.0, 2.5)
REAR_WALL_CLEARANCE_M = 0.5
TF_PUBLISH_PERIOD_SEC = 0.1
#箱子尺寸
BOX_SIZE_M = (0.593, 0.312, 0.395)
BOX_TABLE_SIZE_M = (3 * BOX_SIZE_M[0], BOX_SIZE_M[1], BOX_SIZE_M[2])
# 抓取点相对箱体局部 ±X 和 +Y 边缘的外偏移。
LEFT_GRASP_X_OUTSIDE_M = 0.035
RIGHT_GRASP_X_OUTSIDE_M = 0.035
LEFT_GRASP_Y_OUTSIDE_M = 0.035
RIGHT_GRASP_Y_OUTSIDE_M = 0.055
# 抓取点相对箱体局部 Z 中心的偏移。
LEFT_GRASP_Z_OFFSET_M = -0.026
RIGHT_GRASP_Z_OFFSET_M = -0.025
# 抓取姿态从箱体姿态出发，先绕局部 Y 轴转 180°，再绕旋转后的局部 Z 轴转指定角度。
GRASP_OFFSETS = (
    (
        "l_box_grasp", -1, LEFT_GRASP_X_OUTSIDE_M, LEFT_GRASP_Y_OUTSIDE_M,
        LEFT_GRASP_Z_OFFSET_M, 225.0,
    ),
    (
        "r_box_grasp", 1, RIGHT_GRASP_X_OUTSIDE_M, RIGHT_GRASP_Y_OUTSIDE_M,
        RIGHT_GRASP_Z_OFFSET_M, 135.0,
    ),
)
ARMS = (
    ("left_arm", "l_tool", "l_box_grasp", -1),
    ("right_arm", "r_tool", "r_box_grasp", 1),
)
GRASP_BASE_FRAMES = {frame: f"{link[0]}_base_link" for _, link, frame, _ in ARMS}
ARM_JOINTS = tuple(f"{side}_arm_joint{i}" for side in "lr" for i in range(1, 7))
TORSO_JOINTS = ("body_joint1", "body_joint2")
BOX_RECOGNITION_TORSO_TARGET = {"body_joint1": 0.0, "body_joint2": math.radians(0.0)}
BOX_RECOGNITION_SETTLE_SEC = 1.0
BODY_JOINTS = ("base_joint1", "base_joint2", "body_joint1", "body_joint2", "head_joint")
LINEAR_JOINTS = ("base_joint1", "body_joint1")
# Shared parameters copied from grasp_actor.py input 7.
PLAN_FRAME = "r_base_link"
PLANNER_ID = "RRTConnect"
PLAN_TIME_SEC = 10.0
#ik
IK_TIMEOUT_SEC = 0.005
SVC_COMPUTE_IK = "compute_ik"
SVC_CARTESIAN_PATH = "compute_cartesian_path"
UNLOAD_PLACE_JOINT_SPEED = 0.3
UNLOAD_PLACE_POSITION_TOLERANCE = 0.001
UNLOAD_PLACE_ORIENTATION_TOLERANCE = math.radians(0.1)
UNLOAD_PLACE_ARM_IK_ATTEMPTS = 200
UNLOAD_PLACE_ARM_IK_MAX_SOLUTIONS = 20
UNLOAD_PLACE_BODY_IK_ATTEMPTS = 200
UNLOAD_PLACE_BODY_IK_MAX_SOLUTIONS = 20
UNLOAD_PLACE_RIGHT_IK_ATTEMPTS_PER_BODY = 40
UNLOAD_PLACE_RIGHT_IK_MAX_SOLUTIONS_PER_BODY = 10
UNLOAD_PLACE_MAX_PAIR_PLANS = 200
UNLOAD_PLACE_IK_PERTURB = math.pi
UNLOAD_PLACE_IK_RANDOM_SEED = 20260724
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
UNLOAD_PLACE_CARTESIAN_SPEED = 0.3
UNLOAD_SYNC_SAMPLE_PERIOD = 0.01
UNLOAD_PLACE_SELF_COLLISION_SAMPLE_PERIOD = 0.02
CART_EEF_STEP = 0.005
CART_MIN_FRACTION = 0.99
CART_JUMP_THRESHOLD = 2.0
CART_REVOLUTE_JUMP_THRESHOLD = 0.2
CART_PRISMATIC_JUMP_THRESHOLD = 0.02
JOINT_STATES_TOPIC = "/g01/joint_states"
PREGRASP_OUTSIDE_M = 0.15
LIFT_M = 0.15
CLAW_TOPIC = "/driver_control/claw"
CLAW_OPEN = -45.0
CLAW_CLOSED = 12.0
CLAW_SETTLE_SEC = 1.0
MOBILE_SPEED_TOPIC = "/mobile_set_speed"
MOBILE_SPEED_Y_MPS = 0.05
MOBILE_SPEED_DURATION_SEC = 6
MOBILE_SPEED_RATE_HZ = 10
TRANSFER_REFERENCE_TAG_XZ_M = (0.5407, 0.7426)
# REFERENCE_RETRACT_M = 0.43
REFERENCE_RETRACT_M = 0.43
REFERENCE_LOWER_M = 0.342
POST_RELEASE_LOWER_M = 0.03
SPREAD_M = 0.10
RETREAT_TRAJECTORY_SPEED_SCALE = 2.0
CARTESIAN_STEP_M = CART_EEF_STEP
COLLISION_CHECK_STEP_RAD = 0.025
COLLISION_CHECK_STEP_M = 0.005
SYNC_TRANSLATION_TOLERANCE_M = 0.01
MOVEIT_START_TOLERANCE = 0.01
FOLLOWER_JOINT_STEP_RAD = 0.12
CONTROLLER_PERIOD_SEC = 0.01
# Conservative velocity and acceleration caps from the G01 joint limits.
ARM_PATH_LIMIT = 3.14 * UNLOAD_PLACE_CARTESIAN_SPEED
TORSO_PATH_LIMITS = {
    "body_joint1": 0.1 * UNLOAD_PLACE_CARTESIAN_SPEED,
    "body_joint2": 0.3 * UNLOAD_PLACE_CARTESIAN_SPEED,
}
# 箱体局部 X、Y、Z 的单侧保护量。
PADDING_PER_SIDE_M = (0.02, 0.08, 0.02)
BOX_RPY_DEG = (-1.6, -0.5, -2.0)  # 箱体相对识别到的二维码，绕局部 X、Y、Z 轴旋转（度）
# 二维码位于 X 中线，距箱体 +Y 边缘 6.15 cm、+Z 边缘 0 cm。
MARKER_TO_CENTER_M = (0.0, 0.0615 - BOX_SIZE_M[1] / 2, -BOX_SIZE_M[2] / 2)
# 搬运距离基准按旧贴标位置标定，距 +Y 边缘 18.7 cm、+Z 边缘 1.6 cm。
TRANSFER_REFERENCE_MARKER_TO_CENTER_M = (0.0, 0.187 - BOX_SIZE_M[1] / 2, 0.016 - BOX_SIZE_M[2] / 2)
TIMEOUT_SEC = 3.0
MARKER_SAMPLE_COUNT = 10
GREEN = "\033[32m"
RESET = "\033[0m"


def make_pose(position, rotation):
    x, y, z = map(float, position)
    qx, qy, qz, qw = rotation.as_quat().tolist()
    return Pose(position=Point(x=x, y=y, z=z), orientation=Quaternion(x=qx, y=qy, z=qz, w=qw))


def position_xyz(pose):
    p = pose.position
    return np.array((p.x, p.y, p.z))


def pose_rotation(pose):
    q = pose.orientation
    return Rotation.from_quat((q.x, q.y, q.z, q.w))


def pose_matrix(pose):
    matrix = np.eye(4)
    matrix[:3, :3] = pose_rotation(pose).as_matrix()
    matrix[:3, 3] = position_xyz(pose)
    return matrix


def load_extrinsics(arm, base_frame):
    path = Path(__file__).with_name(f"g01_handeye_{arm}.json")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        transform = np.asarray(result["calibration"]["best"]["T_base_camera"], dtype=float)
        head_angle = float(result["head_joint_rad"])
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"无法从 {path} 读取完整的手眼标定结果：{error}") from error

    if (result.get("arm") != arm or result.get("base_frame") != base_frame
            or result.get("camera_frame") != CAMERA_FRAME):
        raise RuntimeError(f"{path} 的坐标系应为 {CAMERA_FRAME} → {base_frame}。")
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1]) or not isfinite(head_angle)):
        raise RuntimeError(f"{path} 的外参矩阵或头部关节角度无效。")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1):
        raise RuntimeError(f"{path} 的外参旋转矩阵无效。")
    return transform, head_angle


def shifted_pose(pose, offset):
    shifted = deepcopy(pose)
    x, y, z = (position_xyz(pose) + offset).tolist()
    shifted.position = Point(x=x, y=y, z=z)
    return shifted


def robot_state(joints):
    return RobotState(
        is_diff=True,
        joint_state=JointState(name=list(joints), position=list(joints.values())),
    )


def point_time_ns(point):
    return point.time_from_start.sec * 1_000_000_000 + point.time_from_start.nanosec


def trajectory_state(start, trajectory, point_index):
    joints = trajectory.joint_trajectory
    return {**start, **dict(zip(joints.joint_names, joints.points[point_index].positions))}


def scale_joint_trajectory_speed(trajectory, speed_scale):
    for point in trajectory.joint_trajectory.points:
        elapsed_ns = round(point_time_ns(point) / speed_scale)
        sec, nanosec = divmod(elapsed_ns, 1_000_000_000)
        point.time_from_start = Duration(sec=sec, nanosec=nanosec)
        point.velocities = [value * speed_scale for value in point.velocities]
        point.accelerations = [value * speed_scale ** 2 for value in point.accelerations]


def reverse_joint_trajectory(trajectory):
    result = deepcopy(trajectory)
    points = result.joint_trajectory.points
    end_ns = point_time_ns(points[-1])
    points.reverse()
    for point in points:
        sec, nanosec = divmod(end_ns - point_time_ns(point), 1_000_000_000)
        point.time_from_start = Duration(sec=sec, nanosec=nanosec)
        point.velocities = [-value for value in point.velocities]
    return result


def collision_step_sizes(names):
    return np.array([
        COLLISION_CHECK_STEP_M if name in LINEAR_JOINTS else COLLISION_CHECK_STEP_RAD
        for name in names
    ])


def time_shared_body_path(times, positions):
    """Fit a smooth joint path and time it for the G01 10 ms torso controllers."""
    names = TORSO_JOINTS + ARM_JOINTS
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    slopes = PchipInterpolator(times, positions).derivative()(times)
    slopes[[0, -1]] = 0.0
    path = CubicHermiteSpline(times, positions, slopes)

    # Cubic velocity extrema occur at each span boundary or where acceleration is zero.
    a, b, c, _ = path.c
    span = np.diff(times)[:, None]
    extrema = np.divide(-b, 3 * a, out=np.zeros_like(a), where=a != 0)
    extrema = np.clip(extrema, 0.0, span)
    velocity = np.maximum.reduce([
        np.abs(c),
        np.abs((3 * a * span + 2 * b) * span + c),
        np.abs((3 * a * extrema + 2 * b) * extrema + c),
    ])
    acceleration = np.maximum(np.abs(2 * b), np.abs(6 * a * span + 2 * b))
    limits = np.array([TORSO_PATH_LIMITS.get(name, ARM_PATH_LIMIT) for name in names])
    scale = max(1.0, float(np.max(velocity / limits)), float(np.sqrt(np.max(acceleration / limits))))
    sample_count = ceil(times[-1] * scale / CONTROLLER_PERIOD_SEC)
    trajectory = RobotTrajectory()
    trajectory.joint_trajectory.header.frame_id = SCENE_FRAME
    trajectory.joint_trajectory.joint_names = list(names)
    # Positions only: the synchronized samples are also used by the standard arm controllers.
    samples = path(np.linspace(0.0, times[-1], sample_count + 1))
    period_ns = round(CONTROLLER_PERIOD_SEC * 1_000_000_000)
    for index, values in enumerate(samples):
        sec, nanosec = divmod(index * period_ns, 1_000_000_000)
        trajectory.joint_trajectory.points.append(JointTrajectoryPoint(
            positions=values.tolist(), time_from_start=Duration(sec=sec, nanosec=nanosec)
        ))
    return trajectory


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


class BoxNode(Node):
    def __init__(self, sim=False):
        super().__init__("aruco_box_obstacle")
        self.sim = sim
        self.marker_id = self.declare_parameter("marker_id", 33).value
        if self.sim:
            self.marker_id = 33
            self.get_logger().info("--sim：使用标签 33 的固定视觉数据，跳过识别定位；运动仍由 MoveIt 执行")
        self.tag_frame = f"aruco_{self.marker_id}"
        self.tf_broadcaster = TransformBroadcaster(self)
        self.frame_poses = {}
        self.box_collision = None
        self.grasp_boxes = {}
        self.box_rotation_in_tag = Rotation.from_euler("XYZ", BOX_RPY_DEG, degrees=True)
        self.box_center_in_tag = self.box_rotation_in_tag.apply(MARKER_TO_CENTER_M)
        edge_x = BOX_SIZE_M[0] / 2
        edge_y = BOX_SIZE_M[1] / 2
        self.grasps_in_box = {
            frame: (
                (side * (edge_x + x_outside), edge_y + y_outside, z_offset),
                Rotation.from_euler("YZ", (180.0, z_rotation_deg), degrees=True),
            )
            for frame, side, x_outside, y_outside, z_offset, z_rotation_deg in GRASP_OFFSETS
        }
        self.tf_timer = self.create_timer(TF_PUBLISH_PERIOD_SEC, self.publish_tfs)
        self.scene_client = self.create_client(ApplyPlanningScene, "apply_planning_scene")
        self.ik_client = self.create_client(GetPositionIK, "compute_ik")
        self.plan_client = ActionClient(self, MoveGroup, "move_action")
        self.cartesian_client = self.create_client(GetCartesianPath, "compute_cartesian_path")
        self.validity_client = self.create_client(GetStateValidity, "check_state_validity")
        self.fk_client = self.create_client(GetPositionFK, "compute_fk")
        self.execution_client = ActionClient(self, ExecuteTrajectory, "execute_trajectory")
        self.claw_publisher = self.create_publisher(Float32, CLAW_TOPIC, 10)
        self.mobile_speed_publisher = self.create_publisher(Twist, MOBILE_SPEED_TOPIC, 10)

    def read_marker(self):
        """Aggregate fresh detections of the target marker."""
        samples = []
        mismatched_frame = None

        def on_poses(msg):
            nonlocal mismatched_frame
            if len(msg.data) % 8:
                return
            source_frame = msg.layout.dim[0].label if msg.layout.dim else ""
            if source_frame != CAMERA_FRAME:
                mismatched_frame = source_frame or "未标记"
                return
            # /aruco_poses packs [x, y, z, qx, qy, qz, qw, id] for each tag.
            for start in range(0, len(msg.data), 8):
                *pose, marker_id = msg.data[start:start + 8]
                if marker_id != self.marker_id or not all(map(isfinite, pose)):
                    continue
                try:
                    rotation = Rotation.from_quat(pose[3:])
                except ValueError:
                    continue
                samples.append((pose[:3], rotation))
                break

        subscription = self.create_subscription(Float64MultiArray, "/aruco_poses", on_poses, 1)
        try:
            deadline = monotonic() + TIMEOUT_SEC
            while len(samples) < MARKER_SAMPLE_COUNT and rclpy.ok():
                remaining = deadline - monotonic()
                if remaining <= 0:
                    break
                rclpy.spin_once(self, timeout_sec=remaining)
        finally:
            self.destroy_subscription(subscription)

        if len(samples) < MARKER_SAMPLE_COUNT:
            if mismatched_frame is not None:
                self.get_logger().error(f"/aruco_poses 的坐标系为 {mismatched_frame}，要求 {CAMERA_FRAME}。")
            self.get_logger().error(
                f"/aruco_poses 中标签 {self.marker_id} 的有效位姿只收到 "
                f"{len(samples)}/{MARKER_SAMPLE_COUNT} 条"
            )
            return None

        positions, rotations = zip(*samples)
        position = np.median(positions, axis=0)
        rotation = Rotation.concatenate(rotations).mean()
        self.get_logger().info(f"已汇总标签 {self.marker_id} 的 {MARKER_SAMPLE_COUNT} 条有效位姿")
        return make_pose(position, rotation)

    def marker_and_box(self, transform):
        tag_xyz = transform[:3, 3]
        tag_rotation = Rotation.from_matrix(transform[:3, :3])
        center = tag_xyz + tag_rotation.apply(self.box_center_in_tag)
        return make_pose(tag_xyz, tag_rotation), make_pose(center, tag_rotation * self.box_rotation_in_tag)

    def box_poses(self, camera_marker, joints):
        log = self.get_logger()
        links = [HEAD_LINK, *GRASP_BASE_FRAMES.values(), *TARGET_FRAMES[1:]]
        current = self.forward_poses(joints, links)
        if current is None:
            log.error("无法使用实测关节状态计算头部和双臂基座的 FK。")
            return None

        scene_from_head = pose_matrix(current[HEAD_LINK])
        camera_from_marker = pose_matrix(camera_marker)
        arm_poses = {}
        scene_from_markers = {}
        grasp_boxes = {}
        for grasp_frame, base_frame in GRASP_BASE_FRAMES.items():
            arm = "left" if base_frame == "l_base_link" else "right"
            try:
                calibrated_base_from_camera, calibrated_angle = load_extrinsics(arm, base_frame)
            except RuntimeError as error:
                log.error(str(error))
                return None
            calibrated = self.forward_poses({**joints, HEAD_JOINT: calibrated_angle}, [HEAD_LINK, base_frame])
            if calibrated is None:
                log.error(f"无法计算 {base_frame} 在标定头部角度 {calibrated_angle:.6f} rad 时的 FK。")
                return None

            scene_from_base = pose_matrix(current[base_frame])
            base_from_head = np.linalg.solve(scene_from_base, scene_from_head)
            calibrated_base_from_head = np.linalg.solve(pose_matrix(calibrated[base_frame]),
                                                        pose_matrix(calibrated[HEAD_LINK]))
            head_delta = Rotation.from_matrix(calibrated_base_from_head[:3, :3].T @ base_from_head[:3, :3]).magnitude()
            if abs(head_delta - abs(joints[HEAD_JOINT] - calibrated_angle)) > 1e-3:
                log.error(f"{base_frame} 的 FK 头部转角与实测关节角不一致，请检查 MoveIt 的 head_joint 模型。")
                return None

            head_from_camera = np.linalg.solve(calibrated_base_from_head, calibrated_base_from_camera)
            base_from_marker = base_from_head @ head_from_camera @ camera_from_marker
            arm_poses[base_frame] = self.marker_and_box(base_from_marker)
            scene_from_markers[base_frame] = scene_from_base @ base_from_marker
            grasp_boxes[grasp_frame] = self.marker_and_box(scene_from_markers[base_frame])[1]
            log.info(f"{base_frame} 使用头部标定角 {calibrated_angle:.6f} rad，实测角 {joints[HEAD_JOINT]:.6f} rad。")

        # 保留单个场景碰撞箱和标签 TF，采用右臂标定结果。规划目标使用各臂自己的结果。
        scene_from_marker = scene_from_markers["r_base_link"]
        poses = {SCENE_FRAME: self.marker_and_box(scene_from_marker), **arm_poses}
        for frame in TARGET_FRAMES[1:]:
            poses[frame] = self.marker_and_box(np.linalg.solve(pose_matrix(current[frame]), scene_from_marker))
        return poses, grasp_boxes

    def simulated_box_poses(self):
        box_rotation = Rotation.from_quat(SIM_BOX_XYZW)
        # 未提供标签朝向，沿用箱体相对标签的标定关系反推。
        marker_rotation = box_rotation * self.box_rotation_in_tag.inv()
        return {SCENE_FRAME: (
            make_pose(SIM_MARKER_XYZ, marker_rotation),
            make_pose(SIM_BOX_XYZ, box_rotation),
        )}

    def add_box(self):
        if not self.scene_client.wait_for_service(timeout_sec=TIMEOUT_SEC):
            self.get_logger().error("MoveIt 的 apply_planning_scene 服务不可用")
            return

        if self.sim:
            poses = self.simulated_box_poses()
            grasp_boxes = {frame: poses[SCENE_FRAME][1] for frame in GRASP_BASE_FRAMES}
        else:
            detection = self.read_marker()
            if detection is None:
                return
            joints = self.read_joints()
            if joints is None:
                return
            result = self.box_poses(detection, joints)
            if result is None:
                return
            poses, grasp_boxes = result
        for frame, (marker_pose, pose) in poses.items():
            tag = marker_pose.position
            p = pose.position
            q = pose.orientation
            print(
                f"{GREEN}[{frame}] 标签 {self.marker_id} xyz (m) "
                f"({tag.x:.4f}, {tag.y:.4f}, {tag.z:.4f})，"
                f"箱中心 xyz (m) ({p.x:.4f}, {p.y:.4f}, {p.z:.4f})，"
                f"箱体四元数 xyzw ({q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f}){RESET}"
            )

        if self.apply_box(*poses[SCENE_FRAME], grasp_boxes):
            print(f"{GREEN}已更新箱体 {BOX_ID} 和桌子 {BOX_TABLE_ID}，发布 TF {SCENE_FRAME} → "
                  f"{', '.join(self.frame_poses)}{RESET}")

    def apply_box(self, marker_pose, box_pose, grasp_boxes):
        collision = CollisionObject(id=BOX_ID, operation=CollisionObject.ADD)
        collision.header.frame_id = SCENE_FRAME
        collision.pose = box_pose
        dimensions = [
            round(size + 2 * padding, 3) for size, padding in zip(BOX_SIZE_M, PADDING_PER_SIDE_M)
        ]
        collision.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=dimensions)]
        primitive_pose = make_pose((0.0, 0.0, 0.0), Rotation.identity())
        collision.primitive_poses = [primitive_pose]

        table = CollisionObject(id=BOX_TABLE_ID, operation=CollisionObject.ADD)
        table.header.frame_id = SCENE_FRAME
        # 箱体局部 +Y 朝上，桌面与箱体不含保护量的底面贴合。
        table_center_y = -(BOX_SIZE_M[1] + BOX_TABLE_SIZE_M[1]) / 2
        table.pose = shifted_pose(box_pose, pose_rotation(box_pose).apply((0.0, table_center_y, 0.0)))
        table.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=list(BOX_TABLE_SIZE_M))]
        table.primitive_poses = [primitive_pose]

        if not self.apply_scene(collision, table):
            return False
        self.box_collision = collision
        self.grasp_boxes = grasp_boxes
        self.frame_poses = {self.tag_frame: marker_pose}
        for frame, (position, rotation) in self.grasps_in_box.items():
            center = position_xyz(grasp_boxes[frame])
            box_rotation = pose_rotation(grasp_boxes[frame])
            self.frame_poses[frame] = make_pose(
                center + box_rotation.apply(position), box_rotation * rotation
            )
        self.publish_tfs()
        return True

    def publish_tfs(self):
        if not self.frame_poses:
            return
        stamp = self.get_clock().now().to_msg()
        transforms = []
        for frame, pose in self.frame_poses.items():
            position = pose.position
            transform = TransformStamped(child_frame_id=frame)
            transform.header.frame_id = SCENE_FRAME
            transform.header.stamp = stamp
            transform.transform.translation = Vector3(x=position.x, y=position.y, z=position.z)
            transform.transform.rotation = pose.orientation
            transforms.append(transform)
        self.tf_broadcaster.sendTransform(transforms)

    def wait_result(self, future, label, timeout):
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done():
            self.get_logger().error(f"{label}超时")
        elif future.exception() is not None:
            self.get_logger().error(f"{label}失败：{future.exception()}")
        else:
            return future.result()
        return None

    def call_service(self, client, request, timeout=TIMEOUT_SEC):
        if not client.wait_for_service(timeout_sec=TIMEOUT_SEC):
            self.get_logger().error(f"服务 {client.srv_name} 不可用")
            return None
        future = client.call_async(request)
        result = self.wait_result(future, client.srv_name, timeout)
        if not future.done():
            future.cancel()
        return result

    def _spin_until(self, future, timeout):
        result = self.wait_result(future, "MoveIt 服务响应", timeout)
        if not future.done():
            future.cancel()
        return result is not None

    def read_joints(self):
        required = ARM_JOINTS + BODY_JOINTS
        joints = None

        def on_joints(msg):
            nonlocal joints
            positions = dict(zip(msg.name, msg.position))
            if all(name in positions and isfinite(positions[name]) for name in required):
                joints = {name: positions[name] for name in required}

        subscription = self.create_subscription(
            JointState, JOINT_STATES_TOPIC, on_joints, qos_profile_sensor_data
        )
        try:
            deadline = monotonic() + TIMEOUT_SEC
            while joints is None and rclpy.ok() and (remaining := deadline - monotonic()) > 0:
                rclpy.spin_once(self, timeout_sec=remaining)
        finally:
            self.destroy_subscription(subscription)
        if joints is None:
            self.get_logger().error(f"{JOINT_STATES_TOPIC} 没有完整的 G01 关节状态")
        return joints

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
        if not self.ik_client.wait_for_service(timeout_sec=5.0):
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
        req.ik_request.timeout = Duration(sec=secs, nanosec=nsecs)

        fut = self.ik_client.call_async(req)
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

    def plan_pregrasp(self, current, target, group, names):
        # The plan-only branch of grasp_actor._move_once.
        log = self.get_logger()
        if not self.plan_client.wait_for_server(timeout_sec=10.0):
            log.error("动作 move_action 不可用")
            return None
        goal = MoveGroup.Goal()
        goal.planning_options.plan_only = True
        plan = goal.request
        plan.group_name = group
        plan.planner_id = PLANNER_ID
        plan.start_state = robot_state(current)
        plan.goal_constraints = [make_joint_constraints(group, {name: target[name] for name in names})]
        plan.num_planning_attempts = 50
        plan.allowed_planning_time = PLAN_TIME_SEC
        plan.max_velocity_scaling_factor = UNLOAD_PLACE_JOINT_SPEED
        plan.max_acceleration_scaling_factor = UNLOAD_PLACE_JOINT_SPEED
        send_future = self.plan_client.send_goal_async(goal)
        if not self._spin_until(send_future, 15.0) or not send_future.result().accepted:
            log.error("move_action 目标被拒绝或超时")
            return None
        result_future = send_future.result().get_result_async()
        if not self._spin_until(result_future, PLAN_TIME_SEC + 30.0):
            log.error("move_action 结果超时")
            return None
        result = result_future.result()
        code = result.result.error_code.val if result.result else None
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            error_name = _moveit_error_name(code) if code is not None else "None"
            log.error(f"move_action 状态失败: {result.status} (GoalStatus)，"
                      f" MoveItErrorCodes={code}({error_name})")
            return None
        if code != MoveItErrorCodes.SUCCESS:
            error_name = _moveit_error_name(code) if code is not None else "None"
            log.error(f"MoveIt 错误码: {code} ({error_name})")
            return None
        trajectory = result.result.planned_trajectory
        return trajectory if self.valid_trajectory(trajectory, names) else None

    def valid_trajectory(self, trajectory, names):
        joints = trajectory.joint_trajectory
        valid = (
            sorted(joints.joint_names) == sorted(names)
            and len(joints.points) >= 2
            and point_time_ns(joints.points[0]) >= 0
            and all(len(p.positions) == len(names) and all(map(isfinite, p.positions))
                    for p in joints.points)
            and all(point_time_ns(b) > point_time_ns(a) for a, b in zip(joints.points, joints.points[1:]))
        )
        if not valid:
            self.get_logger().error("MoveIt 返回了无效轨迹")
        return valid

    def plan_cartesian_line(
        self,
        group: str,
        link: str,
        end_pose: Pose,
        *,
        speed_scale: float = 0.2,
        avoid_collisions: bool = True,
        eef_step: float = CART_EEF_STEP,
        min_fraction: float = CART_MIN_FRACTION,
        start_joints: Mapping[str, float] | None = None,
        joint_names: Sequence[str] | None = None,
        plan_frame: str = PLAN_FRAME,
        num_attempts: int = 1,
        verbose: bool = True,
    ) -> RobotTrajectory | None:
        """规划一条从指定关节起点到末端目标位姿的笛卡尔直线。

        参数：
            group:
                SRDF 规划组，例如 ``left_arm``、``right_body``。
            link:
                沿直线运动的末端 link，例如 ``l_tool`` 或 ``r_tool``。
            end_pose:
                直线终点位姿，必须在 ``plan_frame`` 中表达。
            speed_scale:
                轨迹速度缩放。Humble 的 GetCartesianPath 没有速度字段，因此
                本函数统一缩放时间戳、速度和加速度后再返回轨迹。
            avoid_collisions:
                是否在 Cartesian 离散 IK 求解过程中检查规划场景碰撞。
            eef_step:
                末端直线的最大离散步长，单位 m；必须大于 0。
            min_fraction:
                可接受的最小完成比例，范围 0~1；低于该值视为失败。
            start_joints:
                起点关节 ``{name: pos}``。省略时读取 ``joint_names`` 对应的最新
                joint_states。服务会先对该起点做 FK，因此无需另传 start_pose。
                显式传入时可在不移动机器人的情况下预演任意起点。
            joint_names:
                未指定 ``start_joints`` 时要读取的关节名；省略则使用
                ``POSE_START_JOINTS``。
            plan_frame:
                ``end_pose`` 所属的参考坐标系。
            num_attempts:
                compute_cartesian_path 的最大调用次数，至少为 1。任意一次达到
                ``min_fraction`` 即返回。
            verbose:
                是否打印详细日志。批量 IK 候选预检时可关闭，避免刷屏。

        返回：
            成功时返回已按 ``speed_scale`` 缩放的 ``RobotTrajectory``；所有
            尝试均失败时返回 None。轨迹可直接执行、缓存或反向播放。

        关闭碰撞检查只影响本次服务请求，不会修改全局规划场景。
        """
        log = self.get_logger()
        total_t0 = time.monotonic()

        if eef_step <= 0.0:
            log.error(f"[{group}] eef_step 必须大于 0，当前={eef_step}")
            return None
        if not 0.0 <= min_fraction <= 1.0:
            log.error(f"[{group}] min_fraction 必须在 0~1，当前={min_fraction}")
            return None
        attempts = max(1, int(num_attempts))

        if not self.cartesian_client.wait_for_service(timeout_sec=10.0):
            log.error(f"服务 {SVC_CARTESIAN_PATH} 不可用")
            return None

        start = dict(start_joints) if start_joints is not None else self.read_joints()
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
                f"jump={CART_JUMP_THRESHOLD:.1f}/"
                f"{CART_REVOLUTE_JUMP_THRESHOLD:.2f}rad, "
                f"speed_scale={_clamp01(speed_scale):.2f}, attempts={attempts}"
            )

        trajectory: RobotTrajectory | None = None
        last_fraction = 0.0
        last_code: int | None = None
        for attempt in range(1, attempts + 1):
            attempt_t0 = time.monotonic()
            fut = self.cartesian_client.call_async(req)
            if not self._spin_until(fut, 15.0):
                if verbose:
                    log.warning(
                        f"[{group}] compute_cartesian_path "
                        f"第 {attempt}/{attempts} 次超时"
                    )
                continue

            res = fut.result()
            last_fraction = res.fraction
            last_code = res.error_code.val
            if verbose:
                log.info(
                    f"[{group}] cartesian planning {attempt}/{attempts}: "
                    f"{(time.monotonic() - attempt_t0) * 1000.0:.3f} ms, "
                    f"fraction={last_fraction:.3f}, "
                    f"error_code={last_code}({_moveit_error_name(last_code)})"
                )
            if (
                last_fraction >= min_fraction
                and last_code == MoveItErrorCodes.SUCCESS
            ):
                trajectory = res.solution
                break

        if trajectory is None:
            if verbose:
                error_name = (
                    _moveit_error_name(last_code)
                    if last_code is not None
                    else "timeout"
                )
                log.error(
                    f"[{group}] 笛卡尔直线 {attempts} 次规划均未达标："
                    f"fraction={last_fraction:.3f} < {min_fraction}, "
                    f"error_code={last_code}({error_name}), "
                    f"total={(time.monotonic() - total_t0) * 1000.0:.3f} ms"
                )
            return None

        scale = _clamp01(speed_scale)
        if scale > 0.0 and abs(scale - 1.0) > 1e-6:
            scale_joint_trajectory_speed(trajectory, scale)

        return trajectory

    def plan_cartesian(self, start, targets, cache=None):
        paths = []
        for group, link, frame, _ in ARMS:
            names = [name for name in ARM_JOINTS if name.startswith(link[0] + "_")]
            key = (group, tuple(start[name] for name in (*names, *BODY_JOINTS)))
            if cache is not None and key in cache:
                path = cache[key]
            else:
                path = self.plan_cartesian_line(
                    group, link, targets[frame], speed_scale=UNLOAD_PLACE_CARTESIAN_SPEED,
                    avoid_collisions=False, start_joints=start, plan_frame=SCENE_FRAME, verbose=False,
                )
                if path is not None and not self.valid_trajectory(path, names):
                    path = None
                if cache is not None:
                    cache[key] = path
            if path is None:
                return None
            paths.append(path)
        try:
            return merge_dual_arm_cartesian_trajectories(*paths)
        except ValueError as exc:
            self.get_logger().error(str(exc))
            return None

    def forward_poses(self, state, links):
        request = GetPositionFK.Request()
        request.header.frame_id = SCENE_FRAME
        request.fk_link_names = links
        request.robot_state = robot_state(state)
        response = self.call_service(self.fk_client, request)
        if response is None or response.error_code.val != MoveItErrorCodes.SUCCESS:
            return None
        poses = {
            link: stamped.pose for link, stamped in zip(response.fk_link_names, response.pose_stamped)
            if stamped.header.frame_id == SCENE_FRAME
        }
        return poses if all(link in poses for link in links) else None

    def plan_shared_body_line(self, start, starts, targets, lead, follow):
        lead_group, lead_link, lead_frame, _ = lead
        follow_group, follow_link, follow_frame, _ = follow
        lead_group = f"{lead_group.removesuffix('_arm')}_body"
        lead_names = TORSO_JOINTS + tuple(name for name in ARM_JOINTS if name[0] == lead_link[0])
        path = self.plan_cartesian_line(
            lead_group, lead_link, targets[lead_frame], speed_scale=UNLOAD_PLACE_CARTESIAN_SPEED,
            start_joints=start, plan_frame=SCENE_FRAME, verbose=False,
        )
        if path is None or not self.valid_trajectory(path, lead_names):
            return None

        joints = path.joint_trajectory
        names = joints.joint_names
        follow_names = tuple(name for name in ARM_JOINTS if name[0] == follow_link[0])
        output_names = TORSO_JOINTS + ARM_JOINTS
        lead_origin = position_xyz(starts[lead_frame])
        lead_line = position_xyz(targets[lead_frame]) - lead_origin
        follow_origin = position_xyz(starts[follow_frame])
        follow_line = position_xyz(targets[follow_frame]) - follow_origin
        origin_ns = point_time_ns(joints.points[0])
        steps = collision_step_sizes(names)
        if np.any(np.abs(np.array(joints.points[0].positions) - [start[name] for name in names]) > steps):
            return None
        state = dict(start)
        times = [0.0]
        positions = [[state[name] for name in output_names]]
        progress = 0.0

        for index, (before, after) in enumerate(zip(joints.points, joints.points[1:]), 1):
            a = np.array(before.positions)
            delta = np.array(after.positions) - a
            samples = max(1, ceil(np.max(np.abs(delta) / steps)))
            start_ns = point_time_ns(before) - origin_ns
            delta_ns = point_time_ns(after) - point_time_ns(before)
            for sample in range(1, samples + 1):
                fraction = sample / samples
                state.update(zip(names, (a + delta * fraction).tolist()))
                fk = self.forward_poses(state, [lead_link])
                if fk is None:
                    return None
                distance = position_xyz(fk[lead_link]) - lead_origin
                phase = float(np.clip(distance @ lead_line / (lead_line @ lead_line), 0.0, 1.0))
                if np.linalg.norm(distance - phase * lead_line) > CARTESIAN_STEP_M:
                    return None
                if phase + 0.02 < progress:
                    return None
                progress = max(progress, phase)
                if index == len(joints.points) - 1 and sample == samples:
                    progress = 1.0
                follower_pose = make_pose(
                    follow_origin + progress * follow_line, pose_rotation(targets[follow_frame])
                )
                solution = self._solve_ik(follow_group, follow_link, follower_pose, state, plan_frame=SCENE_FRAME)
                if solution is None:
                    return None
                if not all(name in solution and isfinite(solution[name])
                           and abs(solution[name] - state[name]) <= FOLLOWER_JOINT_STEP_RAD
                           for name in follow_names):
                    return None
                state.update({name: solution[name] for name in follow_names})
                times.append((start_ns + delta_ns * fraction) * 1e-9)
                positions.append([state[name] for name in output_names])
        return time_shared_body_path(times, positions)

    def cartesian_candidates(self, start, starts, targets, synchronized=False, cache=None):
        path = self.plan_cartesian(start, targets, cache)
        if path is not None and self.validate_path(path, start, (starts, targets), synchronized):
            yield path
        for lead, follow in (ARMS, ARMS[::-1]):
            path = self.plan_shared_body_line(start, starts, targets, lead, follow)
            if path is not None and self.validate_path(path, start, (starts, targets), synchronized):
                yield path

    def check_cartesian_pair(self, state, starts, targets, synchronized, endpoint):
        poses = self.forward_poses(state, [link for _, link, _, _ in ARMS])
        if poses is None:
            return False
        distances = []
        for _, link, frame, _ in ARMS:
            actual = poses[link]
            start = position_xyz(starts[frame])
            end = position_xyz(targets[frame])
            line = end - start
            progress = endpoint
            if progress is None:
                progress = np.clip((position_xyz(actual) - start) @ line / (line @ line), 0.0, 1.0)
            nearest = start + progress * line
            if (np.linalg.norm(position_xyz(actual) - nearest) > CARTESIAN_STEP_M
                    or (pose_rotation(actual).inv() * pose_rotation(targets[frame])).magnitude() > 0.03):
                return False
            distances.append((position_xyz(actual) - start) @ line / np.linalg.norm(line))
        return not synchronized or abs(distances[0] - distances[1]) <= SYNC_TRANSLATION_TOLERANCE_M

    def validate_path(self, trajectory, start, cartesian=None, synchronized=False):
        joints = trajectory.joint_trajectory
        names = joints.joint_names
        points = joints.points
        if cartesian is not None:
            stride = round(UNLOAD_PLACE_SELF_COLLISION_SAMPLE_PERIOD / UNLOAD_SYNC_SAMPLE_PERIOD)
            points = [*points[:-1:stride], points[-1]]
        points = [np.array(p.positions) for p in points]
        request = GetStateValidity.Request()
        # Leaving group_name empty checks collisions for the whole robot.
        step_sizes = collision_step_sizes(names)
        samples = [points[0]]
        for a, b in zip(points, points[1:]):
            count = max(1, ceil(np.max(np.abs(b - a) / step_sizes)))
            samples.extend(a + (b - a) * (j / count) for j in range(1, count + 1))
        for index, point in enumerate(samples):
            state = {**start, **dict(zip(names, point.tolist()))}
            request.robot_state = robot_state(state)
            result = self.call_service(self.validity_client, request)
            if result is None or not result.valid:
                self.get_logger().error(f"双臂合并轨迹状态 {index} 未通过场景碰撞检查")
                return False
            if cartesian is not None:
                endpoint = 0.0 if index == 0 else 1.0 if index == len(samples) - 1 else None
                if not self.check_cartesian_pair(state, *cartesian, synchronized, endpoint):
                    self.get_logger().error(f"双臂笛卡尔状态 {index} 偏离直线或双臂不同步")
                    return False
        return True

    def restore_box_collision(self, collision):
        if self.apply_scene(collision):
            return True
        self.box_collision = None
        self.get_logger().error("箱体碰撞体恢复失败，请输入 1 重新识别")
        return False

    def pregrasp_candidates(self, current, pregrasp):
        log = self.get_logger()
        left_names, right_names = ARM_JOINTS[:6], ARM_JOINTS[6:]
        left_pose, right_pose = (pregrasp[frame] for _, _, frame, _ in ARMS)
        left_point, right_point = 1, 2
        log.info(f"[unload] 左右纯臂 IK 使用 {SCENE_FRAME} 目标和完整预计关节状态")
        left_solutions = self._solve_ik_candidates_from_seed(
            "left_arm", "l_tool", left_pose, left_names, current, plan_frame=SCENE_FRAME,
            n_attempts=UNLOAD_PLACE_ARM_IK_ATTEMPTS, max_solutions=UNLOAD_PLACE_ARM_IK_MAX_SOLUTIONS,
            random_seed=UNLOAD_PLACE_IK_RANDOM_SEED + left_point * 100 + 1,
            perturb_joint_names=left_names, avoid_collisions=True,
        )
        right_solutions = self._solve_ik_candidates_from_seed(
            "right_arm", "r_tool", right_pose, right_names, current, plan_frame=SCENE_FRAME,
            n_attempts=UNLOAD_PLACE_ARM_IK_ATTEMPTS, max_solutions=UNLOAD_PLACE_ARM_IK_MAX_SOLUTIONS,
            random_seed=UNLOAD_PLACE_IK_RANDOM_SEED + right_point * 100 + 2,
            perturb_joint_names=right_names, avoid_collisions=True,
        )
        log.info(f"[unload] 纯臂多构型：左点{left_point}={len(left_solutions)} 解，"
                 f"右点{right_point}={len(right_solutions)} 解")
        missing_sides = [side for side, solutions in (("左臂", left_solutions), ("右臂", right_solutions))
                         if not solutions]
        if missing_sides:
            log.warning(f"[unload] {'、'.join(missing_sides)}纯臂无 IK，"
                        "纯臂组合无法成立，将加入腰部和升降")
        pairs = [{**left, **right} for left in left_solutions for right in right_solutions]
        pairs.sort(key=lambda pair: joint_distance_squared(pair, current, ARM_JOINTS))
        selected = spread_sorted_candidates(pairs, UNLOAD_PLACE_MAX_PAIR_PLANS)
        log.info(f"[unload] 纯臂共生成 {len(pairs)} 个左右组合，"
                 f"按关节距离全范围分布验证 {len(selected)} 组")
        for index, target in enumerate(selected, start=1):
            yield "dual_arm", ARM_JOINTS, target, f"dual_arm 纯臂构型 {index}/{len(selected)}"

        log.warning("[unload] 纯臂候选均未通过验证，加入腰部和升降，开始求 left_body 构型")
        lead_names = TORSO_JOINTS + left_names
        left_body_solutions = self._solve_ik_candidates_from_seed(
            "left_body", "l_tool", left_pose, lead_names, current, plan_frame=SCENE_FRAME,
            n_attempts=UNLOAD_PLACE_BODY_IK_ATTEMPTS, max_solutions=UNLOAD_PLACE_BODY_IK_MAX_SOLUTIONS,
            random_seed=UNLOAD_PLACE_IK_RANDOM_SEED + left_point * 1000 + 3,
            perturb_joint_names=lead_names, avoid_collisions=True,
        )
        if not left_body_solutions:
            log.error(f"[unload] 左点{left_point} 即使加入腰部和升降仍无 IK")
            return
        body_plan_count = 0
        for body_index, left in enumerate(left_body_solutions, start=1):
            state = {**current, **left}
            log.info(f"[unload] 左侧 body 构型 {body_index}/{len(left_body_solutions)}："
                     f"body_joint1={state['body_joint1']:.4f} m，body_joint2={state['body_joint2']:.4f} rad")
            right_solutions = self._solve_ik_candidates_from_seed(
                "right_arm", "r_tool", right_pose, right_names, state, plan_frame=SCENE_FRAME,
                n_attempts=UNLOAD_PLACE_RIGHT_IK_ATTEMPTS_PER_BODY,
                max_solutions=UNLOAD_PLACE_RIGHT_IK_MAX_SOLUTIONS_PER_BODY,
                random_seed=UNLOAD_PLACE_IK_RANDOM_SEED + right_point * 1000 + body_index + 100,
                perturb_joint_names=right_names, avoid_collisions=True,
            )
            if not right_solutions:
                log.info(f"[unload] body 构型 {body_index} 下右点{right_point}"
                         "无纯臂 IK，切换下一组腰部/升降")
            for right_index, right in enumerate(right_solutions, start=1):
                label = (f"dual_arm_body 左侧构型{body_index}/{len(left_body_solutions)}，"
                         f"右候选解{right_index}/{len(right_solutions)}")
                yield "dual_arm_body", TORSO_JOINTS + ARM_JOINTS, {**left, **right}, label
                body_plan_count += 1
                if body_plan_count >= UNLOAD_PLACE_MAX_PAIR_PLANS:
                    return

    def check_pregrasp_endpoint(self, endpoint, targets, label):
        actual = self.forward_poses(endpoint, [link for _, link, _, _ in ARMS])
        if actual is None:
            self.get_logger().error(f"[unload] {label} 规划阶段记录抱箱预备点 FK 失败")
            return False
        for (_, link, frame, _), side in zip(ARMS, ("左臂", "右臂")):
            pose, expected = actual[link], targets[frame]
            position_error = np.linalg.norm(position_xyz(pose) - position_xyz(expected))
            orientation_error = (pose_rotation(pose).inv() * pose_rotation(expected)).magnitude()
            self.get_logger().info(f"[unload] {label} {side}规划终点 FK 误差："
                                   f"位置={position_error * 1000.0:.3f} mm，"
                                   f"姿态={math.degrees(orientation_error):.4f}°")
            if (not isfinite(position_error) or not isfinite(orientation_error)
                    or position_error > UNLOAD_PLACE_POSITION_TOLERANCE
                    or orientation_error > UNLOAD_PLACE_ORIENTATION_TOLERANCE):
                self.get_logger().warning(f"[unload] {label} {side}规划终点位姿误差超限，拒绝该候选")
                return False
        return True

    def box_transfer_moves(self, box_pose):
        tag_x, _, tag_z = position_xyz(box_pose) - pose_rotation(box_pose).apply(
            TRANSFER_REFERENCE_MARKER_TO_CENTER_M
        )
        reference_x, reference_z = TRANSFER_REFERENCE_TAG_XZ_M
        retract = REFERENCE_RETRACT_M + (tag_x - reference_x)
        lower = REFERENCE_LOWER_M + (tag_z - reference_z)
        if not all(isfinite(value) and value > 0 for value in (retract, lower)):
            self.get_logger().error("根据箱体位置计算的后退或下降距离无效")
            return None
        self.get_logger().info(
            f"旧贴标等效位置 X={tag_x:.4f} m、Z={tag_z:.4f} m，"
            f"抱箱后退 {retract:.3f} m、下降 {lower:.3f} m"
        )
        return (
            ("双臂抬升 15 cm", ((0.0, 0.0, LIFT_M),) * len(ARMS)),
            (f"双臂后退 {retract * 100:.1f} cm", ((-retract, 0.0, 0.0),) * len(ARMS)),
            (f"双臂下降 {lower * 100:.1f} cm", ((0.0, 0.0, -lower),) * len(ARMS)),
            (f"双臂再下降 {POST_RELEASE_LOWER_M * 100:g} cm",
             ((0.0, 0.0, -POST_RELEASE_LOWER_M),) * len(ARMS)),
            ("双臂张开 10 cm", ((0.0, SPREAD_M, 0.0), (0.0, -SPREAD_M, 0.0))),
        )

    def plan_box_transfers(self, start, starts, moves):
        if not moves:
            return []
        label, offsets = moves[0]
        targets = {
            frame: shifted_pose(starts[frame], offset)
            for (_, _, frame, _), offset in zip(ARMS, offsets)
        }
        for trajectory in self.cartesian_candidates(start, starts, targets, synchronized=True):
            endpoint = trajectory_state(start, trajectory, -1)
            rest = self.plan_box_transfers(endpoint, targets, moves[1:])
            if rest is not None:
                return [(label, trajectory, start, (starts, targets)), *rest]
        return None

    def plan_box_pick(self, current, collision):
        log = self.get_logger()
        moves = self.box_transfer_moves(collision.pose)
        if moves is None:
            return None
        grasp = {frame: self.frame_poses[frame] for _, _, frame, _ in ARMS}
        pregrasp = {
            frame: shifted_pose(grasp[frame], pose_rotation(self.grasp_boxes[frame]).apply(
                (side * PREGRASP_OUTSIDE_M, 0.0, 0.0)))
            for _, _, frame, side in ARMS
        }
        cache = {}
        for group, names, target, label in self.pregrasp_candidates(current, pregrasp):
            candidate = {**current, **target}
            lines = self.plan_box_lines(candidate, collision, pregrasp, grasp, moves, cache)
            if self.box_collision is None:
                return None
            if lines is None:
                log.info(f"[unload] {label}：至少一段双臂直线不可行，换下一构型")
                continue
            log.info(f"[unload] {label}：左右直线和同步碰撞检查均通过，已缓存同步轨迹")
            started = monotonic()
            preparation = self.plan_pregrasp(current, target, group, names)
            used_ms = (monotonic() - started) * 1000
            log.info(f"[unload] {label}: {used_ms:.1f} ms "
                     f"({'planned' if preparation is not None else 'failed'})")
            if preparation is None or not self.validate_path(preparation, current):
                continue
            endpoint = trajectory_state(current, preparation, -1)
            endpoint_delta = max(abs(endpoint[name] - candidate[name]) for name in names)
            if endpoint_delta > 1e-4:
                log.info(f"[unload] {label}：OMPL 末点与 IK 目标最大偏差 "
                         f"{endpoint_delta:.6f}，按实际末点更新直线缓存")
                lines = self.plan_box_lines(endpoint, collision, pregrasp, grasp, moves, cache)
                if self.box_collision is None:
                    return None
                if lines is None:
                    continue
            if not self.check_pregrasp_endpoint(endpoint, pregrasp, label):
                continue
            log.info(f"[unload] {label}：IK、左右 Cartesian、OMPL 均通过，已缓存 {len(lines) + 1} 段抱箱轨迹")
            return [
                ("移动到预备位", preparation, current, None),
                *lines,
            ]
        log.error("[unload] 左点1/右点2 的纯臂及 dual_arm_body 多构型全部失败")
        return None

    def plan_box_lines(self, start, collision, pregrasp, grasp, moves, cache):
        lines = None
        # Cartesian services read the live scene. Restore the box before any execution.
        try:
            if not self.apply_scene(CollisionObject(id=BOX_ID, operation=CollisionObject.REMOVE)):
                return None
            for approach in self.cartesian_candidates(start, pregrasp, grasp, cache=cache):
                at_grasp = trajectory_state(start, approach, -1)
                transfers = self.plan_box_transfers(at_grasp, grasp, moves)
                if transfers is not None:
                    lines = [("双臂直线接近", approach, start, (pregrasp, grasp)), *transfers]
                    break
        finally:
            restored = self.restore_box_collision(collision)
        return lines if restored else None

    def execute_trajectory(self, trajectory):
        send = self.execution_client.send_goal_async(ExecuteTrajectory.Goal(trajectory=trajectory))
        goal = self.wait_result(send, "发送轨迹", TIMEOUT_SEC)
        if goal is None:
            # If acceptance arrives late, request cancellation without starting another stage.
            def cancel_late_goal(future):
                if rclpy.ok() and not future.cancelled() and future.exception() is None:
                    late_goal = future.result()
                    if late_goal is not None and late_goal.accepted:
                        late_goal.cancel_goal_async()

            send.add_done_callback(cancel_late_goal)
            return False
        if not goal.accepted:
            self.get_logger().error("MoveIt 拒绝轨迹执行")
            return False
        duration = point_time_ns(trajectory.joint_trajectory.points[-1]) * 1e-9
        try:
            result = self.wait_result(goal.get_result_async(), "执行轨迹", 2 * duration + 10.0)
        except (KeyboardInterrupt, ExternalShutdownException):
            if rclpy.ok():
                goal.cancel_goal_async()
            raise
        if result is None:
            self.wait_result(goal.cancel_goal_async(), "取消轨迹", TIMEOUT_SEC)
            return False
        if (result.status != GoalStatus.STATUS_SUCCEEDED
                or result.result.error_code.val != MoveItErrorCodes.SUCCESS):
            self.get_logger().error(
                f"轨迹执行失败，状态 {result.status}，错误码 {result.result.error_code.val}"
            )
            return False
        return True

    def prepare_transfer_for_execution(self, stages, actual):
        label, trajectory, start, cartesian = stages[0]
        expected = trajectory_state(start, trajectory, 0)

        def start_error(path, measured):
            joints = path.joint_trajectory
            return max(
                (abs(measured[name] - position), name)
                for name, position in zip(joints.joint_names, joints.points[0].positions)
            )

        error, joint = start_error(trajectory, actual)
        if error <= MOVEIT_START_TOLERANCE:
            # 规划阶段已验证整段轨迹，执行前只确认实测起点。
            return stages
        if any(abs(actual[name] - value) > (0.005 if name in LINEAR_JOINTS else 0.02)
               for name, value in expected.items()):
            self.get_logger().error(f"{label}前实测关节状态已偏离规划起点，停止执行")
            return None

        self.get_logger().info(
            f"{label}起点关节 {joint} 偏差 {error:.5f} 超过 MoveIt 容差 "
            f"{MOVEIT_START_TOLERANCE:.3f}，从当前实测姿态重新规划剩余直线"
        )
        poses = self.forward_poses(actual, [link for _, link, _, _ in ARMS])
        if poses is None:
            self.get_logger().error(f"无法计算{label}起点实测末端位姿")
            return None
        starts = {frame: poses[link] for _, link, frame, _ in ARMS}
        moves = [
            (step_label, tuple(position_xyz(to_poses[frame]) - position_xyz(from_poses[frame])
                               for _, _, frame, _ in ARMS))
            for step_label, _, _, (from_poses, to_poses) in stages
        ]
        refreshed = self.plan_box_transfers(actual, starts, moves)
        if refreshed is None:
            self.get_logger().error("从实测姿态重新规划剩余直线失败")
            return None
        latest = self.read_joints()
        if latest is None:
            return None
        error, joint = start_error(refreshed[0][1], latest)
        if error > MOVEIT_START_TOLERANCE:
            self.get_logger().error(f"重规划后关节 {joint} 起点仍偏差 {error:.5f}，停止执行")
            return None
        return refreshed

    def prepare_box_recognition(self, confirm):
        if self.sim:
            return True
        target = BOX_RECOGNITION_TORSO_TARGET

        def torso_at_target(state):
            return state is not None and all(
                abs(state[name] - value) <= (0.005 if name in LINEAR_JOINTS else MOVEIT_START_TOLERANCE)
                for name, value in target.items()
            )

        current = self.read_joints()
        if current is None:
            return False
        if not torso_at_target(current):
            if not self.execution_client.wait_for_server(timeout_sec=TIMEOUT_SEC):
                self.get_logger().error("execute_trajectory 动作服务不可用")
                return False
            trajectory = self.plan_pregrasp(current, target, "body", TORSO_JOINTS)
            if trajectory is None:
                self.get_logger().error("无法规划到箱体识别位（升降 0 m、腰部 10°）")
                return False
            if confirm is not None and not confirm("按回车移动到箱体识别位（升降 0 m、腰部 10°），输入 q 退出："):
                return False
            actual = self.read_joints()
            expected = trajectory_state(current, trajectory, 0)
            if actual is None or any(abs(actual[name] - expected[name]) > MOVEIT_START_TOLERANCE
                                     for name in TORSO_JOINTS):
                self.get_logger().error("识别位轨迹起点与实测躯干位置不一致")
                return False
            if not self.validate_path(trajectory, actual) or not self.execute_trajectory(trajectory):
                return False

        if not self.spin_for(BOX_RECOGNITION_SETTLE_SEC):
            return False
        if not torso_at_target(self.read_joints()):
            self.get_logger().error("实测升降或腰部未到达箱体识别位")
            return False
        return True

    def spin_for(self, seconds):
        deadline = monotonic() + seconds
        while rclpy.ok() and (remaining := deadline - monotonic()) > 0:
            rclpy.spin_once(self, timeout_sec=min(TF_PUBLISH_PERIOD_SEC, remaining))
        return rclpy.ok()

    def command_claw(self, value):
        self.claw_publisher.publish(Float32(data=value))
        self.get_logger().info(f"已向 {CLAW_TOPIC} 发布夹爪指令 {value:g}")

    def move_mobile_after_placement(self):
        speed = Twist(linear=Vector3(y=MOBILE_SPEED_Y_MPS))
        period = 1.0 / MOBILE_SPEED_RATE_HZ
        start = monotonic()
        try:
            for index in range(MOBILE_SPEED_DURATION_SEC * MOBILE_SPEED_RATE_HZ):
                send_at = start + index * period
                if not self.spin_for(send_at - monotonic()):
                    return False
                if monotonic() < send_at + period:
                    self.mobile_speed_publisher.publish(speed)
            return self.spin_for(start + MOBILE_SPEED_DURATION_SEC - monotonic())
        finally:
            if rclpy.ok():
                self.mobile_speed_publisher.publish(Twist())

    def return_box(self, stages, confirm):
        # 倒放实际执行的六段直线，最后停在取箱预备位。
        return_steps = (
            ("双臂合拢", None),
            (f"双臂向上抬 {POST_RELEASE_LOWER_M * 100:g} cm", ("闭合夹爪", CLAW_CLOSED)),
            ("双臂向上抬至搬运高度", None),
            ("双臂向前", ("松开夹爪", CLAW_OPEN)),
            (f"双臂向下 {LIFT_M * 100:g} cm", None),
            ("双臂张开至取箱预备位", None),
        )
        for (_, forward, start, _), (label, claw_step) in zip(reversed(stages[1:]), return_steps, strict=True):
            if all(name in forward.joint_trajectory.joint_names for name in TORSO_JOINTS):
                label += "（含腰部和升降）"
            if confirm is not None and not confirm(f"按回车{label}，输入 q 退出："):
                return
            actual = self.read_joints()
            expected = trajectory_state(start, forward, -1)
            if actual is None or any(
                abs(actual[name] - value) > (0.005 if name in LINEAR_JOINTS else MOVEIT_START_TOLERANCE)
                for name, value in expected.items()
            ):
                self.get_logger().error(f"{label}的实际关节状态与缓存轨迹起点不一致，停止返程")
                return
            if not self.execute_trajectory(reverse_joint_trajectory(forward)):
                return
            print(f"{GREEN}已完成{label}{RESET}")
            if claw_step is not None:
                claw_label, value = claw_step
                if confirm is not None and not confirm(f"按回车{claw_label}，输入 q 退出："):
                    return
                self.command_claw(value)
                if not self.spin_for(CLAW_SETTLE_SEC):
                    return

    def pick_box(self, confirm=None):
        collision = self.box_collision
        if collision is None:
            self.get_logger().error("请先输入 1 识别并添加箱体")
            return
        self.command_claw(CLAW_OPEN)
        current = self.read_joints()
        if current is None:
            return
        stages = self.plan_box_pick(current, collision)
        if stages is None:
            self.get_logger().error("抱箱和放箱路径未全部通过规划检查")
            return
        print(f"{GREEN}预备、接近、抬升、后退、下降、再下降、张开共 {len(stages)} 段路径均已通过规划检查{RESET}")
        if not self.execution_client.wait_for_server(timeout_sec=TIMEOUT_SEC):
            self.get_logger().error("execute_trajectory 动作服务不可用")
            return

        for index in range(len(stages)):
            label, trajectory, start, cartesian = stages[index]
            if cartesian is not None and all(name in trajectory.joint_trajectory.joint_names for name in TORSO_JOINTS):
                label += "（含腰部和升降）"
            if confirm is not None and not confirm(f"按回车{label}，输入 q 退出："):
                return
            actual = self.read_joints()
            expected = trajectory_state(start, trajectory, 0)
            if actual is None or any(
                abs(actual[name] - value) > (0.005 if name in LINEAR_JOINTS else 0.02)
                for name, value in expected.items()
            ):
                self.get_logger().error(f"{label}的实际关节状态与规划起点不一致")
                return
            if index == 1 and not self.apply_scene(CollisionObject(id=BOX_ID, operation=CollisionObject.REMOVE)):
                self.restore_box_collision(collision)
                return
            if index >= 2:
                remaining_stages = self.prepare_transfer_for_execution(stages[index:], actual)
                if remaining_stages is None:
                    return
                stages[index:] = remaining_stages
                trajectory = stages[index][1]
            if index == 1:
                # The box is out of the scene after contact; require a fresh detection for a new pick.
                self.box_collision = None
                self.grasp_boxes.clear()
            if index == 2:
                # Stop publishing the original grasp locations as the box starts moving.
                self.frame_poses.clear()
            if index == 3:
                # 只加速抱箱后退，保留原始缓存给返程倒放。
                trajectory = deepcopy(trajectory)
                scale_joint_trajectory_speed(trajectory, RETREAT_TRAJECTORY_SPEED_SCALE)
            if not self.execute_trajectory(trajectory):
                return
            print(f"{GREEN}已完成{label}{RESET}")
            if index == 2:
                # 抬升成功后合爪，再后退。
                self.command_claw(CLAW_CLOSED)
            elif index == 4:
                # 到达放箱高度后松爪，再下降并张开双臂。
                self.command_claw(CLAW_OPEN)
            if index in (2, 4) and not self.spin_for(CLAW_SETTLE_SEC):
                return

        if confirm is not None and not confirm(f"按回车移动底盘 {MOBILE_SPEED_DURATION_SEC:g} 秒，输入 q 退出："):
            return
        if self.move_mobile_after_placement():
            self.return_box(stages, confirm)

    def add_rear_wall(self):
        if not self.scene_client.wait_for_service(timeout_sec=TIMEOUT_SEC):
            self.get_logger().error("MoveIt 的 apply_planning_scene 服务不可用")
            return

        thickness, width, height = REAR_WALL_SIZE_M
        wall = CollisionObject(id=REAR_WALL_ID, operation=CollisionObject.ADD)
        wall.header.frame_id = SCENE_FRAME
        wall.pose = make_pose(
            (-REAR_WALL_CLEARANCE_M - thickness / 2, 0.0, height / 2), Rotation.identity()
        )
        wall.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[thickness, width, height])]
        wall.primitive_poses = [Pose(orientation=Quaternion(w=1.0))]
        if self.apply_scene(wall):
            print(f"{GREEN}已添加后墙 {REAR_WALL_ID}，近面位于 {SCENE_FRAME} "
                  f"的 X={-REAR_WALL_CLEARANCE_M:.2f} m{RESET}")

    def clear_scene(self):
        if not self.scene_client.wait_for_service(timeout_sec=TIMEOUT_SEC):
            self.get_logger().error("MoveIt 的 apply_planning_scene 服务不可用")
            return

        # MoveIt interprets REMOVE with an empty ID as clearing all world collision objects.
        if self.apply_scene(CollisionObject(operation=CollisionObject.REMOVE)):
            frames = ", ".join(self.frame_poses)
            self.frame_poses.clear()
            self.box_collision = None
            self.grasp_boxes.clear()
            print(f"{GREEN}已清除世界碰撞物体，停止发布箱体相关 TF {frames}{RESET}")

    def apply_scene(self, *collisions):
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        scene.world.collision_objects = list(collisions)
        future = self.scene_client.call_async(ApplyPlanningScene.Request(scene=scene))
        rclpy.spin_until_future_complete(self, future, timeout_sec=TIMEOUT_SEC)
        if not future.done():
            future.cancel()
            self.get_logger().error("等待 MoveIt 确认场景更新超时")
        elif future.exception() is not None or not future.result().success:
            self.get_logger().error("MoveIt 场景更新失败")
        else:
            return True
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", action="store_true", help="使用固定仿真视觉数据，跳过识别定位；需启动 MoveIt 仿真环境")
    parser.add_argument("--interactive", action="store_true", help="抱箱、底盘移动和返程动作开始前等待回车确认")
    options = parser.parse_args(remove_ros_args(sys.argv)[1:])
    rclpy.init()
    node = BoxNode(sim=options.sim)
    commands = SimpleQueue()

    def read_commands():
        for line in sys.stdin:
            commands.put(line.strip().lower())
        commands.put("q")

    def read_command():
        while rclpy.ok():
            try:
                return commands.get_nowait()
            except Empty:
                rclpy.spin_once(node, timeout_sec=TF_PUBLISH_PERIOD_SEC)
        return "q"

    def confirm(message):
        # Discard keystrokes queued before this stage was ready.
        while True:
            try:
                if commands.get_nowait() == "q":
                    commands.put("q")
                    return False
            except Empty:
                break
        while rclpy.ok():
            print(message, end="", flush=True)
            command = read_command()
            if command == "":
                return True
            if command == "q":
                commands.put("q")
                break
        return False

    # Keep the executor running while the terminal is waiting for the next command.
    Thread(target=read_commands, daemon=True).start()
    confirm_motion = confirm if options.interactive else None
    try:
        while rclpy.ok():
            print("输入 1 定位并识别箱体，2 规划抱箱，3 添加后墙，0 清除，q 退出：", end="", flush=True)
            command = read_command()
            if command == "1":
                if node.prepare_box_recognition(confirm_motion):
                    node.add_box()
            elif command == "2":
                node.pick_box(confirm_motion)
            elif command == "3":
                node.add_rear_wall()
            elif command == "0":
                node.clear_scene()
            elif command == "q":
                break
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
