#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双臂同步抓取桌面两侧。

流程：
  1. 实机模式读取视觉；--sim 使用 SIM_RECOGNITION_XYZ_RPY。
  2. 桌面中心沿识别坐标系局部 -Z 偏移 0.045 m，然后桌子坐标系
     相对视觉姿态依次绕局部 Z 轴旋转 90 度、绕局部 Y 轴旋转 180 度。
  3. 添加 0.8 x 1.8 x 0.97 m 的桌子碰撞体，并显示桌子 XYZ 坐标轴。
  4. 左/右抓取点分别为桌子局部 y=+0.45/-0.445 m，姿态与桌子一致；
     预抓点位于桌面物理上方 0.20 m。
  5. 左右臂分别求 IK，把两组关节解合成一个 dual_arm 目标，一次规划、一次执行。
  6. 实机双臂末端先下电（仿真跳过），再临时移除桌子碰撞体；左右臂分别
     计算 Cartesian 路径，合成共同时间轴后一次同步执行。
  7. 到位后实机左右末端上电（仿真跳过），等待 1 秒，再反向执行同一轨迹
     同步原路返回。
  8. 恢复桌子碰撞体，等待回车后退出。

运行：
  python3 grasp_table.py --sim
  python3 grasp_table.py                  # 实机视觉，执行前要求回车确认
  python3 grasp_table.py --auto           # 实机视觉，跳过回车确认

需要先启动 move_group，接口名和视觉协议复用同目录 grasp_test.py。
"""

from __future__ import annotations

import bisect
import copy
import math
import random
import sys
import time
from itertools import product
from typing import Sequence

import rclpy
from geometry_msgs.msg import Point, Pose, TransformStamped
from moveit_msgs.msg import CollisionObject, ObjectColor, RobotState, RobotTrajectory
from moveit_msgs.srv import GetStateValidity
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import ColorRGBA
from trajectory_msgs.msg import JointTrajectoryPoint
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

import grasp_test as grasp


# =============================================================================
# 用户可调参数
# =============================================================================

PLAN_FRAME = "base_link"

TABLE_ID = "table"
TABLE_SIZE = (0.5, 1.8, 0.97)  # 桌体局部 X/Y/Z 尺寸 [m]
TABLE_COLOR = ColorRGBA(r=0.48, g=0.30, b=0.14, a=0.85)

# 仿真模式下的“视觉识别位姿”，在 base_link 下表达：xyz [m] + rpy [rad]。
SIM_RECOGNITION_XYZ_RPY = (0.8, 0.0, 0.97, 0.0, 0.0, 0.0)

# 识别点比桌面上方中心高 0.045 m，因此桌面中心沿识别局部 -Z 偏移。
RECOGNITION_TO_TABLE_TOP_LOCAL_Z = 0.005
# 桌子、左右目标和预抓点共同使用这些姿态修正。
TABLE_LOCAL_Z_ROTATION =  math.pi / 2.0 * 0
TABLE_LOCAL_Y_ROTATION = math.pi

# grasp_test 的视觉结果同时给出多个规划坐标系下的位姿。
# right_body 表示使用右侧视觉标定后转换到 base_link 的结果。
VISION_TABLE_POSE_KEY = "right_body"
VISION_TRIGGER_COMMAND = "p,2"

LEFT_GROUP = "left_arm"
RIGHT_GROUP = "right_arm"
DUAL_GROUP = "dual_arm"
LEFT_LINK = "l_tool"
RIGHT_LINK = "r_tool"
LEFT_JOINTS = grasp.joint_names_for_group(LEFT_GROUP)
RIGHT_JOINTS = grasp.joint_names_for_group(RIGHT_GROUP)
DUAL_JOINTS = grasp.joint_names_for_group(DUAL_GROUP)

LEFT_LOCAL_Y = 0.45
RIGHT_LOCAL_Y = -0.445
# 桌子局部 +Z 指向物理下方：左右臂分别越过桌面继续下降 1.5 cm / 0.1 cm。
LEFT_GRASP_LOCAL_Z = 0.015
RIGHT_GRASP_LOCAL_Z = 0.001
PRE_GRASP_LOCAL_Z = 0.15

JOINT_PLAN_SPEED = 0.20
CARTESIAN_SPEED = 0.15
CARTESIAN_EEF_STEP = 0.005
# Cartesian 阶段会先临时移除桌子；其他环境和自碰撞检查仍然启用。
CARTESIAN_AVOID_COLLISIONS = True
TOOL_POWER_SETTLE_SEC = 1.0

IK_ATTEMPTS_PER_ARM = 48
IK_MAX_SOLUTIONS_PER_ARM = 8
IK_SEED_PERTURB = 2.5
MAX_IK_PAIR_PLANS = 24

# 合并轨迹的公共采样周期；控制器更新频率为 200 Hz，因此使用 5 ms。
SYNC_SAMPLE_PERIOD = 0.005
# 合并轨迹碰撞检查无需每 5 ms 调一次服务，按此周期抽样并始终检查终点。
VALIDITY_SAMPLE_PERIOD = 0.02

TABLE_MARKER_TOPIC = "table_coordinate_system"
TABLE_MARKER_NS = "table_coordinate_system"
TABLE_TF_FRAME = "table_top"
TABLE_AXIS_LENGTH = 0.30


# =============================================================================
# 位姿、桌体和 Marker
# =============================================================================

def pose_offset_local(pose: Pose, dx: float, dy: float, dz: float) -> Pose:
    """沿 pose 自身 XYZ 轴平移，姿态保持不变。"""
    q = pose.orientation
    ox, oy, oz = grasp.rotate_xyz_by_quat(dx, dy, dz, q.x, q.y, q.z, q.w)
    out = copy.deepcopy(pose)
    out.position.x += ox
    out.position.y += oy
    out.position.z += oz
    return out


def pose_rotate_local_y(pose: Pose, angle: float) -> Pose:
    """保持位置不变，将姿态绕自身局部 Y 轴右乘旋转 angle。"""
    q = pose.orientation
    half = angle / 2.0
    sin_half = math.sin(half)
    cos_half = math.cos(half)

    out = copy.deepcopy(pose)
    out.orientation.x = q.x * cos_half - q.z * sin_half
    out.orientation.y = q.w * sin_half + q.y * cos_half
    out.orientation.z = q.x * sin_half + q.z * cos_half
    out.orientation.w = q.w * cos_half - q.y * sin_half
    return out


def pose_rotate_local_z(pose: Pose, angle: float) -> Pose:
    """保持位置不变，将姿态绕自身局部 Z 轴右乘旋转 angle。"""
    q = pose.orientation
    half = angle / 2.0
    sin_half = math.sin(half)
    cos_half = math.cos(half)

    out = copy.deepcopy(pose)
    out.orientation.x = q.x * cos_half + q.y * sin_half
    out.orientation.y = q.y * cos_half - q.x * sin_half
    out.orientation.z = q.w * sin_half + q.z * cos_half
    out.orientation.w = q.w * cos_half - q.z * sin_half
    return out


def make_table_collision(table_top_pose: Pose) -> CollisionObject:
    """由桌面上方中心位姿生成桌体。

    桌子坐标系相对识别坐标系绕局部 Y 翻转了 180 度，所以物理桌面上方
    法向是桌子局部 -Z；桌体内部则沿桌子局部 +Z 延伸。
    """
    center_pose = pose_offset_local(table_top_pose, 0.0, 0.0, TABLE_SIZE[2] / 2.0)
    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = list(TABLE_SIZE)

    obj = CollisionObject()
    obj.header.frame_id = PLAN_FRAME
    obj.id = TABLE_ID
    obj.primitives.append(primitive)
    obj.primitive_poses.append(center_pose)
    obj.operation = CollisionObject.ADD
    return obj


def _axis_marker(
    table_top_pose: Pose,
    marker_id: int,
    local_axis: tuple[float, float, float],
    color: ColorRGBA,
) -> Marker:
    q = table_top_pose.orientation
    dx, dy, dz = grasp.rotate_xyz_by_quat(
        local_axis[0] * TABLE_AXIS_LENGTH,
        local_axis[1] * TABLE_AXIS_LENGTH,
        local_axis[2] * TABLE_AXIS_LENGTH,
        q.x,
        q.y,
        q.z,
        q.w,
    )
    start = table_top_pose.position
    marker = Marker()
    marker.header.frame_id = PLAN_FRAME
    marker.ns = TABLE_MARKER_NS
    marker.id = marker_id
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.points = [
        Point(x=start.x, y=start.y, z=start.z),
        Point(x=start.x + dx, y=start.y + dy, z=start.z + dz),
    ]
    marker.scale.x = 0.015
    marker.scale.y = 0.030
    marker.scale.z = 0.035
    marker.color = color
    return marker


def _point_marker(pose: Pose, marker_id: int, color: ColorRGBA, scale: float) -> Marker:
    marker = Marker()
    marker.header.frame_id = PLAN_FRAME
    marker.ns = TABLE_MARKER_NS
    marker.id = marker_id
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose = copy.deepcopy(pose)
    marker.scale.x = marker.scale.y = marker.scale.z = scale
    marker.color = color
    return marker


def make_table_markers(
    table_top_pose: Pose,
    left_target: Pose,
    right_target: Pose,
    left_pre: Pose,
    right_pre: Pose,
) -> MarkerArray:
    """桌子局部 XYZ 坐标轴，以及左右目标/预抓点。"""
    markers = MarkerArray()
    markers.markers.extend(
        [
            _axis_marker(table_top_pose, 0, (1.0, 0.0, 0.0), ColorRGBA(r=1.0, a=1.0)),
            _axis_marker(table_top_pose, 1, (0.0, 1.0, 0.0), ColorRGBA(g=1.0, a=1.0)),
            _axis_marker(table_top_pose, 2, (0.0, 0.0, 1.0), ColorRGBA(b=1.0, a=1.0)),
            _point_marker(left_target, 10, ColorRGBA(g=1.0, b=0.2, a=1.0), 0.045),
            _point_marker(right_target, 11, ColorRGBA(r=1.0, g=0.45, a=1.0), 0.045),
            _point_marker(left_pre, 12, ColorRGBA(g=1.0, b=0.2, a=0.45), 0.035),
            _point_marker(right_pre, 13, ColorRGBA(r=1.0, g=0.45, a=0.45), 0.035),
        ]
    )
    return markers


def acquire_recognition_pose(node: "TableGraspDemo", sim_mode: bool) -> Pose | None:
    """返回 base_link 下未经 4.5 cm 偏移和 180 度修正的视觉识别位姿。"""
    log = node.get_logger()
    if sim_mode:
        pose = grasp.make_pose(*SIM_RECOGNITION_XYZ_RPY)
        log.info(f"[sim] 使用默认视觉识别位姿: {SIM_RECOGNITION_XYZ_RPY}")
        return pose

    result = grasp.read_vision_object_pose(
        node,
        log,
        sim_mode=False,
        trigger_command=VISION_TRIGGER_COMMAND,
    )
    if result is None:
        return None
    _, all_xyz_rpy = result
    if not all_xyz_rpy:
        log.error("视觉没有返回识别位姿")
        return None
    if VISION_TABLE_POSE_KEY not in all_xyz_rpy[0]:
        log.error(
            f"视觉结果中没有 {VISION_TABLE_POSE_KEY!r}，"
            f"可选键: {list(all_xyz_rpy[0])}"
        )
        return None

    values = all_xyz_rpy[0][VISION_TABLE_POSE_KEY]
    pose = grasp.make_pose(*values)
    log.info(
        "视觉识别位姿 @ base_link: "
        f"xyz=({values[0]:.4f}, {values[1]:.4f}, {values[2]:.4f}) m, "
        f"rpy=({math.degrees(values[3]):.2f}, {math.degrees(values[4]):.2f}, "
        f"{math.degrees(values[5]):.2f}) deg"
    )
    return pose


# =============================================================================
# IK 与 dual_arm 关节空间规划
# =============================================================================

def _joint_distance(
    solution: dict[str, float], current: dict[str, float], names: Sequence[str]
) -> float:
    return sum((solution[name] - current[name]) ** 2 for name in names)


def solve_arm_ik_candidates(
    node: "TableGraspDemo",
    group: str,
    link: str,
    pose: Pose,
    joint_names: Sequence[str],
    current: dict[str, float],
) -> list[dict[str, float]]:
    """从当前构型及扰动种子寻找多组无碰撞 IK，按离当前构型的距离排序。"""
    rng = random.Random(20260720 + (0 if group == LEFT_GROUP else 1))
    solutions: list[dict[str, float]] = []
    failure_codes: dict[int, int] = {}

    for attempt in range(IK_ATTEMPTS_PER_ARM):
        if attempt == 0:
            seed = {name: current[name] for name in joint_names}
        else:
            seed = {
                name: current[name] + rng.uniform(-IK_SEED_PERTURB, IK_SEED_PERTURB)
                for name in joint_names
            }

        solution, code = node._solve_ik(
            group,
            link,
            pose,
            seed,
            avoid_collisions=True,
            return_code=True,
            plan_frame=PLAN_FRAME,
        )
        if solution is None:
            if code is not None:
                failure_codes[int(code)] = failure_codes.get(int(code), 0) + 1
            continue

        arm_solution = {name: solution[name] for name in joint_names if name in solution}
        if len(arm_solution) != len(joint_names):
            continue
        duplicate = any(
            all(abs(arm_solution[name] - old[name]) < 1e-2 for name in joint_names)
            for old in solutions
        )
        if not duplicate:
            solutions.append(arm_solution)
        if len(solutions) >= IK_MAX_SOLUTIONS_PER_ARM:
            break

    solutions.sort(key=lambda item: _joint_distance(item, current, joint_names))
    if failure_codes:
        breakdown = ", ".join(
            f"{grasp._moveit_error_name(code)}={count}"
            for code, count in sorted(failure_codes.items())
        )
        node.get_logger().info(f"[{group}] IK 失败分布: {breakdown}")
    node.get_logger().info(
        f"[{group}] 找到 {len(solutions)} 组无碰撞 IK / 最多 {IK_ATTEMPTS_PER_ARM} 次尝试"
    )
    return solutions


def plan_dual_arm_to_pregrasp(
    node: "TableGraspDemo", left_pre: Pose, right_pre: Pose
) -> tuple[RobotTrajectory, dict[str, float]] | None:
    """左右分别 IK，再把一对解作为一个 dual_arm 关节目标联合规划。"""
    log = node.get_logger()
    current = node._get_joints(DUAL_JOINTS, wait_new=True)
    if current is None:
        log.error("读取 dual_arm 当前关节失败")
        return None

    left_solutions = solve_arm_ik_candidates(
        node, LEFT_GROUP, LEFT_LINK, left_pre, LEFT_JOINTS, current
    )
    right_solutions = solve_arm_ik_candidates(
        node, RIGHT_GROUP, RIGHT_LINK, right_pre, RIGHT_JOINTS, current
    )
    if not left_solutions or not right_solutions:
        log.error("至少一只手臂没有抓取点上方 20 cm 的无碰撞 IK")
        return None

    pairs = list(product(left_solutions, right_solutions))
    pairs.sort(
        key=lambda pair: _joint_distance(pair[0], current, LEFT_JOINTS)
        + _joint_distance(pair[1], current, RIGHT_JOINTS)
    )

    for pair_index, (left_solution, right_solution) in enumerate(
        pairs[:MAX_IK_PAIR_PLANS], start=1
    ):
        target = {**left_solution, **right_solution}
        constraints = [grasp.make_joint_constraints(DUAL_GROUP, target)]
        ok, used_ms, trajectory = node.move(
            DUAL_GROUP,
            constraints,
            start=current,
            plan_only=True,
            speed_scale=JOINT_PLAN_SPEED,
            max_retries=2,
        )
        log.info(
            f"[dual_arm] IK 对 {pair_index}/{min(len(pairs), MAX_IK_PAIR_PLANS)} "
            f"联合规划 {used_ms:.1f} ms: {'成功' if ok else '失败'}"
        )
        if ok and trajectory is not None and trajectory.joint_trajectory.points:
            return trajectory, target

    log.error("所有候选 IK 组合均无法生成 dual_arm 无碰撞轨迹")
    return None


# =============================================================================
# 两条 Cartesian 轨迹同步合并
# =============================================================================

def _duration_seconds(point: JointTrajectoryPoint) -> float:
    return point.time_from_start.sec + point.time_from_start.nanosec * 1e-9


def _positions_at_phase(trajectory: RobotTrajectory, phase: float) -> list[float]:
    """按轨迹自身归一化时间 phase 线性插值关节位置。"""
    points = trajectory.joint_trajectory.points
    if not points:
        raise ValueError("轨迹没有路径点")
    if len(points) == 1:
        return list(points[0].positions)

    times = [_duration_seconds(point) for point in points]
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
        a + ratio * (b - a)
        for a, b in zip(points[lower].positions, points[upper].positions)
    ]


def merge_synchronized_trajectories(
    left: RobotTrajectory, right: RobotTrajectory
) -> RobotTrajectory:
    """两臂按相同路径进度重采样，使其同刻开始、同刻结束。"""
    left_names = list(left.joint_trajectory.joint_names)
    right_names = list(right.joint_trajectory.joint_names)
    overlap = set(left_names).intersection(right_names)
    if overlap:
        raise ValueError(f"左右轨迹包含重复关节: {sorted(overlap)}")
    if not left.joint_trajectory.points or not right.joint_trajectory.points:
        raise ValueError("左右 Cartesian 轨迹不能为空")

    left_duration = _duration_seconds(left.joint_trajectory.points[-1])
    right_duration = _duration_seconds(right.joint_trajectory.points[-1])
    duration = max(left_duration, right_duration)
    if duration <= 0.0:
        raise ValueError(
            f"Cartesian 轨迹时长无效: left={left_duration}, right={right_duration}"
        )

    sample_count = max(1, int(math.ceil(duration / SYNC_SAMPLE_PERIOD)))
    merged = RobotTrajectory()
    merged.joint_trajectory.header.frame_id = PLAN_FRAME
    merged.joint_trajectory.joint_names = left_names + right_names

    for index in range(sample_count + 1):
        phase = index / sample_count
        timestamp = duration * phase
        point = JointTrajectoryPoint()
        point.positions = _positions_at_phase(left, phase) + _positions_at_phase(right, phase)
        point.time_from_start.sec = int(timestamp)
        point.time_from_start.nanosec = int(round((timestamp - int(timestamp)) * 1e9))
        if point.time_from_start.nanosec >= 1_000_000_000:
            point.time_from_start.sec += 1
            point.time_from_start.nanosec -= 1_000_000_000
        merged.joint_trajectory.points.append(point)
    return merged


class TableGraspDemo(grasp.G01Demo):
    def __init__(self, sim_mode: bool = False):
        super().__init__(sim_mode=sim_mode)
        self._validity_cli = self.create_client(GetStateValidity, "check_state_validity")
        self._table_tf_broadcaster = StaticTransformBroadcaster(self)
        marker_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._table_marker_pub = self.create_publisher(
            MarkerArray, TABLE_MARKER_TOPIC, marker_qos
        )

    def add_table(self, table_top_pose: Pose) -> bool:
        obj = make_table_collision(table_top_pose)
        color = ObjectColor(id=TABLE_ID, color=TABLE_COLOR)
        return self._apply_scene([obj], [color])

    def remove_table(self) -> bool:
        obj = CollisionObject(id=TABLE_ID, operation=CollisionObject.REMOVE)
        return self._apply_scene([obj])

    def show_table_markers(
        self,
        table_top_pose: Pose,
        left_target: Pose,
        right_target: Pose,
        left_pre: Pose,
        right_pre: Pose,
    ) -> None:
        table_tf = TransformStamped()
        table_tf.header.frame_id = PLAN_FRAME
        table_tf.header.stamp = self.get_clock().now().to_msg()
        table_tf.child_frame_id = TABLE_TF_FRAME
        table_tf.transform.translation.x = table_top_pose.position.x
        table_tf.transform.translation.y = table_top_pose.position.y
        table_tf.transform.translation.z = table_top_pose.position.z
        table_tf.transform.rotation = copy.deepcopy(table_top_pose.orientation)
        self._table_tf_broadcaster.sendTransform(table_tf)

        marker_array = make_table_markers(
            table_top_pose, left_target, right_target, left_pre, right_pre
        )
        stamp = self.get_clock().now().to_msg()
        for marker in marker_array.markers:
            marker.header.stamp = stamp
        self._table_marker_pub.publish(marker_array)
        self.get_logger().info(
            f"已发布桌子坐标系: TF={PLAN_FRAME}->{TABLE_TF_FRAME}, "
            f"MarkerArray=/{TABLE_MARKER_TOPIC}（X 红 / Y 绿 / Z 蓝）"
        )

    def clear_table_markers(self) -> None:
        marker = Marker()
        marker.header.frame_id = PLAN_FRAME
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = TABLE_MARKER_NS
        marker.action = Marker.DELETEALL
        self._table_marker_pub.publish(MarkerArray(markers=[marker]))

    def set_both_tool_power(self, status: int) -> bool:
        """设置左右末端电源；仿真模式完全跳过工具服务调用。"""
        if self.sim_mode:
            self.get_logger().info(
                f"[sim] 跳过双臂末端{'上电' if status else '下电'}"
            )
            return True

        left_ok = self.set_tool_power("left", status)
        right_ok = self.set_tool_power("right", status)
        if not left_ok or not right_ok:
            self.get_logger().error(
                f"双臂末端{'上电' if status else '下电'}失败: "
                f"left={left_ok}, right={right_ok}"
            )
            return False
        self.get_logger().info(f"双臂末端{'上电' if status else '下电'}成功")
        return True

    def validate_dual_trajectory(self, trajectory: RobotTrajectory) -> bool:
        """在同一 dual_arm 状态中检查合并轨迹，捕获双臂相互碰撞。"""
        log = self.get_logger()
        if not self._validity_cli.wait_for_service(timeout_sec=10.0):
            log.error("服务 check_state_validity 不可用")
            return False

        points = trajectory.joint_trajectory.points
        names = list(trajectory.joint_trajectory.joint_names)
        stride = max(1, int(round(VALIDITY_SAMPLE_PERIOD / SYNC_SAMPLE_PERIOD)))
        indices = list(range(0, len(points), stride))
        if not indices or indices[-1] != len(points) - 1:
            indices.append(len(points) - 1)

        for checked, index in enumerate(indices, start=1):
            request = GetStateValidity.Request()
            request.group_name = DUAL_GROUP
            request.robot_state = RobotState()
            request.robot_state.is_diff = True
            request.robot_state.joint_state.name = names
            request.robot_state.joint_state.position = list(points[index].positions)
            future = self._validity_cli.call_async(request)
            if not self._spin_until(future, 5.0):
                log.error(f"合并轨迹状态检查超时: point={index}")
                return False
            response = future.result()
            if not response.valid:
                contacts = [
                    f"{contact.contact_body_1}<->{contact.contact_body_2}"
                    for contact in response.contacts[:5]
                ]
                detail = ", ".join(contacts) if contacts else "未返回 contact 详情"
                log.error(f"合并轨迹存在碰撞: point={index}, {detail}")
                return False

        log.info(f"合并轨迹 dual_arm 有效性检查通过: {len(indices)} 个采样状态")
        return True


def plan_synchronized_cartesian_descent(
    node: TableGraspDemo,
    left_target: Pose,
    right_target: Pose,
) -> RobotTrajectory | None:
    """从同一真实起点分别规划两臂直线，再合成一个同步双臂轨迹。"""
    log = node.get_logger()
    start = node._get_joints(DUAL_JOINTS, wait_new=True)
    if start is None:
        log.error("读取 Cartesian 规划起点失败")
        return None

    left = node._cartesian_plan(
        LEFT_GROUP,
        LEFT_LINK,
        left_target,
        speed_scale=CARTESIAN_SPEED,
        avoid_collisions=CARTESIAN_AVOID_COLLISIONS,
        eef_step=CARTESIAN_EEF_STEP,
        start_joints=start,
        joint_names=LEFT_JOINTS,
        plan_frame=PLAN_FRAME,
    )
    if left is None:
        log.error("左臂 Cartesian 下降规划失败")
        return None

    right = node._cartesian_plan(
        RIGHT_GROUP,
        RIGHT_LINK,
        right_target,
        speed_scale=CARTESIAN_SPEED,
        avoid_collisions=CARTESIAN_AVOID_COLLISIONS,
        eef_step=CARTESIAN_EEF_STEP,
        start_joints=start,
        joint_names=RIGHT_JOINTS,
        plan_frame=PLAN_FRAME,
    )
    if right is None:
        log.error("右臂 Cartesian 下降规划失败")
        return None

    try:
        merged = merge_synchronized_trajectories(left, right)
    except ValueError as exc:
        log.error(f"同步合并 Cartesian 轨迹失败: {exc}")
        return None

    duration = _duration_seconds(merged.joint_trajectory.points[-1])
    log.info(
        f"同步轨迹已生成: joints={len(merged.joint_trajectory.joint_names)}, "
        f"points={len(merged.joint_trajectory.points)}, duration={duration:.3f}s"
    )
    if not node.validate_dual_trajectory(merged):
        return None
    return merged


# =============================================================================
# 主流程
# =============================================================================

def split_args(argv: list[str] | None = None) -> tuple[bool, bool, list[str]]:
    raw = list(sys.argv if argv is None else argv)
    sim_mode = "--sim" in raw
    auto_confirm = "--auto" in raw
    ros_args = [arg for arg in raw if arg not in ("--sim", "--auto")]
    return sim_mode, auto_confirm, ros_args


def _confirm_execution(auto_confirm: bool) -> bool:
    if auto_confirm:
        return True
    try:
        answer = input(
            "桌子碰撞体、坐标系和目标点已显示；"
            "按回车开始计算 IK 和规划，输入 q 回车取消: "
        )
    except EOFError:
        return False
    return answer.strip().lower() != "q"


def main(argv: list[str] | None = None) -> int:
    sim_mode, auto_confirm, ros_args = split_args(argv)
    rclpy.init(args=ros_args)
    node = TableGraspDemo(sim_mode=sim_mode)
    log = node.get_logger()
    table_added = False
    code = 1

    try:
        # 必须先视觉识别，再计算桌面中心和统一姿态。
        recognition_pose = acquire_recognition_pose(node, sim_mode)
        if recognition_pose is None:
            return 1

        # 右乘局部旋转：T_table = T_vision * Rz(90 deg) * Ry(180 deg)。
        table_top = pose_rotate_local_y(
            pose_rotate_local_z(
                pose_offset_local(
                    recognition_pose,
                    0.0,
                    0.0,
                    RECOGNITION_TO_TABLE_TOP_LOCAL_Z,
                ),
                TABLE_LOCAL_Z_ROTATION,
            ),
            TABLE_LOCAL_Y_ROTATION,
        )
        # 桌子和所有目标直接继承 table_top 的姿态，不再对目标单独旋转。
        left_target = pose_offset_local(
            table_top,
            0.0,
            LEFT_LOCAL_Y,
            LEFT_GRASP_LOCAL_Z,
        )
        right_target = pose_offset_local(
            table_top,
            0.0,
            RIGHT_LOCAL_Y,
            RIGHT_GRASP_LOCAL_Z,
        )

        # 绕 Y 翻转后，桌面物理上方对应桌子局部 -Z。
        left_pre = pose_offset_local(table_top, 0.0, LEFT_LOCAL_Y, -PRE_GRASP_LOCAL_Z)
        right_pre = pose_offset_local(table_top, 0.0, RIGHT_LOCAL_Y, -PRE_GRASP_LOCAL_Z)

        if not node.add_table(table_top):
            log.error("添加桌子碰撞体失败")
            return 1
        table_added = True
        node.show_table_markers(table_top, left_target, right_target, left_pre, right_pre)

        log.info(
            f"桌子尺寸={TABLE_SIZE} m；"
            f"左目标局部 (y,z)=({LEFT_LOCAL_Y:+.3f}, {LEFT_GRASP_LOCAL_Z:+.3f}) m；"
            f"右目标局部 (y,z)=({RIGHT_LOCAL_Y:+.3f}, {RIGHT_GRASP_LOCAL_Z:+.3f}) m；"
            f"预抓高度={PRE_GRASP_LOCAL_Z:.3f} m；"
            f"识别到桌面偏移={RECOGNITION_TO_TABLE_TOP_LOCAL_Z:+.3f} m；"
            f"桌子/目标姿态绕局部 Z={math.degrees(TABLE_LOCAL_Z_ROTATION):.1f} deg，"
            f"再绕局部 Y={math.degrees(TABLE_LOCAL_Y_ROTATION):.1f} deg"
        )
        log.info(
            "左抓取点(base_link): "
            f"x={left_target.position.x:.4f}, y={left_target.position.y:.4f}, "
            f"z={left_target.position.z:.4f} m"
        )
        log.info(
            "右抓取点(base_link): "
            f"x={right_target.position.x:.4f}, y={right_target.position.y:.4f}, "
            f"z={right_target.position.z:.4f} m"
        )
        if not _confirm_execution(auto_confirm):
            log.info("用户取消执行")
            return 0

        planned = plan_dual_arm_to_pregrasp(node, left_pre, right_pre)
        if planned is None:
            return 1
        pregrasp_trajectory, _ = planned
        log.info("执行一条 dual_arm 联合关节轨迹到左右预抓点")
        if not node._execute_traj(pregrasp_trajectory):
            log.error("dual_arm 预抓轨迹执行失败")
            return 1

        log.info("双臂已到达预抓点，下降前先让左右末端下电")
        if not node.set_both_tool_power(0):
            return 1

        # 下降和原路返回阶段不考虑与桌子的碰撞，但保留其他碰撞检查。
        log.info("临时从 MoveIt 规划场景移除桌子碰撞体")
        if not node.remove_table():
            log.error("临时移除桌子碰撞体失败")
            return 1
        table_added = False

        descent = plan_synchronized_cartesian_descent(node, left_target, right_target)
        if descent is None:
            return 1
        log.info("执行一条 12 关节共同时基轨迹，两臂同步直线下降")
        if not node._execute_traj(descent):
            log.error("双臂同步 Cartesian 轨迹执行失败")
            return 1

        log.info("双臂已同步到达桌面两侧抓取点")

        try:
            input("按回车: ")
        except EOFError:
            pass

        if not node.set_both_tool_power(1):
            return 1

        log.info(f"双臂末端均已上电，等待 {TOOL_POWER_SETTLE_SEC:.1f} s")
        time.sleep(TOOL_POWER_SETTLE_SEC)

        log.info("反向执行同一条 12 关节轨迹，两臂同步直线原路返回")
        retreat = node._reverse_trajectory(descent)
        if not node._execute_traj(retreat):
            log.error("双臂同步 Cartesian 原路返回失败")
            return 1

        log.info("双臂已返回预抓点，重新添加桌子碰撞体")
        if not node.add_table(table_top):
            log.error("恢复桌子碰撞体失败")
            return 1
        table_added = True

        code = 0
        if sys.stdin.isatty() and not auto_confirm:
            try:
                input("桌子碰撞体已恢复；按回车移除桌子并退出: ")
            except EOFError:
                pass
        return code
    finally:
        node.clear_table_markers()
        if table_added and not node.remove_table():
            log.error("移除桌子碰撞体失败")
            code = 1
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
