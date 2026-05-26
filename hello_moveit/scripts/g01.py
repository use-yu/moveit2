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
  - 已启动 move_group：ros2 launch g01_moveit_config demo.launch.py
  - 本节点与 move_group 在同一 ROS 域

运行（Python）：
  colcon build --packages-select hello_moveit && source install/setup.bash
  ros2 run hello_moveit g01.py

"""

from __future__ import annotations

import math
import multiprocessing
import sys
import time
from typing import Iterable, Sequence

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose
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
)
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import ColorRGBA

# =============================================================================
# 用户可调参数（改这里即可，无需动下面逻辑）
# =============================================================================

# 第一步：关节空间规划使用哪个 SRDF 组
ACTIVE_GROUP = "dual_arm"

# 第二步：末端位姿规划组与目标（连杆 L6，坐标系 base_link）
POSE_GROUP = "left_body"
EE_LINK = "L6"
EE_POSE = dict(
    x=-0.75,
    y=-0.25,
    z=0.0,
    roll=-math.pi / 2,
    pitch=-math.pi / 2,
    yaw=-math.pi,
)
EE_POSE2 = dict(
    x=-0.75,
    y=-0.35,
    z=0.0,
    roll=-math.pi / 2,
    pitch=-math.pi / 2,
    yaw=-math.pi,
)
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
}

# 位姿规划时作为起始状态的关节（不含底盘，与 left_body 组一致）
POSE_START_JOINTS = [
    "body_joint1",
    "body_joint2",
    "l_arm_joint1",
    "l_arm_joint2",
    "l_arm_joint3",
    "l_arm_joint4",
    "l_arm_joint5",
    "l_arm_joint6",
]

# 规划器
PLANNER_ID = "RRTConnect"
NUM_ATTEMPTS = 20
PLAN_TIME_SEC = 10.0

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

# ROS 接口名（move_group 默认）
SVC_APPLY_SCENE = "apply_planning_scene"
SVC_CARTESIAN_PATH = "compute_cartesian_path"
ACT_MOVE_GROUP = "move_action"
ACT_EXEC_TRAJ = "execute_trajectory"

# 末端目标容差（与 MoveGroupInterface 默认一致）
_POS_TOL = 1e-4
_ORI_TOL = 1e-3

# 笛卡尔直线运动参数
CART_EEF_STEP = 0.005     # 服务端 IK 离散步长（m）
CART_MIN_FRACTION = 0.99  # 接受的最小成功比例（<1 表示直线被截断）


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


def make_pose(x: float, y: float, z: float, roll=0.0, pitch=0.0, yaw=0.0) -> Pose:
    """构造 geometry_msgs/Pose。"""
    p = Pose()
    p.position.x, p.position.y, p.position.z = x, y, z
    qx, qy, qz, qw = quat_from_rpy(roll, pitch, yaw)
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = qx, qy, qz, qw
    return p


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


def make_pose_constraints(link: str, pose: Pose) -> Constraints:
    """
    末端位姿目标 → PositionConstraint（位置球）+ OrientationConstraint。
    约束在 PLAN_FRAME 下表达，link_name 为末端连杆名。
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
    pc.header.frame_id = PLAN_FRAME
    pc.link_name = link
    pc.constraint_region = region
    pc.weight = 1.0
    c.position_constraints.append(pc)

    oc = OrientationConstraint()
    oc.header.frame_id = PLAN_FRAME
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
        self._move_cli = ActionClient(self, MoveGroup, ACT_MOVE_GROUP)
        self._exec_cli = ActionClient(self, ExecuteTrajectory, ACT_EXEC_TRAJ)
        # 缓存最新 joint_states，供规划起点使用
        self._joints: dict[str, float] = {}
        self._js_count = 0
        self.create_subscription(JointState, "joint_states", self._on_js, 10)

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

    def move(
        self,
        group: str,
        goal_constraints: list[Constraints],
        joint_names: list[str] | None = None,
        start: dict[str, float] | None = None,
        plan_only: bool = False,
        speed_scale: float | None = None,
    ) -> tuple[bool, float]:
        """
        调用 move_action 一次：规划（plan_only=True）或规划并执行（False）。

        返回 (是否成功, 墙钟耗时 [ms])，从发送目标到收到结果。
        joint_names + 未传 start：从 joint_states 读当前位置作为起点。
        start：显式指定起点（位姿规划在关节运动后用）。
        """
        t0 = time.monotonic()
        elapsed_ms = lambda: (time.monotonic() - t0) * 1000.0

        if not self._move_cli.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(f"动作 {ACT_MOVE_GROUP} 不可用")
            return False, elapsed_ms()

        if start is None and joint_names:
            start = self._get_joints(joint_names)
            if start is None:
                return False, elapsed_ms()

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
            self.get_logger().error("move_action 目标被拒绝或超时")
            return False, elapsed_ms()

        res_fut = send_fut.result().get_result_async()
        if not self._spin_until(res_fut, PLAN_TIME_SEC + 30.0):
            self.get_logger().error("move_action 结果超时")
            return False, elapsed_ms()

        ar = res_fut.result()
        code_val = ar.result.error_code.val if ar.result else None
        if ar.status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error(
                f"move_action 状态失败: {ar.status} (GoalStatus)，"
                f" MoveItErrorCodes={code_val}({_moveit_error_name(code_val) if code_val is not None else 'None'})"
            )
            return False, elapsed_ms()
        if code_val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f"MoveIt 错误码: {code_val} ({_moveit_error_name(code_val)})")
            return False, elapsed_ms()
        return True, elapsed_ms()

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

            ok, used_ms = self.move(
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
        """位姿目标：输入 xyz + rpy + group + 速度缩放。

        use_cartesian=False（默认）：走 move_action（OMPL 关节空间规划，可绕障）。
        use_cartesian=True         ：走 compute_cartesian_path 服务（笛卡尔直线，
            等价于 RViz MotionPlanning 插件中 "Use Cartesian Path" 复选框）。
        """
        pose = make_pose(x, y, z, roll, pitch, yaw)
        p = pose.position
        log = self.get_logger()
        log.info(
            f"[{group}] 位姿目标 {link} @ {PLAN_FRAME}: "
            f"pos({p.x:.3f}, {p.y:.3f}, {p.z:.3f}), rpy({roll:.3f}, {pitch:.3f}, {yaw:.3f}), "
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

        ok, used_ms = self.move(group, goal, start=start, plan_only=False, speed_scale=speed_scale)
        log.info(f"[{group}] pose plan+exec: {used_ms:.3f} ms ({'success' if ok else 'failed'})")
        return ok

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
        """从当前末端位姿直线移动到 end_pose（笛卡尔路径，绝对位姿）。

        - 走 compute_cartesian_path 服务：服务从 start_state 做 FK 得到当前 EE 位姿，
          再沿 waypoints[0]=end_pose 做直线段（关节空间逐段 IK 拼接而成）。
        - 拿到 RobotTrajectory 后用 execute_trajectory action 执行。
        - Humble 的 GetCartesianPath 没有 max_velocity_scaling_factor 字段，速度缩放
          在客户端通过缩放 time_from_start / velocities / accelerations 实现。
        - fraction < min_fraction 视为失败（默认 0.99，要求基本走完整条直线）。
        """
        log = self.get_logger()
        t0 = time.monotonic()
        elapsed_ms = lambda: (time.monotonic() - t0) * 1000.0

        if not self._cart_cli.wait_for_service(timeout_sec=10.0):
            log.error(f"服务 {SVC_CARTESIAN_PATH} 不可用")
            return False
        if not self._exec_cli.wait_for_server(timeout_sec=10.0):
            log.error(f"动作 {ACT_EXEC_TRAJ} 不可用")
            return False

        start = self._get_joints(POSE_START_JOINTS, wait_new=True)
        if start is None:
            return False

        req = GetCartesianPath.Request()
        req.header.frame_id = PLAN_FRAME
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
        log.info(
            f"[{group}] 笛卡尔直线 {link} @ {PLAN_FRAME}: "
            f"end pos({p.x:.3f}, {p.y:.3f}, {p.z:.3f}), "
            f"max_step={eef_step:.4f}, avoid_collisions={avoid_collisions}, "
            f"speed_scale={_clamp01(speed_scale):.2f}"
        )

        fut = self._cart_cli.call_async(req)
        if not self._spin_until(fut, 15.0):
            log.error("compute_cartesian_path 超时")
            return False

        res = fut.result()
        code_val = res.error_code.val
        plan_ms = elapsed_ms()
        log.info(
            f"[{group}] cartesian planning: {plan_ms:.3f} ms, fraction={res.fraction:.3f}, "
            f"error_code={code_val}({_moveit_error_name(code_val)})"
        )
        if res.fraction < min_fraction or code_val != MoveItErrorCodes.SUCCESS:
            log.error(
                f"笛卡尔直线规划未达标：fraction={res.fraction:.3f} < {min_fraction}, "
                f"error_code={code_val}({_moveit_error_name(code_val)})"
            )
            return False

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

        goal = ExecuteTrajectory.Goal(trajectory=traj)
        send_fut = self._exec_cli.send_goal_async(goal)
        if not self._spin_until(send_fut, 15.0) or not send_fut.result().accepted:
            log.error("execute_trajectory 目标被拒绝或超时")
            return False
        res_fut = send_fut.result().get_result_async()
        if not self._spin_until(res_fut, 60.0):
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

        log.info(f"[{group}] cartesian plan+exec: {elapsed_ms():.3f} ms (success)")
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
        2.352639, -1.521807, 2.147846, 0.945048, 2.702450, -2.565092, 
        80 * math.pi / 180, -102 * math.pi / 180, -92 * math.pi / 180, 137 * math.pi / 180, -0 * math.pi, -0 * math.pi / 180]
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
        for name in joint_names:
            log.info(f"  {name}: {current[name]:.6f}")

        if not node.plan_execute_joint_waypoints(ACTIVE_GROUP, DEFAULT_SPEED_SCALE, joint_names, waypoints):
            log.error("关节规划/执行失败")
            return 1

        # --- 3. 末端位姿运动
        ep = EE_POSE
        log.info(f"位姿规划组: {POSE_GROUP}，末端连杆: {EE_LINK}")
        if not node.plan_execute_pose_xyz_rpy(
            POSE_GROUP,
            0.2,  # 速度快容易失败
            EE_LINK,
            ep["x"],
            ep["y"],
            ep["z"],
            ep["roll"],
            ep["pitch"],
            ep["yaw"],
            use_cartesian=False,
        ):
            log.error("末端位姿规划/执行失败")
            return 1

        # --- 4. 末端位姿运动（笛卡尔直线，等价 RViz "Use Cartesian Path"） ---
        ep = EE_POSE2
        log.info(f"位姿规划组: {POSE_GROUP}，末端连杆: {EE_LINK}")
        if not node.plan_execute_pose_xyz_rpy(
            POSE_GROUP,
            0.2,  # 速度快容易失败
            EE_LINK,
            ep["x"],
            ep["y"],
            ep["z"],
            ep["roll"],
            ep["pitch"],
            ep["yaw"],
            use_cartesian=True,
        ):
            log.error("末端位姿规划/执行失败")
            return 1

        log.info("全部完成。按回车退出并移除深框 …")
        try:
            input()
        except EOFError:
            pass
        code = 0

    finally:
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
