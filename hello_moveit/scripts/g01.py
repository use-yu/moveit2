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
from geometry_msgs.msg import Pose, PoseStamped
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
    roll=-math.pi / 4,
    pitch=-math.pi / 4,
    yaw=-math.pi,
)
EE_POSE2 = dict(
    x=-0.75,
    y=-0.35,
    z=0.0,
    roll=-math.pi / 4,
    pitch=-math.pi / 4,
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
POSE_START_JOINTS = list(JOINT_TARGETS.get(POSE_GROUP, {}).keys())
if not POSE_START_JOINTS:
    raise KeyError(f"JOINT_TARGETS 中未找到 POSE_GROUP={POSE_GROUP} 的关节列表")


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

# 抓取流程默认参数
PRE_GRASP_OFFSET = -0.1  # 预备抓取点沿末端坐标系 z 轴外移的距离 [m]
POST_RETURN_Z_OFFSET = 0.1  # 复位后沿末端坐标系 +z 轴直线移动距离 [m]

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
        self._ik_cli = self.create_client(GetPositionIK, SVC_COMPUTE_IK)
        self._fk_cli = self.create_client(GetPositionFK, SVC_COMPUTE_FK)
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

    def _get_link_pose_fk(
        self, link: str, joints: dict[str, float] | None = None
    ) -> Pose | None:
        """用 /compute_fk 根据关节角求 link 在 PLAN_FRAME 下的位姿。"""
        log = self.get_logger()
        if not self._fk_cli.wait_for_service(timeout_sec=5.0):
            log.error(f"服务 {SVC_COMPUTE_FK} 不可用")
            return None

        if joints is None:
            joints = self._get_joints(POSE_START_JOINTS, wait_new=True)
        if joints is None:
            return None

        req = GetPositionFK.Request()
        req.header.frame_id = PLAN_FRAME
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

    def move(
        self,
        group: str,
        goal_constraints: list[Constraints],
        joint_names: list[str] | None = None,
        start: dict[str, float] | None = None,
        plan_only: bool = False,
        speed_scale: float | None = None,
    ) -> tuple[bool, float, RobotTrajectory | None]:
        """
        调用 move_action 一次：规划（plan_only=True）或规划并执行（False）。

        返回 (是否成功, 墙钟耗时 [ms], planned_trajectory)；
        失败或无轨迹时第三项为 None。
        joint_names + 未传 start：从 joint_states 读当前位置作为起点。
        start：显式指定起点（位姿规划在关节运动后用）。
        """
        t0 = time.monotonic()
        elapsed_ms = lambda: (time.monotonic() - t0) * 1000.0
        no_traj = lambda ok: (ok, elapsed_ms(), None)

        if not self._move_cli.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(f"动作 {ACT_MOVE_GROUP} 不可用")
            return no_traj(False)

        if start is None and joint_names:
            start = self._get_joints(joint_names)
            if start is None:
                return no_traj(False)

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
            return no_traj(False)

        res_fut = send_fut.result().get_result_async()
        if not self._spin_until(res_fut, PLAN_TIME_SEC + 30.0):
            self.get_logger().error("move_action 结果超时")
            return no_traj(False)

        ar = res_fut.result()
        code_val = ar.result.error_code.val if ar.result else None
        if ar.status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error(
                f"move_action 状态失败: {ar.status} (GoalStatus)，"
                f" MoveItErrorCodes={code_val}({_moveit_error_name(code_val) if code_val is not None else 'None'})"
            )
            return no_traj(False)
        if code_val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f"MoveIt 错误码: {code_val} ({_moveit_error_name(code_val)})")
            return no_traj(False)

        traj = None
        if ar.result and ar.result.planned_trajectory.joint_trajectory.points:
            traj = ar.result.planned_trajectory
        return True, elapsed_ms(), traj

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

        start = start_joints if start_joints is not None else self._get_joints(POSE_START_JOINTS, wait_new=True)
        if start is None:
            return None

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
        if verbose:
            log.info(
                f"[{group}] 笛卡尔直线 {link} @ {PLAN_FRAME}: "
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
        req.ik_request.pose_stamped.header.frame_id = PLAN_FRAME
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
                group, link, pose, seed, avoid_collisions=avoid_collisions, return_code=True
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
                f"[ik-multi] target pose 详情: frame={PLAN_FRAME}, link={link}, group={group}\n"
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
                    avoid_collisions=False,
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
    ):
        """从 target_pose 的多个 IK 解里挑「能直线退回 pre_pose」的一组。

        返回 (q_pre, q_target, approach_traj) 三元组；找不到返回 None。
        approach_traj = reverse( cartesian(start=q_target, end=pre_pose) )
            即真正用来「从 pre_pose 直线接近 target_pose」的轨迹。
        """
        log = self.get_logger()
        candidates = self._solve_ik_multi(group, link, target_pose, joint_names, n_candidates)
        if not candidates:
            log.error("[grasp-select] target_pose 在该 group 上没有任何 IK 解")
            return None

        for idx, q_target in enumerate(candidates):
            retreat_traj = self._cartesian_plan(
                group, link, pre_pose,
                speed_scale=speed_scale,
                start_joints=q_target,
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
        link: str = EE_LINK,
        pre_grasp_offset: float = PRE_GRASP_OFFSET,
        post_return_z_offset: float = POST_RETURN_Z_OFFSET,
    ) -> bool:
        """抓取流程（5 步，IK 多解 + approach 预检版）：

        关键改造：提前枚举 target_pose 的多个 IK 解，对每个解预演「直线退回 pre_pose」，
        挑出能走通的那一组 (q_pre, q_target, approach_traj)；然后用 OMPL 关节空间目标
        奔向 q_pre（而不是奔向 pre_pose 的位姿），从根本上消除「OMPL 押错 IK 分支
        导致 cartesian approach 偶发不可达」的问题。

        步骤：
        1. 选解 + 预检：_select_feasible_grasp_pair → (q_pre, q_target, approach_traj)。
        2. OMPL 关节空间 → q_pre（确定的关节配置，可绕障）。
        3. 直接执行缓存的 approach_traj（笛卡尔直线进入 target，免重规划）。
        4. 等待用户按回车（模拟抓取动作）。
        5. 反向播放 approach_traj + 1/6 的 OMPL 轨迹 → 回到 1/6 初始关节配置。
        6. OMPL → 回到 f_joints 复位。
        7. 沿末端坐标系 +z 轴直线移动 POST_RETURN_Z_OFFSET（笛卡尔，不做 IK 多解枚举）。

        参数：
            target_pose      : 末端抓取位姿（geometry_msgs/Pose，在 PLAN_FRAME 下）
            speed_scale      : 各段共用的速度缩放（0~1）
            group            : SRDF 规划组（如 left_body）
            link             : 末端连杆名（默认 EE_LINK）
            pre_grasp_offset : 预备点沿末端 z 轴的退距（m，负值=沿 -z）
            post_return_z_offset : 复位后沿末端 +z 轴直线移动距离（m）
        """
        log = self.get_logger()

        pre_pose = pose_offset_local_z(target_pose, pre_grasp_offset)
        pp = pre_pose.position
        tp = target_pose.position
        log.info(
            f"[pick] 抓取目标 pos({tp.x:.3f}, {tp.y:.3f}, {tp.z:.3f}); "
            f"预备点（沿末端 z 退 {pre_grasp_offset:.3f} m）pos({pp.x:.3f}, {pp.y:.3f}, {pp.z:.3f})"
        )

        log.info("[pick] 0/7  IK 多解枚举 + approach 预检 …")
        picked = self._select_feasible_grasp_pair(
            group, link, target_pose, pre_pose,
            joint_names=POSE_START_JOINTS,
            speed_scale=speed_scale,
        )
        if picked is None:
            log.error("[pick] 未找到「IK 可解 + cartesian approach 可行」的 IK 解")
            return False
        q_pre, q_target, approach_traj = picked
        log.info(
            "[pick] 选定 q_pre: "
            + ", ".join(f"{n}={q_pre[n]:.3f}" for n in POSE_START_JOINTS)
        )

        log.info("[pick] 1/7  OMPL  → q_pre（关节目标，IK 解已确定）")
        current = self._get_joints(POSE_START_JOINTS, wait_new=True)
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
            log.error("[pick] 1/7 未返回 OMPL 轨迹，无法原路退回初始位置")
            return False

        log.info(
            f"[pick] 2/7  执行已缓存的 approach 轨迹 → q_target "
            f"（{len(approach_traj.joint_trajectory.points)} 点，免重规划）"
        )
        if not self._execute_traj(approach_traj):
            log.error("[pick] 直线接近执行失败")
            return False

        log.info("[pick] 3/7  到达抓取位置，按回车继续 …")
        try:
            input()
        except EOFError:
            pass

        log.info(
            "[pick] 4/7  反向播放 approach + 1/7 OMPL → 回到 1/7 初始位置（不再求解）"
        )
        retreat_approach = self._reverse_trajectory(approach_traj)
        if not self._execute_traj(retreat_approach):
            log.error("[pick] 反向播放 approach 失败")
            return False
        retreat_to_start = self._reverse_trajectory(to_pre_traj)
        if not self._execute_traj(retreat_to_start):
            log.error("[pick] 反向播放 1/7 OMPL 回到初始位置失败")
            return False
        log.info(
            "[pick] 已回到 1/7 初始位置: "
            + ", ".join(f"{n}={pick_start_joints[n]:.3f}" for n in POSE_START_JOINTS)
        )

        log.info("[pick] 5/7  OMPL  → 复位关节配置")
        current = self._get_joints(POSE_START_JOINTS, wait_new=True)
        if current is None:
            log.error("[pick] 读取当前关节失败")
            return False
        f_joints = [-0.01292, 1.015203, -0.712975, -0.550402, 1.300752, 0.543868, -0.143126, -0.338787]
        goal = [make_joint_constraints_from_vector(group, POSE_START_JOINTS, f_joints)]
        ok, used_ms, _ = self.move(group, goal, start=current, plan_only=False, speed_scale=0.5)
        log.info(f"[pick] return-to-home: {used_ms:.3f} ms ({'success' if ok else 'failed'})")
        if not ok:
            log.error("[pick] 回到起点失败")
            return False

        log.info(
            f"[pick] 6/7  沿末端 +z 轴直线移动 {post_return_z_offset:.3f} m "
            f"（笛卡尔，从当前位姿 FK 偏移）"
        )
        current = self._get_joints(POSE_START_JOINTS, wait_new=True)
        if current is None:
            log.error("[pick] 读取当前关节失败")
            return False
        ee_pose = self._get_link_pose_fk(link, current)
        if ee_pose is None:
            log.error("[pick] FK 读取当前末端位姿失败")
            return False
        offset_pose = pose_offset_local_z(ee_pose, post_return_z_offset)
        if not self.plan_execute_cartesian_line(
            group, link, offset_pose, speed_scale=speed_scale
        ):
            log.error("[pick] 复位后沿末端 z 轴直线移动失败")
            return False

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

        log.info("按回车继续，输入 q 回车退出并移除深框 …")
        try:
            if input().strip().lower() == "q":
                return 0
        except EOFError:
            pass


        if not node.plan_execute_joint_waypoints(ACTIVE_GROUP, DEFAULT_SPEED_SCALE, joint_names, waypoints):
            log.error("关节规划/执行失败")
            return 1

        # --- 3. 抓取流程：OMPL → 预备点 → Cart 接近 → 回车 → Cart 退回 → OMPL 复位 ---
        ep = EE_POSE2  # 实际抓取目标位姿
        target_pose = make_pose(
            ep["x"], ep["y"], ep["z"], ep["roll"], ep["pitch"], ep["yaw"]
        )
        log.info(f"抓取规划组: {POSE_GROUP}，末端连杆: {EE_LINK}")
        if not node.pick_and_return(
            target_pose=target_pose,
            speed_scale=0.2,
            group=POSE_GROUP,
            link=EE_LINK,
            pre_grasp_offset=PRE_GRASP_OFFSET,
        ):
            log.error("抓取流程失败")
            return 1

        log.info("按回车继续，输入 q 回车退出并移除深框 …")
        try:
            if input().strip().lower() == "q":
                return 0
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
