#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

C++ 等价实现（推荐）：src/hello_g01.cpp → ros2 run hello_moveit hello_g01
"""

from __future__ import annotations

import math
import sys
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
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
from moveit_msgs.srv import ApplyPlanningScene
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
NUM_ATTEMPTS = 100
PLAN_TIME_SEC = 20.0

# 深框障碍物（固定在 world，与机器人 base_link 无关）
SCENE_FRAME = "world"
PLAN_FRAME = "base_link"  # 末端位姿约束坐标系
FRAME_ID = "深框"
FRAME_SIZE = (0.8, 0.8, 0.7)  # 长×宽×高 [m]
WALL_T = 0.02
FRAME_CENTER = (2.0, 0.0, 0.0)  # 底面中心 [m]
FRAME_COLOR = ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.5)

# ROS 接口名（move_group 默认）
SVC_APPLY_SCENE = "apply_planning_scene"
ACT_MOVE_GROUP = "move_action"

# 末端目标容差（与 MoveGroupInterface 默认一致）
_POS_TOL = 1e-4
_ORI_TOL = 1e-3


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


def make_joint_constraints(group: str, joints: dict[str, float]) -> Constraints:
    """关节目标 → MoveIt goal_constraints（每个关节一个 JointConstraint）。"""
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
        self._move_cli = ActionClient(self, MoveGroup, ACT_MOVE_GROUP)
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
        if ar.status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error(f"move_action 状态失败: {ar.status}")
            return False, elapsed_ms()
        if ar.result.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f"MoveIt 错误码: {ar.result.error_code.val}")
            return False, elapsed_ms()
        return True, elapsed_ms()

    def plan_execute_joints(self, group: str, targets: dict[str, float]) -> bool:
        """关节目标：先仅规划验证，再规划并执行，并打印各阶段耗时。"""
        names = list(targets.keys())
        goal = [make_joint_constraints(group, targets)]
        log = self.get_logger()
        log.info(
            f"[{group}] 关节目标 (planner={PLANNER_ID}, "
            f"attempts={NUM_ATTEMPTS}, time={PLAN_TIME_SEC:.1f}s): {targets}"
        )

        plan_ok, plan_ms = self.move(group, goal, joint_names=names, plan_only=True)
        log.info(f"Planning time: {plan_ms:.3f} ms ({'success' if plan_ok else 'failed'})")
        if not plan_ok:
            return False

        exec_ok, exec_ms = self.move(group, goal, joint_names=names, plan_only=False)
        log.info(f"Execution time: {exec_ms:.3f} ms ({'success' if exec_ok else 'failed'})")
        log.info(f"Total time: {plan_ms + exec_ms:.3f} ms")
        return exec_ok

    def plan_execute_pose(self, group: str, link: str, pose: Pose) -> bool:
        """末端位姿：先仅规划再执行，与关节目标相同分步计时。"""
        start = self._get_joints(POSE_START_JOINTS, wait_new=True)
        if start is None:
            return False
        p = pose.position
        log = self.get_logger()
        log.info(
            f"[{group}] 位姿目标 {link} @ {PLAN_FRAME}: "
            f"({p.x:.3f}, {p.y:.3f}, {p.z:.3f})"
        )
        goal = [make_pose_constraints(link, pose)]

        plan_ok, plan_ms = self.move(group, goal, start=start, plan_only=True)
        log.info(f"Pose planning time: {plan_ms:.3f} ms ({'success' if plan_ok else 'failed'})")
        if not plan_ok:
            return False

        exec_ok, exec_ms = self.move(group, goal, start=start, plan_only=False)
        log.info(f"Pose execution time: {exec_ms:.3f} ms ({'success' if exec_ok else 'failed'})")
        log.info(f"Pose total time: {plan_ms + exec_ms:.3f} ms")
        return exec_ok


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
        log.info(f"关节规划组: {ACTIVE_GROUP}")
        if not node.plan_execute_joints(ACTIVE_GROUP, targets):
            log.error("关节规划/执行失败")
            return 1

        # --- 3. 末端位姿运动 ---
        # ep = EE_POSE
        # pose = make_pose(ep["x"], ep["y"], ep["z"], ep["roll"], ep["pitch"], ep["yaw"])
        # log.info(f"位姿规划组: {POSE_GROUP}，末端连杆: {EE_LINK}")
        # if not node.plan_execute_pose(POSE_GROUP, EE_LINK, pose):
        #     log.error("末端位姿规划/执行失败")
        #     return 1

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
