#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向 MoveIt 规划场景添加半透明「深框」碰撞体，并规划/执行指定 group 的关节目标。

流程：
  1. 通过 apply_planning_scene 服务添加深框障碍物（全局生效）
  2. 通过 move_group 的 move_action 发送指定 group 的关节目标，执行规划与运动
  3. 程序退出（正常/异常）时，在 finally 中移除深框

使用前提：
  1. 已启动 move_group（例如：ros2 launch g01_moveit_config demo.launch.py）
  2. 本节点与 move_group 在同一 ROS 域

编译与运行：
  colcon build --packages-select hello_moveit
  source install/setup.bash
  ros2 run hello_moveit hello_deep_frame_body.py
"""

from __future__ import annotations

import sys
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    CollisionObject,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    ObjectColor,
    PlanningScene,
)
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import ColorRGBA

# --------------------------- 场景与障碍物参数 ---------------------------
# 规划坐标系（与 SRDF virtual_joint 的 parent_frame 一致）
PLANNING_FRAME = "world"

# 深框外形尺寸 [m]：长 L × 宽 W × 高 H（开口朝上，无顶盖）
FRAME_LENGTH = 0.8
FRAME_WIDTH = 0.8
FRAME_HEIGHT = 0.7
# 板厚 [m]
WALL_THICKNESS = 0.02

# 深框底面中心在规划坐标系下的位置 [m]
BASE_X = 2.0
BASE_Y = 0.0
BASE_Z = 0.0

COLLISION_OBJECT_ID = "深框"
DISPLAY_COLOR = ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.5)

APPLY_PLANNING_SCENE_SERVICE = "apply_planning_scene"
MOVE_GROUP_ACTION = "move_action"

# --------------------------- 规划参数（可快速切换） ---------------------------
# 想规划哪个 group，只改这里即可（例如: "body" / "dual" / "left_arm"）
ACTIVE_GROUP = "body"

# 快速切换配置：key 是你要填写的 ACTIVE_GROUP，group_name 是 SRDF 真实组名
GROUP_CONFIGS = {
    "body": {
        "group_name": "body",
        "joint_target": {
            "base_joint1": 1.2,
            "base_joint2": 0.0,
            "body_joint1": 0.0,
            "body_joint2": 1.1,
        },
    },
    "dual": {
        "group_name": "dual_arm",
        "joint_target": {
            "base_joint1": 0.2,
            "base_joint2": 0.0,
            "body_joint1": 0.0,
            "body_joint2": 1.1,
            "l_arm_joint1": 0.0,
            "l_arm_joint2": 0.0,
            "l_arm_joint3": 0.0,
            "l_arm_joint4": 0.0,
            "l_arm_joint5": 0.0,
            "l_arm_joint6": 0.0,
            "r_arm_joint1": 0.0,
            "r_arm_joint2": 0.0,
            "r_arm_joint3": 0.0,
            "r_arm_joint4": 0.0,
            "r_arm_joint5": 0.0,
            "r_arm_joint6": 0.0,
        },
    },
}
PLANNER_ID = "RRTConnect"
NUM_PLANNING_ATTEMPTS = 20
ALLOWED_PLANNING_TIME = 5.0  # [s]


def _make_pose(x: float, y: float, z: float) -> Pose:
    """构造单位姿态（无旋转）的位置 Pose。"""
    pose = Pose()
    pose.orientation.w = 1.0
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    return pose


def _add_wall(
    collision_object: CollisionObject,
    dx: float,
    dy: float,
    dz: float,
    x: float,
    y: float,
    z: float,
) -> None:
    """向深框碰撞对象中追加一块 BOX 墙板。"""
    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = [dx, dy, dz]

    pose = _make_pose(BASE_X + x, BASE_Y + y, BASE_Z + z)
    collision_object.primitives.append(primitive)
    collision_object.primitive_poses.append(pose)


def make_deep_frame_collision_object(frame_id: str) -> CollisionObject:
    """构造深框 CollisionObject：底板 + 四面侧墙，顶部敞开。"""
    L = FRAME_LENGTH
    W = FRAME_WIDTH
    H = FRAME_HEIGHT
    tb = WALL_THICKNESS

    obj = CollisionObject()
    obj.header.frame_id = frame_id
    obj.id = COLLISION_OBJECT_ID

    _add_wall(obj, L, W, tb, 0.0, 0.0, tb / 2.0)
    _add_wall(obj, tb, W, H, L / 2.0 - tb / 2.0, 0.0, H / 2.0)
    _add_wall(obj, tb, W, H, -(L / 2.0 - tb / 2.0), 0.0, H / 2.0)
    _add_wall(obj, L - 2.0 * tb, tb, H, 0.0, W / 2.0 - tb / 2.0, H / 2.0)
    _add_wall(obj, L - 2.0 * tb, tb, H, 0.0, -(W / 2.0 - tb / 2.0), H / 2.0)

    obj.operation = CollisionObject.ADD
    return obj


def make_remove_collision_object() -> CollisionObject:
    """构造用于从场景中删除深框的 CollisionObject。"""
    obj = CollisionObject()
    obj.id = COLLISION_OBJECT_ID
    obj.operation = CollisionObject.REMOVE
    return obj


def make_display_object_color() -> ObjectColor:
    """构造 RViz 中深框显示颜色。"""
    oc = ObjectColor()
    oc.id = COLLISION_OBJECT_ID
    oc.color = DISPLAY_COLOR
    return oc


def make_joint_goal_constraints(
    group_name: str, joint_target: dict[str, float]
) -> Constraints:
    """将关节目标字典转换为 MoveIt 关节约束集合。"""
    constraints = Constraints()
    constraints.name = f"{group_name}_joint_goal"

    # 关节约束阈值越小越严格。这里给较小容差，便于精确到位又避免数值抖动导致失败。
    tolerance_above = 1e-3
    tolerance_below = 1e-3

    for joint_name, position in joint_target.items():
        jc = JointConstraint()
        jc.joint_name = joint_name
        jc.position = position
        jc.tolerance_above = tolerance_above
        jc.tolerance_below = tolerance_below
        jc.weight = 1.0
        constraints.joint_constraints.append(jc)

    return constraints


class HelloDeepFrameBody(Node):
    """添加/移除深框，并请求 move_group 执行指定 group 的关节目标规划。"""

    def __init__(self) -> None:
        super().__init__("hello_deep_frame_body")
        self._apply_scene_client = self.create_client(
            ApplyPlanningScene, APPLY_PLANNING_SCENE_SERVICE
        )
        self._move_group_action_client = ActionClient(self, MoveGroup, MOVE_GROUP_ACTION)
        self._latest_joint_positions: dict[str, float] = {}
        self._joint_state_sub = self.create_subscription(
            JointState, "joint_states", self._on_joint_state, 10
        )

    def _on_joint_state(self, msg: JointState) -> None:
        for idx, name in enumerate(msg.name):
            if idx < len(msg.position):
                self._latest_joint_positions[name] = msg.position[idx]

    def _read_current_joint_positions(
        self, joint_names: list[str], timeout_sec: float = 5.0
    ) -> dict[str, float] | None:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            missing = [name for name in joint_names if name not in self._latest_joint_positions]
            if not missing:
                return {name: self._latest_joint_positions[name] for name in joint_names}
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().error(f"读取当前位置超时，缺失关节: {missing}")
        return None

    def _apply_planning_scene(
        self,
        collision_objects: list[CollisionObject],
        object_colors: list[ObjectColor] | None = None,
    ) -> bool:
        """调用 apply_planning_scene，将场景 diff 写入 move_group。"""
        if not self._apply_scene_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(
                f"服务 {APPLY_PLANNING_SCENE_SERVICE} 不可用，请先启动 move_group。"
            )
            return False

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.world.collision_objects.extend(collision_objects)
        if object_colors:
            scene.object_colors.extend(object_colors)

        request = ApplyPlanningScene.Request()
        request.scene = scene
        future = self._apply_scene_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if not future.done() or future.result() is None:
            self.get_logger().error("调用 apply_planning_scene 超时或失败。")
            return False
        if not future.result().success:
            self.get_logger().error("apply_planning_scene 返回 success=False。")
            return False
        return True

    def add_deep_frame(self, collision_object: CollisionObject) -> bool:
        """添加深框并设置半透明颜色。"""
        return self._apply_planning_scene(
            [collision_object], [make_display_object_color()]
        )

    def remove_deep_frame(self) -> bool:
        """移除深框障碍物。"""
        return self._apply_planning_scene([make_remove_collision_object()])

    def _run_move_group(
        self, group_name: str, joint_target: dict[str, float]
        , plan_only: bool
    ) -> tuple[bool, float]:
        """发送一次 move_action 请求，返回(是否成功, wall_time_ms)。"""
        if not self._move_group_action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                f"动作 {MOVE_GROUP_ACTION} 不可用，请先启动 move_group。"
            )
            return False, 0.0

        joint_names = list(joint_target.keys())
        current_positions = self._read_current_joint_positions(joint_names)
        if current_positions is None:
            return False, 0.0

        goal = MoveGroup.Goal()
        goal.request.group_name = group_name
        goal.request.planner_id = PLANNER_ID
        goal.request.num_planning_attempts = NUM_PLANNING_ATTEMPTS
        goal.request.allowed_planning_time = ALLOWED_PLANNING_TIME
        goal.request.start_state.is_diff = True
        goal.request.start_state.joint_state.name = joint_names
        goal.request.start_state.joint_state.position = [
            current_positions[name] for name in joint_names
        ]
        goal.request.goal_constraints.append(
            make_joint_goal_constraints(group_name, joint_target)
        )

        goal.planning_options.plan_only = plan_only
        goal.planning_options.look_around = False
        goal.planning_options.replan = False

        t0 = time.monotonic()
        send_goal_future = self._move_group_action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=15.0)
        if not send_goal_future.done() or send_goal_future.result() is None:
            self.get_logger().error("发送 move_action 目标超时或失败。")
            return False, 0.0

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("move_action 拒绝该目标。")
            return False, 0.0

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=ALLOWED_PLANNING_TIME + 30.0
        )
        if not result_future.done() or result_future.result() is None:
            self.get_logger().error("等待 move_action 结果超时或失败。")
            return False, 0.0

        action_result = result_future.result()
        status = action_result.status
        move_result = action_result.result

        wall_ms = (time.monotonic() - t0) * 1000.0
        if status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error(
                f"move_action 未成功完成，status={status}, "
                f"moveit_error_code={move_result.error_code.val if move_result else 'None'}。"
            )
            return False, wall_ms

        if move_result is None or move_result.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f"MoveIt 规划/执行失败，error_code="
                f"{move_result.error_code.val if move_result else 'None'}。"
            )
            return False, wall_ms

        return True, wall_ms

    def plan_and_execute_group_joint_goal(
        self, group_name: str, joint_target: dict[str, float]
    ) -> bool:
        """按 hello_moveit.cpp 逻辑：先规划，再执行，并分别统计耗时。"""
        self.get_logger().info(
            f"发送 {group_name} 关节目标: {joint_target} "
            f"(planner={PLANNER_ID}, attempts={NUM_PLANNING_ATTEMPTS}, "
            f"time={ALLOWED_PLANNING_TIME:.1f}s)"
        )

        plan_ok, plan_ms = self._run_move_group(group_name, joint_target, plan_only=True)
        self.get_logger().info(
            f"Planning time: {plan_ms:.3f} ms ({'success' if plan_ok else 'failed'})"
        )
        if not plan_ok:
            return False

        exec_ok, exec_ms = self._run_move_group(group_name, joint_target, plan_only=False)
        self.get_logger().info(
            f"Execution time: {exec_ms:.3f} ms ({'success' if exec_ok else 'failed'})"
        )
        self.get_logger().info(f"Total time: {plan_ms + exec_ms:.3f} ms")
        return exec_ok


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node = HelloDeepFrameBody()
    logger = node.get_logger()
    exit_code = 1
    added = False

    try:
        if ACTIVE_GROUP not in GROUP_CONFIGS:
            logger.error(
                f"未找到 group='{ACTIVE_GROUP}' 的配置，请检查 GROUP_CONFIGS。"
            )
            return 1

        group_cfg = GROUP_CONFIGS[ACTIVE_GROUP]
        planning_group = group_cfg["group_name"]
        joint_target = group_cfg["joint_target"]

        logger.info(f"规划坐标系: {PLANNING_FRAME}")
        deep_frame = make_deep_frame_collision_object(PLANNING_FRAME)
        logger.info(
            f"正在添加半透明碰撞体「{COLLISION_OBJECT_ID}」"
            f"（{len(deep_frame.primitives)} 块 BOX，alpha={DISPLAY_COLOR.a}）…"
        )
        if not node.add_deep_frame(deep_frame):
            logger.error("添加深框失败。")
            return 1
        added = True

        logger.info(f"开始规划并执行 {ACTIVE_GROUP}（实际组: {planning_group}）…")
        if not node.plan_and_execute_group_joint_goal(planning_group, joint_target):
            logger.error(f"{ACTIVE_GROUP}（实际组: {planning_group}）规划/执行失败。")
            return 1

        logger.info(
            f"{ACTIVE_GROUP}（实际组: {planning_group}）规划与执行完成。按回车键退出并清理深框…"
        )
        try:
            input()
        except EOFError:
            pass

        exit_code = 0
    finally:
        if added:
            logger.info(f"正在移除碰撞体「{COLLISION_OBJECT_ID}」…")
            if node.remove_deep_frame():
                logger.info("深框已从规划场景中移除。")
            else:
                logger.error("移除深框失败。")
                exit_code = 1

        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

