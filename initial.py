#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从机器人当前位置规划并执行到 MOVE_TO_GRASP_RESET_Q。

前提：
    已启动 MoveIt move_group 和机器人状态桥接，且
    /g01/joint_states 正在发布。

运行：
    python3 /home/ws_moveit/src/initial.py
"""

from __future__ import annotations

import math
import sys
import time

import rclpy
from action_msgs.msg import GoalStatus
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState


NODE_NAME = "g01_reset_to_grasp_initial"
JOINT_STATE_TOPIC = "/g01/joint_states"
MOVE_ACTION = "move_action"
PLANNING_GROUP = "dual_arm_body"
PLANNER_ID = "RRTConnect"

JOINT_STATE_TIMEOUT_SEC = 5.0
ACTION_SERVER_TIMEOUT_SEC = 10.0
PLANNING_TIME_SEC = 10.0
RESULT_TIMEOUT_SEC = 90.0
NUM_PLANNING_ATTEMPTS = 30
SPEED_SCALE = 0.2
JOINT_TOLERANCE = 1e-5

# 顺序必须与 g01_moveit_config/config/G01.srdf 的 dual_arm_body 一致：
# body_joint1、body_joint2、左臂 6 轴、右臂 6 轴。
DUAL_ARM_BODY_JOINTS = [
    "body_joint1",
    "body_joint2",
    "l_arm_joint1",
    "l_arm_joint2",
    "l_arm_joint3",
    "l_arm_joint4",
    "l_arm_joint5",
    "l_arm_joint6",
    "r_arm_joint1",
    "r_arm_joint2",
    "r_arm_joint3",
    "r_arm_joint4",
    "r_arm_joint5",
    "r_arm_joint6",
]

MOVE_TO_GRASP_RESET_Q = [
    0.0,
    0 * math.pi / 180,
    -90 * math.pi / 180,
    87 * math.pi / 180,
    0 * math.pi / 180,
    0 * math.pi / 180,
    0 * math.pi / 180,
    0 * math.pi / 180,
    90 * math.pi / 180,
    -87 * math.pi / 180,
    0 * math.pi / 180,
    0 * math.pi / 180,
    0 * math.pi / 180,
    0 * math.pi / 180,
]

if len(DUAL_ARM_BODY_JOINTS) != len(MOVE_TO_GRASP_RESET_Q):
    raise ValueError("dual_arm_body 关节数与 MOVE_TO_GRASP_RESET_Q 长度不一致")


def make_joint_goal() -> Constraints:
    """构造 MOVE_TO_GRASP_RESET_Q 的关节目标约束。"""
    constraints = Constraints()
    constraints.name = "move_to_grasp_reset_q"
    for name, position in zip(DUAL_ARM_BODY_JOINTS, MOVE_TO_GRASP_RESET_Q):
        joint = JointConstraint()
        joint.joint_name = name
        joint.position = float(position)
        joint.tolerance_above = JOINT_TOLERANCE
        joint.tolerance_below = JOINT_TOLERANCE
        joint.weight = 1.0
        constraints.joint_constraints.append(joint)
    return constraints


class ResetToGraspInitial(Node):
    def __init__(self) -> None:
        super().__init__(NODE_NAME)
        self._joint_positions: dict[str, float] = {}
        self._joint_state_count = 0
        self.create_subscription(
            JointState,
            JOINT_STATE_TOPIC,
            self._on_joint_state,
            10,
        )
        self._move_client = ActionClient(self, MoveGroup, MOVE_ACTION)

    def _on_joint_state(self, msg: JointState) -> None:
        for index, name in enumerate(msg.name):
            if index >= len(msg.position):
                continue
            position = float(msg.position[index])
            if math.isfinite(position):
                self._joint_positions[str(name)] = position
        self._joint_state_count += 1

    def wait_for_current_state(self) -> dict[str, float] | None:
        """等待调用后到达的一帧完整 dual_arm_body 当前状态。"""
        count_before_wait = self._joint_state_count
        deadline = time.monotonic() + JOINT_STATE_TIMEOUT_SEC
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(
                self,
                timeout_sec=min(0.1, max(0.0, deadline - time.monotonic())),
            )
            if self._joint_state_count <= count_before_wait:
                continue
            missing = [
                name
                for name in DUAL_ARM_BODY_JOINTS
                if name not in self._joint_positions
            ]
            if not missing:
                return {
                    name: self._joint_positions[name]
                    for name in DUAL_ARM_BODY_JOINTS
                }

        missing = [
            name
            for name in DUAL_ARM_BODY_JOINTS
            if name not in self._joint_positions
        ]
        self.get_logger().error(
            f"{JOINT_STATE_TIMEOUT_SEC:.1f}s 内未收到完整 {JOINT_STATE_TOPIC}；"
            f"缺少关节: {missing}"
        )
        return None

    def reset(self) -> bool:
        """从最新实际关节状态规划并执行到复位构型。"""
        log = self.get_logger()
        current = self.wait_for_current_state()
        if current is None:
            return False

        log.info(
            "当前 dual_arm_body 状态: "
            + ", ".join(f"{name}={value:.6f}" for name, value in current.items())
        )
        log.info(
            "目标 MOVE_TO_GRASP_RESET_Q: "
            + ", ".join(
                f"{name}={value:.6f}"
                for name, value in zip(
                    DUAL_ARM_BODY_JOINTS,
                    MOVE_TO_GRASP_RESET_Q,
                )
            )
        )

        if not self._move_client.wait_for_server(
            timeout_sec=ACTION_SERVER_TIMEOUT_SEC
        ):
            log.error(
                f"{ACTION_SERVER_TIMEOUT_SEC:.1f}s 内动作 {MOVE_ACTION} 不可用"
            )
            return False

        goal = MoveGroup.Goal()
        goal.request.group_name = PLANNING_GROUP
        goal.request.planner_id = PLANNER_ID
        goal.request.num_planning_attempts = NUM_PLANNING_ATTEMPTS
        goal.request.allowed_planning_time = PLANNING_TIME_SEC
        goal.request.max_velocity_scaling_factor = SPEED_SCALE
        goal.request.max_acceleration_scaling_factor = SPEED_SCALE
        goal.request.goal_constraints = [make_joint_goal()]

        # 显式使用最新实际位置作为规划起点；is_diff=True 让 MoveIt 用当前完整
        # RobotState 补齐组外关节。未设置 ACM，因此正常检查自碰撞和场景碰撞。
        goal.request.start_state.is_diff = True
        goal.request.start_state.joint_state.name = list(current.keys())
        goal.request.start_state.joint_state.position = list(current.values())
        goal.planning_options.plan_only = False

        log.info(
            f"提交 {PLANNING_GROUP} 复位：planner={PLANNER_ID}, "
            f"attempts={NUM_PLANNING_ATTEMPTS}, "
            f"planning_time={PLANNING_TIME_SEC:.1f}s, "
            f"speed_scale={SPEED_SCALE:.2f}，保留碰撞检查"
        )
        send_future = self._move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(
            self,
            send_future,
            timeout_sec=ACTION_SERVER_TIMEOUT_SEC,
        )
        if not send_future.done():
            send_future.cancel()
            log.error("发送复位目标超时")
            return False

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            log.error("复位目标被 move_group 拒绝")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self,
            result_future,
            timeout_sec=RESULT_TIMEOUT_SEC,
        )
        if not result_future.done():
            log.error(f"{RESULT_TIMEOUT_SEC:.1f}s 内复位规划/执行未完成")
            return False

        action_result = result_future.result()
        if action_result is None or action_result.result is None:
            log.error("move_group 未返回有效复位结果")
            return False

        error_code = int(action_result.result.error_code.val)
        if (
            action_result.status != GoalStatus.STATUS_SUCCEEDED
            or error_code != MoveItErrorCodes.SUCCESS
        ):
            log.error(
                f"复位失败：action_status={action_result.status}, "
                f"MoveItErrorCode={error_code}"
            )
            return False

        log.info("已从当前位置复位到 MOVE_TO_GRASP_RESET_Q")
        return True


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node = ResetToGraspInitial()
    try:
        return 0 if node.reset() else 1
    except KeyboardInterrupt:
        node.get_logger().warning("用户中断复位")
        return 130
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
