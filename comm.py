#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G01 真机话题桥接：

- 订阅实际左臂/右臂/升降/腰部状态话题
  -> 合成为 /g01/joint_states（sensor_msgs/JointState）

约束：
- 只有当 left/right/lift/waist 四部分都至少收到过一次状态后，才开始发布 /g01/joint_states
- 未覆盖的关节（base_joint1/base_joint2）发布为 0
升降/腰部50hz
臂100hz
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import rclpy
from pkg_msg_define.msg import MotorRuntimeInfo
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray


JOINT_ORDER: List[str] = [
    "base_joint1",
    "base_joint2",
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

LEFT_JOINTS = JOINT_ORDER[4:10]
RIGHT_JOINTS = JOINT_ORDER[10:16]
LIFT_JOINT = "body_joint1"
WAIST_JOINT = "body_joint2"
LIFT_MM_TO_M = 0.001


@dataclass(frozen=True)
class HwTopics:
    # Aggregated topics used by ros2_control / MoveIt
    state_topic: str

    # Arm state topics use Float32MultiArray with fixed joint ordering.
    # Lift/waist state topics use MotorRuntimeInfo.
    left_state_topic: str
    right_state_topic: str
    lift_state_topic: str
    waist_state_topic: str


def _load_topics() -> HwTopics:
    return HwTopics(
        state_topic="/g01/joint_states",
        left_state_topic="/g01/left_arm/state",
        right_state_topic="/g01/right_arm/state",
        lift_state_topic="/driver_report/lift_motor_info",
        waist_state_topic="/driver_report/waist_motor_info",
    )


class G01Comm(Node):
    def __init__(self) -> None:
        super().__init__("g01_comm")
        self._topics = _load_topics()

        self._pos: Dict[str, float] = {name: 0.0 for name in JOINT_ORDER}
        self._left_rx = False
        self._right_rx = False
        self._lift_rx = False
        self._waist_rx = False
        self._left_last_ns = 0
        self._right_last_ns = 0
        self._lift_last_ns = 0
        self._waist_last_ns = 0

        self._state_pub = self.create_publisher(JointState, self._topics.state_topic, 10)

        self.create_subscription(
            Float32MultiArray, self._topics.left_state_topic, self._on_left_state_array, 10
        )
        self.create_subscription(
            Float32MultiArray, self._topics.right_state_topic, self._on_right_state_array, 10
        )

        self.create_subscription(
            MotorRuntimeInfo, self._topics.lift_state_topic, self._on_lift_state, 10
        )
        self.create_subscription(
            MotorRuntimeInfo, self._topics.waist_state_topic, self._on_waist_state, 10
        )

        # 周期发布 joint_states；只有四部分都收到过才开始发布
        self._publish_hz = float(self.declare_parameter("publish_hz", 50.0).value)
        # 任意一部分超时（秒）就暂停发布 /g01/joint_states；收到恢复后继续
        self._part_timeout_sec = float(self.declare_parameter("part_timeout_sec", 0.5).value)
        self.create_timer(1.0 / max(self._publish_hz, 1.0), self._publish_joint_states)

        self.get_logger().info(
            f"Bridge: arm states(Float32MultiArray) + lift/waist states(MotorRuntimeInfo)"
            f"->{self._topics.state_topic}"
        )

    def _touch_group(self, group: str) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if group == "left":
            self._left_rx = True
            self._left_last_ns = now_ns
        elif group == "right":
            self._right_rx = True
            self._right_last_ns = now_ns
        elif group == "lift":
            self._lift_rx = True
            self._lift_last_ns = now_ns
        elif group == "waist":
            self._waist_rx = True
            self._waist_last_ns = now_ns

    def _on_left_state_array(self, msg: Float32MultiArray) -> None:
        data = list(msg.data)
        if len(data) < 6:
            return
        for i, joint in enumerate(LEFT_JOINTS):
            self._pos[joint] = float(data[i])
        self._touch_group("left")

    def _on_right_state_array(self, msg: Float32MultiArray) -> None:
        data = list(msg.data)
        if len(data) < 6:
            return
        for i, joint in enumerate(RIGHT_JOINTS):
            self._pos[joint] = float(data[i])
        self._touch_group("right")

    def _on_lift_state(self, msg: MotorRuntimeInfo) -> None:
        # 升降电机反馈单位为 mm；JointState/URDF 的移动关节单位为 m。
        self._pos[LIFT_JOINT] = float(msg.position) * LIFT_MM_TO_M
        self._touch_group("lift")

    def _on_waist_state(self, msg: MotorRuntimeInfo) -> None:
        # 腰部电机反馈和 JointState/URDF 均使用 rad，无需换算。
        self._pos[WAIST_JOINT] = float(msg.position)
        self._touch_group("waist")

    def _publish_joint_states(self) -> None:
        if not (self._left_rx and self._right_rx and self._lift_rx and self._waist_rx):
            return

        # 任意一部分超时则暂停发布（避免 MoveIt / ros2_control 使用过期状态）
        if self._part_timeout_sec > 0.0:
            now_ns = self.get_clock().now().nanoseconds
            timeout_ns = int(self._part_timeout_sec * 1e9)
            if (
                now_ns - self._left_last_ns > timeout_ns
                or now_ns - self._right_last_ns > timeout_ns
                or now_ns - self._lift_last_ns > timeout_ns
                or now_ns - self._waist_last_ns > timeout_ns
            ):
                return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINT_ORDER)
        msg.position = [float(self._pos.get(n, 0.0)) for n in JOINT_ORDER]
        self._state_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node: Optional[G01Comm] = None
    try:
        node = G01Comm()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
