#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G01 机器人话题模拟器（配合 g01_topic_hardware / use_real_hardware:=true）

- 100 Hz 发布 /g01/joint_states（sensor_msgs/JointState）
- 订阅 /g01/joint_commands（仅 RViz 点 Execute、控制器 action 进入 EXECUTING 后才发布）

话题与 g01_moveit_config/config/real_hardware_topics.yaml 一致。

运行（真机模式请先启本脚本，再 launch demo）：
  python3 hello_moveit/scripts/robot.py
  # 或
  ros2 run hello_moveit robot.py   # 需先在 CMakeLists 中 install
"""

from __future__ import annotations

import math
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

STATE_TOPIC = "/g01/joint_states"
COMMAND_TOPIC = "/g01/joint_commands"
PUBLISH_HZ = 100.0

# 与 g01.py JOINT_TARGETS["dual_arm"] 键序一致
INITIAL_JOINTS: list[tuple[str, float]] = [
    ("base_joint1", 1.25),
    ("base_joint2", 0.0),
    ("body_joint1", -0.25),
    ("body_joint2", 1.1),
    ("l_arm_joint1", -20 * math.pi / 180),
    ("l_arm_joint2", -102 * math.pi / 180),
    ("l_arm_joint3", -92 * math.pi / 180),
    ("l_arm_joint4", 137 * math.pi / 180),
    ("l_arm_joint5", 0.0),
    ("l_arm_joint6", 0.0),
    ("r_arm_joint1", 80 * math.pi / 180),
    ("r_arm_joint2", -102 * math.pi / 180),
    ("r_arm_joint3", -92 * math.pi / 180),
    ("r_arm_joint4", 137 * math.pi / 180),
    ("r_arm_joint5", 0.0),
    ("r_arm_joint6", 0.0),
]


class G01RobotSimulator(Node):
    def __init__(self) -> None:
        super().__init__("g01_robot_simulator")
        self._lock = threading.Lock()
        self._names = [n for n, _ in INITIAL_JOINTS]
        self._positions = [p for _, p in INITIAL_JOINTS]

        self._state_pub = self.create_publisher(JointState, STATE_TOPIC, 10)
        self.create_subscription(JointState, COMMAND_TOPIC, self._on_command, 10)
        self._publish_state()  # 立即发布，避免 MoveIt 先用旧的 /joint_states
        self.create_timer(1.0 / PUBLISH_HZ, self._publish_state)

        self.get_logger().info(
            f"模拟机器人: 发布 {STATE_TOPIC} @ {PUBLISH_HZ:.0f} Hz, "
            f"订阅 {COMMAND_TOPIC}"
        )

    def _on_command(self, msg: JointState) -> None:
        with self._lock:
            name_to_idx = {n: i for i, n in enumerate(self._names)}
            for i, name in enumerate(msg.name):
                idx = name_to_idx.get(name)
                if idx is None:
                    continue
                if i < len(msg.position):
                    self._positions[idx] = msg.position[i]

    def _publish_state(self) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        with self._lock:
            msg.name = list(self._names)
            msg.position = list(self._positions)
        self._state_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = G01RobotSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()