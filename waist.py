#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32


COMMAND_TOPIC = "/g01/joint_commands"
BODY_JOINT = "body_joint2"
WAIST_COMMAND_TOPIC = "/taihu_motor_control/positon"
WAIST_FIXED_COMMAND = 0.655417263507843


class WaistCommandBridge(Node):
    def __init__(self) -> None:
        super().__init__("waist_command_bridge")
        self._pub = self.create_publisher(Float32, WAIST_COMMAND_TOPIC, 10)
        self.create_subscription(JointState, COMMAND_TOPIC, self._on_joint_commands, 10)
        self.get_logger().info(
            f"Bridge: {COMMAND_TOPIC}/{BODY_JOINT} -> {WAIST_COMMAND_TOPIC}"
        )

    def _on_joint_commands(self, msg: JointState) -> None:
        for index, name in enumerate(msg.name):
            if name != BODY_JOINT:
                continue
            if index >= len(msg.position):
                return
            self._pub.publish(Float32(data=WAIST_FIXED_COMMAND))
            return


def main() -> None:
    rclpy.init()
    node = WaistCommandBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
