#!/usr/bin/env python3
"""Publish zero states for the unconnected parts during right-arm testing."""

import math

import rclpy
from pkg_msg_define.msg import MotorRuntimeInfo
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class ZeroStatePublisher(Node):
    def __init__(self):
        super().__init__("g01_zero_state_publisher")

        self._left_pub = self.create_publisher(
            Float32MultiArray, "/g01/left_arm/state", 10
        )
        motor_topics = {
            "lift": "/driver_report/lift_motor_info",
            "waist": "/driver_report/waist_motor_info",
            "head": "/driver_report/camera_motor_info",
        }
        self._motor_pubs = {
            name: self.create_publisher(MotorRuntimeInfo, topic, 10)
            for name, topic in motor_topics.items()
        }

        for name, default_hz in (
            ("left", 100.0),
            ("lift", 50.0),
            ("waist", 50.0),
            ("head", 50.0),
        ):
            hz = float(self.declare_parameter(f"{name}_hz", default_hz).value)
            if not math.isfinite(hz) or hz <= 0.0:
                raise ValueError(f"{name}_hz must be a positive finite number")
            callback = self._publish_left if name == "left" else (
                lambda motor=name: self._publish_motor(motor)
            )
            self.create_timer(1.0 / hz, callback)
            self.get_logger().info(f"{name}: {hz:g} Hz")

    def _publish_left(self):
        msg = Float32MultiArray()
        msg.data = [0.0] * 6
        self._left_pub.publish(msg)

    def _publish_motor(self, name):
        msg = MotorRuntimeInfo()
        msg.state = 0
        msg.error_code = 0
        msg.reserved = 0
        msg.position = 0.0
        msg.speed = 0.0
        self._motor_pubs[name].publish(msg)


def main():
    rclpy.init()
    node = None
    try:
        node = ZeroStatePublisher()
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
