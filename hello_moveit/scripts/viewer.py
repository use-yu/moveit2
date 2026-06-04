#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viewer.py

代码作用：
  1. 作为 ROS2 节点运行，从参数中读取视觉输入配置。
  2. 当 input_mode == "tcp" 时，连接视觉设备 TCP 服务。
  3. 向视觉设备发送触发命令，例如：320,1,1,1,1,0。
  4. 接收视觉返回字符串，提取其中所有数字。
  5. 取返回数据最后 6 个数作为 Dobot 目标位姿：
       x, y, z, rx, ry, rz
  6. 将 6 维位姿发布到 /vision/dobot_pose，供后续规划或抓取节点使用。

默认配置等价于：
  input_mode: "tcp"
  vision_ip: "192.168.5.111"
  vision_port: 5700
  vision_trigger_command: "320,1,2,1,1,0"
  vision_request_interval: 0.1
  vision_single_shot: true
  vision_pose_is_robot_pose: true

运行示例：
  source install/setup.bash
  python3 viewer.py

参数覆盖示例：
  python3 viewer.py --ros-args -p vision_trigger_command:="320,1,2,1,1,0"
"""

from __future__ import annotations

import re
import socket
from typing import Optional

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String


# 从视觉返回文本中提取整数、小数、科学计数法数字。
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


class VisionTcpViewer(Node):
    """通过 TCP 触发视觉，并发布 Dobot 位姿的 ROS2 节点。"""

    def __init__(self) -> None:
        super().__init__("vision_tcp_viewer")

        # 声明并读取 ROS2 参数；没有外部 yaml 时使用这里的默认值。
        self.input_mode = self.declare_parameter("input_mode", "tcp").value
        self.vision_ip = self.declare_parameter("vision_ip", "192.168.5.111").value
        self.vision_port = self.declare_parameter("vision_port", 5700).value
        self.trigger_command = self.declare_parameter(
            "vision_trigger_command", "320,1,2,1,1,0"
        ).value
        self.request_interval = self.declare_parameter(
            "vision_request_interval", 0.1
        ).value
        self.single_shot = self.declare_parameter("vision_single_shot", True).value
        self.pose_is_robot_pose = self.declare_parameter(
            "vision_pose_is_robot_pose", True
        ).value

        # 发布解析后的 6 维 Dobot 位姿，同时发布原始返回，便于调试视觉协议。
        self.pose_pub = self.create_publisher(Float64MultiArray, "/vision/dobot_pose", 10)
        self.raw_pub = self.create_publisher(String, "/vision/raw_response", 10)

        self.sock: Optional[socket.socket] = None
        self.finished = False

        # 当前实现只处理 TCP 输入；其他输入方式保留参数入口，避免误运行。
        if self.input_mode != "tcp":
            raise ValueError(f"当前 viewer.py 仅支持 input_mode='tcp'，收到：{self.input_mode}")

        if not self.pose_is_robot_pose:
            self.get_logger().warning(
                "vision_pose_is_robot_pose=false，但当前代码没有做相机到机器人坐标变换；"
                "仍会按 Dobot 位姿直接发布。"
            )

        # 定时触发视觉；single_shot=true 时成功读取一次后自动退出。
        self.timer = self.create_timer(float(self.request_interval), self.request_once)
        self.get_logger().info(
            f"视觉 TCP 节点启动：{self.vision_ip}:{self.vision_port}, "
            f"command='{self.trigger_command}', single_shot={self.single_shot}"
        )

    def connect(self) -> None:
        """建立 TCP 连接；已有连接时直接复用。"""
        if self.sock is not None:
            return

        self.sock = socket.create_connection(
            (str(self.vision_ip), int(self.vision_port)),
            timeout=3.0,
        )
        self.sock.settimeout(20.0)
        self.get_logger().info("已连接视觉 TCP 服务")

    def close(self) -> None:
        """关闭 TCP 连接，释放 socket 资源。"""
        if self.sock is None:
            return

        self.sock.close()
        self.sock = None
        self.get_logger().info("已关闭视觉 TCP 连接")

    def request_once(self) -> None:
        """触发一次视觉请求，解析并发布最后 6 个数字。"""
        if self.finished:
            return

        try:
            self.connect()
            assert self.sock is not None

            # 按协议发送纯文本触发命令；不额外添加换行，保证发送内容与配置一致。
            self.sock.sendall(str(self.trigger_command).encode("utf-8"))
            raw_text = self.sock.recv(4096).decode("utf-8", errors="ignore").strip()
        except OSError as exc:
            self.get_logger().error(f"视觉 TCP 通信失败：{exc}")
            self.close()
            return

        if not raw_text:
            self.get_logger().warning("视觉返回为空")
            return

        self.raw_pub.publish(String(data=raw_text))
        pose = self.parse_dobot_pose(raw_text)
        if pose is None:
            return

        msg = Float64MultiArray()
        msg.data = pose
        self.pose_pub.publish(msg)
        self.get_logger().info(
            "Dobot 位姿："
            f"x={pose[0]:.6f}, y={pose[1]:.6f}, z={pose[2]:.6f}, "
            f"rx={pose[3]:.6f}, ry={pose[4]:.6f}, rz={pose[5]:.6f}"
        )

        if self.single_shot:
            self.finished = True
            self.close()
            rclpy.shutdown()

    def parse_dobot_pose(self, raw_text: str) -> Optional[list[float]]:
        """从视觉返回中取最后 6 个数字作为 x,y,z,rx,ry,rz。"""
        numbers = [float(item) for item in NUMBER_PATTERN.findall(raw_text)]
        if len(numbers) < 6:
            self.get_logger().error(
                f"视觉返回数字不足 6 个，无法解析 Dobot 位姿：{raw_text}"
            )
            return None

        return numbers[-6:]


def main() -> None:
    """ROS2 节点入口。"""
    rclpy.init()
    node: Optional[VisionTcpViewer] = None

    try:
        node = VisionTcpViewer()
        rclpy.spin(node)
    except ExternalShutdownException:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
