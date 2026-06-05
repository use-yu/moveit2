#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 txt 文件中的 EE_POSE2 位姿列表，在 RViz 中显示所有圆柱体 Marker。
按回车退出时，移除全部 Marker。
base_link
txt 行格式兼容 sample/reach 脚本输出（以 \t 分隔的 key=value）：
    i    x=...    y=...    z=...    roll=...    pitch=...    yaw=...    ...
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Iterable

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

from reach import (  # type: ignore
    CYLINDER_DIAMETER,
    CYLINDER_HEIGHT,
    CYLINDER_MARKER_ID,
    CYLINDER_COLOR,
    CYLINDER_MARKER_NS,
    CYLINDER_MARKER_TOPIC,
    PLAN_FRAME,
    make_pose,
)


def _default_in_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "ee_pose2_bad.txt")


def _parse_kv_line(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    parts = [p.strip() for p in line.strip().split("\t") if p.strip()]
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _iter_poses_from_txt(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            kv = _parse_kv_line(s)
            try:
                x = float(kv.get("x", "nan"))
                y = float(kv.get("y", "nan"))
                z = float(kv.get("z", "nan"))
                roll = float(kv.get("roll", "nan"))
                pitch = float(kv.get("pitch", "nan"))
                yaw = float(kv.get("yaw", "nan"))
            except ValueError:
                continue

            if any(math.isnan(v) for v in (x, y, z, roll, pitch, yaw)):
                continue

            yield {"x": x, "y": y, "z": z, "roll": roll, "pitch": pitch, "yaw": yaw}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default=_default_in_path(), help="输入 txt 文件路径")
    parser.add_argument("--frame", default=PLAN_FRAME, help="Marker 坐标系（默认 base_link）")
    parser.add_argument("--diameter", type=float, default=CYLINDER_DIAMETER, help="圆柱直径")
    parser.add_argument("--height", type=float, default=CYLINDER_HEIGHT, help="圆柱高度")
    parser.add_argument("--republish-hz", type=float, default=0.0, help=">0 时按频率重发 MarkerArray（避免 RViz 丢消息）")
    args = parser.parse_args(argv)

    rclpy.init(args=None)
    node = Node("show_ee_pose2_txt_markers")
    log = node.get_logger()
    pub = node.create_publisher(MarkerArray, CYLINDER_MARKER_TOPIC, 10)

    try:
        poses = list(_iter_poses_from_txt(args.in_path))
        log.info(f"读取 {len(poses)} 条位姿: {args.in_path}")
        if not poses:
            log.error("没有可显示的位姿（可能文件为空/格式不匹配/包含 nan）")
            return 1

        def publish_all():
            now = node.get_clock().now().to_msg()
            arr = MarkerArray()
            for idx, ep in enumerate(poses):
                pose = make_pose(ep["x"], ep["y"], ep["z"], ep["roll"], ep["pitch"], ep["yaw"])
                m = Marker()
                m.header.frame_id = args.frame
                m.header.stamp = now
                m.ns = CYLINDER_MARKER_NS
                m.id = idx
                m.type = Marker.CYLINDER
                m.action = Marker.ADD
                m.pose = pose
                m.scale.x = float(args.diameter)
                m.scale.y = float(args.diameter)
                m.scale.z = float(args.height)
                m.color = CYLINDER_COLOR
                m.lifetime.sec = 0
                m.lifetime.nanosec = 0
                arr.markers.append(m)
            pub.publish(arr)

        def delete_all():
            now = node.get_clock().now().to_msg()
            m = Marker()
            m.header.frame_id = args.frame
            m.header.stamp = now
            m.ns = CYLINDER_MARKER_NS
            m.id = 0
            m.action = Marker.DELETEALL
            arr = MarkerArray()
            arr.markers.append(m)
            pub.publish(arr)

        publish_all()
        if args.republish_hz > 0.0:
            period = 1.0 / float(args.republish_hz)
            timer = node.create_timer(period, publish_all)

        log.info(
            f"已发布 MarkerArray（{len(poses)} 个圆柱）到话题 {CYLINDER_MARKER_TOPIC}。按回车退出并移除…"
        )
        try:
            input()
        except EOFError:
            pass
        return 0

    finally:
        delete_all()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

