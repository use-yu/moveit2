#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立采样脚本：
在 world 坐标系的深框内（考虑物体尺寸）随机采样点，
通过 TF 变换到 base_link 得到 EE_POSE2（位置在 base_link 下表达），
姿态按 reach.py 里 EE_POSE2 基准 roll/pitch 加 ±0.5，yaw 不变。

仅做可行性判定（不执行）：
- IK 可解（多解枚举）
- Cartesian approach 可行（通过 retreat 预检反推）

结果追加写入两个文件：
- ee_pose2_ok.txt
- ee_pose2_bad.txt

每轮发布 RViz 圆柱 marker 刷新显示。
"""

from __future__ import annotations

import argparse
import math
import os
import random

import rclpy

# 复用 reach.py 内的 MoveIt/Marker/TF/几何工具与参数
from reach import (  # type: ignore
    CART_MIN_FRACTION,
    CYLINDER_MARKER_ID,
    EE_LINK,
    EE_POSE2,
    FRAME_CENTER,
    FRAME_SIZE,
    OBJECT_CLEARANCE,
    OBJECT_DIAMETER,
    OBJECT_HEIGHT,
    PLAN_FRAME,
    POSE_GROUP,
    PRE_GRASP_OFFSET,
    SCENE_FRAME,
    WALL_T,
    G01Demo,
    joint_names_for_group,
    pose_from_dict,
    pose_offset_local_z,
)


def _default_out_path(name: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200, help="采样次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（可复现）")
    parser.add_argument("--ok", type=str, default=_default_out_path("ee_pose2_ok.txt"), help="可行输出文件")
    parser.add_argument("--bad", type=str, default=_default_out_path("ee_pose2_bad.txt"), help="不可行输出文件")
    parser.add_argument("--no-frame", action="store_true", help="不添加深框碰撞体（仅采样/显示）")
    args = parser.parse_args(argv)

    random.seed(args.seed)

    rclpy.init(args=None)
    node = G01Demo()
    log = node.get_logger()
    code = 1
    frame_added = False

    try:
        if not args.no_frame:
            log.info(f"添加深框到场景: frame={SCENE_FRAME}")
            if not node.add_frame():
                return 1
            frame_added = True

        pick_group = POSE_GROUP
        pick_joint_names = joint_names_for_group(pick_group)

        # world 下深框内腔范围（考虑物体尺寸）
        L, W, H = FRAME_SIZE
        t = WALL_T
        cx_w, cy_w, cz_w = FRAME_CENTER
        r_obj = OBJECT_DIAMETER * 0.5 + OBJECT_CLEARANCE
        hz_obj = OBJECT_HEIGHT * 0.5 + OBJECT_CLEARANCE

        inner_x_min_w = cx_w - (L / 2 - t) + r_obj
        inner_x_max_w = cx_w + (L / 2 - t) - r_obj
        inner_y_min_w = cy_w - (W / 2 - t) + r_obj
        inner_y_max_w = cy_w + (W / 2 - t) - r_obj
        inner_z_min_w = cz_w + t + hz_obj
        inner_z_max_w = cz_w + H - hz_obj

        if inner_x_min_w >= inner_x_max_w or inner_y_min_w >= inner_y_max_w or inner_z_min_w >= inner_z_max_w:
            log.error(
                "深框内腔不足以容纳物体尺寸：请减小 OBJECT_* 或增大 FRAME_SIZE / 调整 WALL_T/FRAME_CENTER"
            )
            return 1

        base_roll = EE_POSE2["roll"]
        base_pitch = EE_POSE2["pitch"]
        base_yaw = EE_POSE2["yaw"]

        log.info(
            f"开始采样 {args.trials} 次：world 采样 → TF 到 {PLAN_FRAME} → IK+Cartesian 预检"
        )
        log.info(f"ok: {args.ok}")
        log.info(f"bad: {args.bad}")
        log.info(
            f"cart_min_fraction={CART_MIN_FRACTION}, pose_group={pick_group}, ee_link={EE_LINK}"
        )

        os.makedirs(os.path.dirname(os.path.abspath(args.ok)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.bad)), exist_ok=True)

        with open(args.ok, "a", encoding="utf-8") as f_ok, open(args.bad, "a", encoding="utf-8") as f_bad:
            for i in range(args.trials):
                # 1) world 下采样点（物体中心）
                xw = random.uniform(inner_x_min_w, inner_x_max_w)
                yw = random.uniform(inner_y_min_w, inner_y_max_w)
                zw = random.uniform(inner_z_min_w, inner_z_max_w)

                # 2) TF：world -> base_link
                xb_yb_zb = node._transform_point_via_tf(
                    xw, yw, zw, target_frame=PLAN_FRAME, source_frame=SCENE_FRAME
                )
                if xb_yb_zb is None:
                    feasible = False
                    ep = None
                else:
                    xb, yb, zb = xb_yb_zb
                    ep = dict(EE_POSE2)
                    ep["x"], ep["y"], ep["z"] = xb, yb, zb
                    ep["roll"] = base_roll + random.uniform(-0.5, 0.5)
                    ep["pitch"] = base_pitch + random.uniform(-0.5, 0.5)
                    ep["yaw"] = base_yaw

                    # 3) 刷新 RViz marker
                    node.show_cylinder_at_pose(ep, object_id=CYLINDER_MARKER_ID, frame_id=PLAN_FRAME)

                    # 4) IK + cartesian approach 预检
                    target_pose = pose_from_dict(ep)
                    pre_pose = pose_offset_local_z(target_pose, PRE_GRASP_OFFSET)
                    feasible = (
                        node._select_feasible_grasp_pair(
                            group=pick_group,
                            link=EE_LINK,
                            target_pose=target_pose,
                            pre_pose=pre_pose,
                            joint_names=pick_joint_names,
                            speed_scale=0.2,
                            plan_frame=PLAN_FRAME,
                        )
                        is not None
                    )

                if ep is None:
                    line = (
                        f"{i}\t"
                        f"x={math.nan:.4f}\t"
                        f"y={math.nan:.4f}\t"
                        f"z={math.nan:.4f}\t"
                        f"roll={math.nan:.4f}\t"
                        f"pitch={math.nan:.4f}\t"
                        f"yaw={math.nan:.4f}\t"
                        f"world=({xw:.4f},{yw:.4f},{zw:.4f})\n"
                    )
                else:
                    line = (
                        f"{i}\t"
                        f"x={ep['x']:.4f}\t"
                        f"y={ep['y']:.4f}\t"
                        f"z={ep['z']:.4f}\t"
                        f"roll={ep['roll']:.4f}\t"
                        f"pitch={ep['pitch']:.4f}\t"
                        f"yaw={ep['yaw']:.4f}\t"
                        f"world=({xw:.4f},{yw:.4f},{zw:.4f})\n"
                    )

                if feasible:
                    f_ok.write(line)
                    f_ok.flush()
                else:
                    f_bad.write(line)
                    f_bad.flush()

        code = 0

    finally:
        node.remove_cylinder_at_pose()
        if frame_added:
            node.remove_frame()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return code


if __name__ == "__main__":
    raise SystemExit(main())

