#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Numeric IK for G01-URDF888.urdf.

直接改下面「用户输入参数」后运行：
    python3 ik_test.py

目标位姿 TARGET_XYZ/TARGET_RPY 表达在 BASE_LINK 坐标系下。
SEED_JOINTS 是 IK 迭代初始值，顺序等于脚本输出的 active_joints。
"""

from __future__ import annotations

import math
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation


# =============================================================================
# 用户输入参数：主要改这里
# =============================================================================
URDF_XACRO = Path(__file__).parent / "moveit_resources/g01_description/urdf/G01-URDF888.urdf"

# 基坐标系和末端坐标系 link 名。右臂可改成 BASE_LINK="r_base_link", EE_LINK="r_tool"。
BASE_LINK = "SJ"
EE_LINK = "l_tool"

# 目标末端位姿，表达在 BASE_LINK 坐标系下，单位：m / rad。
# TARGET_XYZ = [-0.738, 0.55, 0]
# TARGET_RPY = [-90.0, -90.0, 180.0]
TARGET_XYZ = [-0.738, 0.41, 0]
TARGET_RPY = [60.0, -90.0, -150.0]
# 固定腰部角度。BASE_LINK="SJ" 到 EE_LINK="r_tool" 的链上会经过 body_joint2。
WAIST_JOINT = "body_joint2"
WAIST_ANGLE = 30.0
WAIST_ANGLE_DEG = True

# IK 迭代初始关节角。腰部已固定，这里只填右臂 6 个关节。
# 当前顺序: r_arm_joint1, r_arm_joint2, ..., r_arm_joint6。
# SEED_JOINTS = [-108.0, -88.0, -48.0, -42.0, 43.0, 0.0]
SEED_JOINTS = [23, -87.0, -35.0, -60.0, 53.0, -2.0]

# 如果 TARGET_RPY / SEED_JOINTS 里写的是角度，把对应开关改成 True。
TARGET_RPY_DEG = True
SEED_JOINTS_DEG = True

# IK 参数
MAX_ITERS = 300
POS_TOL = 1e-4
ORI_TOL = 1e-4
DAMPING = 1e-3
MAX_STEP = 0.2
ORIENTATION_WEIGHT = 1.0


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float | None
    upper: float | None


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def matrix_from_xyz_rpy(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    """URDF rpy: fixed-axis roll, pitch, yaw. R = Rz(yaw) * Ry(pitch) * Rx(roll)."""
    roll, pitch, yaw = [float(v) for v in rpy]
    t = np.eye(4)
    t[:3, :3] = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    t[:3, 3] = np.array(xyz, dtype=float)
    return t


def matrix_from_axis_angle(axis: Sequence[float], angle: float) -> np.ndarray:
    axis = normalize(axis)
    t = np.eye(4)
    t[:3, :3] = Rotation.from_rotvec(axis * float(angle)).as_matrix()
    return t


def matrix_from_axis_translation(axis: Sequence[float], distance: float) -> np.ndarray:
    axis = normalize(axis)
    t = np.eye(4)
    t[:3, 3] = axis * float(distance)
    return t


def rpy_from_matrix(r: np.ndarray) -> tuple[float, float, float]:
    pitch = math.atan2(-r[2, 0], math.hypot(r[0, 0], r[1, 0]))
    cp = math.cos(pitch)
    if abs(cp) > 1e-9:
        roll = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(r[1, 0], r[0, 0])
    else:
        roll = 0.0
        yaw = math.atan2(-r[0, 1], r[1, 1])
    return roll, pitch, yaw


def normalize(v: Sequence[float]) -> np.ndarray:
    out = np.array(v, dtype=float)
    norm = float(np.linalg.norm(out))
    if norm < 1e-12:
        raise ValueError(f"axis 长度为 0: {v}")
    return out / norm


def parse_vec(text: str | None, default: Sequence[float]) -> list[float]:
    if text is None:
        return [float(v) for v in default]
    return [float(v) for v in text.split()]


def expand_xacro(path: Path) -> str:
    result = subprocess.run(
        ["xacro", path.name],
        cwd=path.parent,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def parse_urdf_joints(urdf_text: str) -> dict[str, Joint]:
    root = ET.fromstring(urdf_text)
    child_to_joint: dict[str, Joint] = {}

    for elem in root.findall("joint"):
        name = elem.attrib["name"]
        joint_type = elem.attrib.get("type", "fixed")
        parent = elem.find("parent").attrib["link"]
        child = elem.find("child").attrib["link"]

        origin_elem = elem.find("origin")
        xyz = parse_vec(origin_elem.attrib.get("xyz") if origin_elem is not None else None, [0.0, 0.0, 0.0])
        rpy = parse_vec(origin_elem.attrib.get("rpy") if origin_elem is not None else None, [0.0, 0.0, 0.0])

        axis_elem = elem.find("axis")
        axis = normalize(parse_vec(axis_elem.attrib.get("xyz") if axis_elem is not None else None, [1.0, 0.0, 0.0]))

        lower = None
        upper = None
        limit_elem = elem.find("limit")
        if joint_type != "continuous" and limit_elem is not None:
            if "lower" in limit_elem.attrib:
                lower = float(limit_elem.attrib["lower"])
            if "upper" in limit_elem.attrib:
                upper = float(limit_elem.attrib["upper"])

        child_to_joint[child] = Joint(
            name=name,
            type=joint_type,
            parent=parent,
            child=child,
            origin=matrix_from_xyz_rpy(xyz, rpy),
            axis=axis,
            lower=lower,
            upper=upper,
        )

    return child_to_joint


def find_chain(child_to_joint: dict[str, Joint], base_link: str, ee_link: str) -> list[Joint]:
    reverse_chain = []
    link = ee_link
    visited = set()

    while link != base_link:
        if link in visited:
            raise RuntimeError(f"URDF 链路存在环: {link}")
        visited.add(link)
        if link not in child_to_joint:
            raise ValueError(f"找不到从 {base_link!r} 到 {ee_link!r} 的链：{link!r} 没有父 joint")
        joint = child_to_joint[link]
        reverse_chain.append(joint)
        link = joint.parent

    return list(reversed(reverse_chain))


def is_active_joint(joint: Joint) -> bool:
    return joint.type in ("revolute", "continuous", "prismatic")


def configured_fixed_joint_values(chain: Sequence[Joint]) -> dict[str, float]:
    values = {}
    for joint in chain:
        if joint.name != WAIST_JOINT:
            continue
        value = WAIST_ANGLE
        if WAIST_ANGLE_DEG and joint.type in ("revolute", "continuous"):
            value = math.radians(value)
        values[joint.name] = float(value)
    return values


def seed_vector(active_joints: Sequence[Joint]) -> np.ndarray:
    if len(SEED_JOINTS) != len(active_joints):
        names = ", ".join(joint.name for joint in active_joints)
        raise ValueError(
            f"SEED_JOINTS 长度错误: {len(SEED_JOINTS)} != {len(active_joints)}；"
            f"当前顺序为: {names}"
        )

    values = []
    for value, joint in zip(SEED_JOINTS, active_joints):
        if SEED_JOINTS_DEG and joint.type in ("revolute", "continuous"):
            value = math.radians(value)
        values.append(value)
    return clip_to_limits(np.array(values, dtype=float), active_joints)


def clip_to_limits(q: np.ndarray, active_joints: Sequence[Joint]) -> np.ndarray:
    out = q.copy()
    for i, joint in enumerate(active_joints):
        if joint.lower is not None:
            out[i] = max(out[i], joint.lower)
        if joint.upper is not None:
            out[i] = min(out[i], joint.upper)
    return out


def forward_kinematics(
    chain: Sequence[Joint],
    active_joints: Sequence[Joint],
    q: Sequence[float],
    fixed_values: dict[str, float] | None = None,
) -> tuple[np.ndarray, list[tuple[Joint, np.ndarray, np.ndarray]]]:
    q_by_name = {joint.name: float(value) for joint, value in zip(active_joints, q)}
    if fixed_values:
        q_by_name.update(fixed_values)
    active_names = {joint.name for joint in active_joints}
    t = np.eye(4)
    joint_frames = []

    for joint in chain:
        t_origin = t @ joint.origin
        if is_active_joint(joint):
            axis_base = t_origin[:3, :3] @ joint.axis
            if joint.name in active_names:
                joint_frames.append((joint, t_origin[:3, 3].copy(), axis_base))
            value = q_by_name[joint.name]
            if joint.type in ("revolute", "continuous"):
                t = t_origin @ matrix_from_axis_angle(joint.axis, value)
            elif joint.type == "prismatic":
                t = t_origin @ matrix_from_axis_translation(joint.axis, value)
            else:
                t = t_origin
        else:
            t = t_origin

    return t, joint_frames


def pose_error(target: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos_err = target[:3, 3] - current[:3, 3]
    rot_err = target[:3, :3] @ current[:3, :3].T
    rot_vec = Rotation.from_matrix(rot_err).as_rotvec()
    return pos_err, rot_vec


def solve_ik(
    chain: Sequence[Joint],
    active_joints: Sequence[Joint],
    target_xyz: Sequence[float],
    target_rpy: Sequence[float],
    seed: Sequence[float],
    fixed_values: dict[str, float],
) -> tuple[bool, np.ndarray, int, float, float]:
    q = clip_to_limits(np.array(seed, dtype=float), active_joints)
    target = matrix_from_xyz_rpy(target_xyz, target_rpy)

    last_pos_err = float("inf")
    last_ori_err = float("inf")
    for iteration in range(1, MAX_ITERS + 1):
        current, joint_frames = forward_kinematics(chain, active_joints, q, fixed_values)
        pos_err, rot_err = pose_error(target, current)
        last_pos_err = float(np.linalg.norm(pos_err))
        last_ori_err = float(np.linalg.norm(rot_err))
        if last_pos_err <= POS_TOL and last_ori_err <= ORI_TOL:
            return True, q, iteration, last_pos_err, last_ori_err

        p_ee = current[:3, 3]
        jac = np.zeros((6, len(active_joints)), dtype=float)
        for i, (joint, origin, axis) in enumerate(joint_frames):
            if joint.type in ("revolute", "continuous"):
                jac[:3, i] = np.cross(axis, p_ee - origin)
                jac[3:, i] = axis
            elif joint.type == "prismatic":
                jac[:3, i] = axis
                jac[3:, i] = 0.0

        error = np.concatenate([pos_err, ORIENTATION_WEIGHT * rot_err])
        weighted_jac = jac.copy()
        weighted_jac[3:, :] *= ORIENTATION_WEIGHT

        lhs = weighted_jac @ weighted_jac.T + (DAMPING * DAMPING) * np.eye(6)
        dq = weighted_jac.T @ np.linalg.solve(lhs, error)

        step_norm = float(np.linalg.norm(dq, ord=np.inf))
        if step_norm > MAX_STEP:
            dq *= MAX_STEP / step_norm
        q = clip_to_limits(q + dq, active_joints)

    return False, q, MAX_ITERS, last_pos_err, last_ori_err


def format_named_values(names: Sequence[str], values: Sequence[float], degrees: bool = False) -> str:
    rows = []
    for name, value in zip(names, values):
        shown = math.degrees(value) if degrees else value
        rows.append(f"{name}={shown:.9f}")
    return ", ".join(rows)


def full_solution_from_chain(
    chain: Sequence[Joint],
    active_joints: Sequence[Joint],
    q: Sequence[float],
    fixed_values: dict[str, float],
) -> tuple[list[str], list[float]]:
    active_values = {joint.name: float(value) for joint, value in zip(active_joints, q)}
    names = []
    values = []
    for joint in chain:
        if not is_active_joint(joint):
            continue
        if joint.name in fixed_values:
            names.append(joint.name)
            values.append(fixed_values[joint.name])
        elif joint.name in active_values:
            names.append(joint.name)
            values.append(active_values[joint.name])
    return names, values


def main() -> int:
    target_rpy = np.array(TARGET_RPY, dtype=float)
    if TARGET_RPY_DEG:
        target_rpy = np.deg2rad(target_rpy)

    child_to_joint = parse_urdf_joints(expand_xacro(URDF_XACRO))
    chain = find_chain(child_to_joint, BASE_LINK, EE_LINK)
    fixed_values = configured_fixed_joint_values(chain)
    active_joints = [
        joint for joint in chain
        if is_active_joint(joint) and joint.name not in fixed_values
    ]
    if not active_joints:
        raise RuntimeError(f"{BASE_LINK!r} -> {EE_LINK!r} 链上没有可动关节")

    seed = seed_vector(active_joints)
    ok, q, iters, pos_err, ori_err = solve_ik(
        chain, active_joints, TARGET_XYZ, target_rpy, seed, fixed_values
    )
    fk, _ = forward_kinematics(chain, active_joints, q, fixed_values)
    fk_rpy = rpy_from_matrix(fk[:3, :3])
    active_names = [joint.name for joint in active_joints]
    fixed_names = list(fixed_values.keys())
    fixed_q = list(fixed_values.values())
    full_names, full_q = full_solution_from_chain(chain, active_joints, q, fixed_values)

    print(f"success: {ok}")
    print(f"iterations: {iters}")
    # print(f"base_link: {BASE_LINK}")
    # print(f"ee_link: {EE_LINK}")
    # print("chain_joints:", " -> ".join(joint.name for joint in chain))
    print("active_joints:", ", ".join(active_names))
    # print("target_xyz:", " ".join(f"{v:.9f}" for v in TARGET_XYZ))
    # print("target_rpy_rad:", " ".join(f"{v:.9f}" for v in target_rpy))
    # print("seed_rad:", format_named_values(active_names, seed))
    if fixed_names:
        print("fixed_joints_rad:", format_named_values(fixed_names, fixed_q))
        print("fixed_joints_deg:", format_named_values(fixed_names, fixed_q, degrees=True))
    print("solution_rad:", format_named_values(active_names, q))
    print("solution_deg:", format_named_values(active_names, q, degrees=True))
    print("full_solution_rad:", format_named_values(full_names, full_q))
    print("full_solution_deg:", format_named_values(full_names, full_q, degrees=True))
    # print(
    #     "fk_ee_xyz:",
    #     f"{fk[0, 3]:.9f}",
    #     f"{fk[1, 3]:.9f}",
    #     f"{fk[2, 3]:.9f}",
    # )
    # print("fk_ee_rpy_rad:", " ".join(f"{v:.9f}" for v in fk_rpy))
    print(f"position_error_m: {pos_err:.9g}")
    print(f"orientation_error_rad: {ori_err:.9g}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
