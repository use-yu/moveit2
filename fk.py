#!/usr/bin/env python3
"""CR7 forward kinematics from the xacro/URDF model.

By default command-line angles are degrees. Use --radians for ROS/MoveIt
joint-state values.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_URDF = (
    Path(__file__).resolve().parent
    / "moveit_resources/g01_description/urdf/cr7_robot.urdf"
)


def _matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _identity() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _origin_matrix(xyz: Sequence[float], rpy: Sequence[float]) -> list[list[float]]:
    x, y, z = xyz
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, x],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, y],
        [-sp, cp * sr, cp * cr, z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _axis_rotation(axis: Sequence[float], angle: float) -> list[list[float]]:
    x, y, z = axis
    norm = math.sqrt(x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError(f"invalid joint axis: {axis}")

    x, y, z = x / norm, y / norm, z / norm
    c, s = math.cos(angle), math.sin(angle)
    one_c = 1.0 - c
    return [
        [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s, 0.0],
        [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s, 0.0],
        [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _rpy_from_matrix(matrix: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    pitch = math.atan2(-matrix[2][0], math.hypot(matrix[0][0], matrix[1][0]))
    if abs(math.cos(pitch)) < 1e-9:
        roll = 0.0
        yaw = math.atan2(-matrix[0][1], matrix[1][1])
    else:
        roll = math.atan2(matrix[2][1], matrix[2][2])
        yaw = math.atan2(matrix[1][0], matrix[0][0])
    return roll, pitch, yaw


def _numbers(value: str | None, default: Sequence[float]) -> list[float]:
    if not value:
        return list(default)
    return [float(part) for part in value.split()]


def _expand_cr7_xacro(urdf_path: str | Path = DEFAULT_URDF) -> ET.Element:
    urdf_path = Path(urdf_path).resolve()
    wrapper = f"""<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="cr7_fk">
  <xacro:include filename="{urdf_path}" />
  <link name="world" />
  <xacro:cr7_robot
    prefix=""
    link_prefix="Link"
    joint_prefix="joint"
    parent="world"
    xyz="0 0 0"
    rpy="0 0 0" />
</robot>
"""
    wrapper_file = tempfile.NamedTemporaryFile(
        prefix=".cr7_fk_",
        suffix=".urdf.xacro",
        dir=urdf_path.parent,
        mode="w",
        encoding="utf-8",
        delete=False,
    )
    wrapper_path = Path(wrapper_file.name)
    try:
        wrapper_file.write(wrapper)
        wrapper_file.close()
        try:
            result = subprocess.run(
                ["xacro", str(wrapper_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("xacro command not found; source your ROS environment first") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(exc.stderr.strip() or exc.stdout.strip()) from exc
    finally:
        if not wrapper_file.closed:
            wrapper_file.close()
        wrapper_path.unlink(missing_ok=True)
    return ET.fromstring(result.stdout)


def _joint_info(elem: ET.Element) -> dict[str, object]:
    origin = elem.find("origin")
    axis = elem.find("axis")
    parent = elem.find("parent")
    child = elem.find("child")
    if parent is None or child is None:
        raise ValueError(f"joint {elem.attrib.get('name', '<unnamed>')} is missing parent/child")

    return {
        "name": elem.attrib.get("name", ""),
        "type": elem.attrib.get("type", "fixed"),
        "parent": parent.attrib["link"],
        "child": child.attrib["link"],
        "xyz": _numbers(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0)),
        "rpy": _numbers(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0)),
        "axis": _numbers(axis.attrib.get("xyz") if axis is not None else None, (0.0, 0.0, 1.0)),
    }


def _load_chain(
    urdf_path: str | Path = DEFAULT_URDF,
    *,
    include_tool: bool = True,
) -> list[dict[str, object]]:
    root = _expand_cr7_xacro(urdf_path)
    joints_by_parent: dict[str, list[dict[str, object]]] = defaultdict(list)
    for elem in root.iter("joint"):
        joint = _joint_info(elem)
        joints_by_parent[str(joint["parent"])].append(joint)

    target_link = "tool" if include_tool else "Link6"
    current_link = "base_link"
    chain: list[dict[str, object]] = []
    while current_link != target_link:
        candidates = joints_by_parent.get(current_link, [])
        if len(candidates) != 1:
            names = [str(joint["name"]) for joint in candidates]
            raise ValueError(f"expected one child joint from {current_link}, got {names}")
        joint = candidates[0]
        chain.append(joint)
        current_link = str(joint["child"])

    return chain


def fk(
    joint_angles: Sequence[float] | Iterable[float] | float,
    *more_angles: float,
    urdf_path: str | Path = DEFAULT_URDF,
    include_tool: bool = True,
    degrees: bool = True,
) -> dict[str, object]:
    """Compute CR7 FK from base_link to tool, or to Link6 with include_tool=False."""
    if more_angles:
        q = [float(joint_angles), *[float(value) for value in more_angles]]
    elif isinstance(joint_angles, (int, float)):
        q = [float(joint_angles)]
    else:
        q = [float(value) for value in joint_angles]

    if len(q) != 6:
        raise ValueError(f"fk() needs 6 joint angles, got {len(q)}")
    if degrees:
        q = [math.radians(value) for value in q]

    q_by_joint = {f"joint{i + 1}": q[i] for i in range(6)}
    transform = _identity()

    for joint in _load_chain(urdf_path, include_tool=include_tool):
        transform = _matmul(transform, _origin_matrix(joint["xyz"], joint["rpy"]))
        if joint["type"] != "fixed":
            transform = _matmul(transform, _axis_rotation(joint["axis"], q_by_joint[str(joint["name"])]))

    xyz = (transform[0][3], transform[1][3], transform[2][3])
    rpy = _rpy_from_matrix(transform)
    return {
        "matrix": transform,
        "xyz": xyz,
        "rpy": rpy,
        "rpy_deg": tuple(math.degrees(value) for value in rpy),
        "end_link": "tool" if include_tool else "Link6",
    }


def fk6(q1: float, q2: float, q3: float, q4: float, q5: float, q6: float) -> dict[str, object]:
    return fk(q1, q2, q3, q4, q5, q6)


def _main() -> int:
    parser = argparse.ArgumentParser(description="Compute CR7 forward kinematics")
    parser.add_argument("q", nargs="*", type=float, help="joint1..joint6 angles; default unit is degree")
    parser.add_argument("--radians", action="store_true", help="input angles are radians")
    parser.add_argument("--link6", action="store_true", help="return FK to Link6 instead of tool")
    parser.add_argument("--urdf", default=str(DEFAULT_URDF), help="path to cr7_robot.urdf xacro file")
    args = parser.parse_args()

    if args.q:
        if len(args.q) != 6:
            parser.error(f"需要输入 6 个关节角度，现在是 {len(args.q)} 个")
        joint_angles = args.q
    else:
        joint_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    result = fk(
        joint_angles,
        urdf_path=args.urdf,
        include_tool=not args.link6,
        degrees=not args.radians,
    )

    print(f"transform: base_link -> {result['end_link']}")
    print("matrix:")
    for row in result["matrix"]:
        print("  " + " ".join(f"{value: .9f}" for value in row))
    print(f"xyz [m]:     {tuple(round(value, 9) for value in result['xyz'])}")
    print(f"rpy [rad]:   {tuple(round(value, 9) for value in result['rpy'])}")
    print(f"rpy [deg]:   {tuple(round(value, 6) for value in result['rpy_deg'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
