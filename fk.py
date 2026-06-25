#!/usr/bin/env python3
"""URDF forward kinematics helper.

可以计算 URDF 中任意两个 link 之间的 FK。例如：

    python3 fk.py SJ r_tool --radians 0 0 0 0 0 0 0

默认命令行角度单位是 degree；ROS/MoveIt 的 joint-state 值请加 --radians。
"""

from __future__ import annotations

import argparse
import math
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_G01_URDF = (
    Path(__file__).resolve().parent
    / "moveit_resources/g01_description/urdf/G01-URDF888.urdf"
)
DEFAULT_CR7_URDF = (
    Path(__file__).resolve().parent
    / "DOBOT_6Axis_ROS2/dobot_rviz/urdf/cr7_robot.urdf"
)
DEFAULT_URDF = DEFAULT_G01_URDF

# 这里改默认 FK 起点/终点。
# 直接运行 python3 fk.py 时，等价于计算 DEFAULT_BASE_LINK -> DEFAULT_TARGET_LINK。
DEFAULT_BASE_LINK = "SJ"
DEFAULT_TARGET_LINK = "r_tool"
DEFAULT_JOINT_VALUES = [
    30.1,  # body_joint2
    -38,  # r_arm_joint1
    65,  # r_arm_joint2
    87,  # r_arm_joint3
    27,  # r_arm_joint4
    142,  # r_arm_joint5
    0.0,  # r_arm_joint6
]

# 下面三个只用于兼容旧 CR7 参数 --tool / --link6 / fk6。
DEFAULT_LEGACY_BASE_LINK = "base_link"
DEFAULT_TOOL_LINK = "tool"
DEFAULT_LINK6_LINK = "Link6"

MOVABLE_JOINT_TYPES = {"revolute", "continuous", "prismatic"}

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


def _axis_translation(axis: Sequence[float], distance: float) -> list[list[float]]:
    x, y, z = axis
    norm = math.sqrt(x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError(f"invalid joint axis: {axis}")

    x, y, z = x / norm, y / norm, z / norm
    return [
        [1.0, 0.0, 0.0, x * distance],
        [0.0, 1.0, 0.0, y * distance],
        [0.0, 0.0, 1.0, z * distance],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _inverse_transform(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    rotation_t = [[matrix[j][i] for j in range(3)] for i in range(3)]
    translation = [matrix[i][3] for i in range(3)]
    inverse_translation = [
        -sum(rotation_t[i][j] * translation[j] for j in range(3))
        for i in range(3)
    ]
    return [
        [rotation_t[0][0], rotation_t[0][1], rotation_t[0][2], inverse_translation[0]],
        [rotation_t[1][0], rotation_t[1][1], rotation_t[1][2], inverse_translation[1]],
        [rotation_t[2][0], rotation_t[2][1], rotation_t[2][2], inverse_translation[2]],
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


def _load_urdf_root(urdf_path: str | Path = DEFAULT_URDF) -> ET.Element:
    urdf_path = Path(urdf_path).resolve()
    text = urdf_path.read_text(encoding="utf-8")
    if "xacro:" not in text and urdf_path.suffix != ".xacro":
        return ET.fromstring(text)

    try:
        result = subprocess.run(
            ["xacro", str(urdf_path)],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(urdf_path.parent),
        )
    except FileNotFoundError as exc:
        raise RuntimeError("xacro command not found; source your ROS environment first") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.strip() or exc.stdout.strip()) from exc
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


def _is_movable_joint(joint: dict[str, object]) -> bool:
    return str(joint["type"]) in MOVABLE_JOINT_TYPES


def _joint_transform(joint: dict[str, object], value: float) -> list[list[float]]:
    transform = _origin_matrix(joint["xyz"], joint["rpy"])
    joint_type = str(joint["type"])

    if joint_type == "fixed":
        return transform
    if joint_type in {"revolute", "continuous"}:
        return _matmul(transform, _axis_rotation(joint["axis"], value))
    if joint_type == "prismatic":
        return _matmul(transform, _axis_translation(joint["axis"], value))
    raise ValueError(f"unsupported joint type {joint_type!r} for joint {joint['name']!r}")


def _find_joint_path(
    root: ET.Element,
    base_link: str,
    target_link: str,
) -> list[tuple[dict[str, object], int]]:
    link_names = {elem.attrib["name"] for elem in root.iter("link") if "name" in elem.attrib}
    missing_links = [name for name in (base_link, target_link) if name not in link_names]
    if missing_links:
        available = ", ".join(sorted(link_names)[:20])
        raise ValueError(
            f"link not found: {', '.join(missing_links)}. "
            f"available examples: {available}"
        )

    if base_link == target_link:
        return []

    graph: dict[str, list[tuple[str, dict[str, object], int]]] = defaultdict(list)
    for elem in root.iter("joint"):
        joint = _joint_info(elem)
        parent = str(joint["parent"])
        child = str(joint["child"])
        graph[parent].append((child, joint, 1))
        graph[child].append((parent, joint, -1))

    queue: deque[tuple[str, list[tuple[dict[str, object], int]]]] = deque()
    queue.append((base_link, []))
    visited = {base_link}

    while queue:
        current_link, path = queue.popleft()
        for next_link, joint, direction in graph[current_link]:
            if next_link in visited:
                continue
            next_path = [*path, (joint, direction)]
            if next_link == target_link:
                return next_path
            visited.add(next_link)
            queue.append((next_link, next_path))

    raise ValueError(f"no joint path found from {base_link!r} to {target_link!r}")


def _load_chain(
    urdf_path: str | Path = DEFAULT_URDF,
    *,
    base_link: str = "base_link",
    target_link: str = "r_tool",
) -> list[tuple[dict[str, object], int]]:
    root = _load_urdf_root(urdf_path)
    return _find_joint_path(root, base_link, target_link)


def _coerce_joint_values(
    joint_angles: Sequence[float] | Iterable[float] | float,
    more_angles: Sequence[float],
) -> list[float]:
    if more_angles:
        return [float(joint_angles), *[float(value) for value in more_angles]]
    if isinstance(joint_angles, (int, float)):
        return [float(joint_angles)]
    return [float(value) for value in joint_angles]


def fk_between(
    base_link: str,
    target_link: str,
    joint_angles: Sequence[float] | Iterable[float] | float = (),
    *more_angles: float,
    urdf_path: str | Path = DEFAULT_URDF,
    degrees: bool = True,
    joint_names: Sequence[str] | None = None,
) -> dict[str, object]:
    """Compute FK from ``base_link`` to ``target_link``.

    如果没有传 ``joint_names``，关节角会按 URDF 中 base->target 路径上的
    活动关节顺序匹配。revolute/continuous 关节受 ``degrees`` 控制，
    prismatic 关节始终按 URDF 单位（通常是 m）输入。
    """
    q = _coerce_joint_values(joint_angles, more_angles)

    root = _load_urdf_root(urdf_path)
    path = _find_joint_path(root, base_link, target_link)
    active_joints = [str(joint["name"]) for joint, _ in path if _is_movable_joint(joint)]

    if not q and base_link == DEFAULT_BASE_LINK and target_link == DEFAULT_TARGET_LINK:
        q = list(DEFAULT_JOINT_VALUES)
        if len(q) != len(active_joints):
            raise ValueError(
                "DEFAULT_JOINT_VALUES length does not match "
                f"{base_link}->{target_link}: expected {len(active_joints)}, got {len(q)}. "
                f"Path joints: {', '.join(active_joints)}"
            )
    elif not q:
        q = [0.0] * len(active_joints)

    if joint_names is None:
        if len(q) != len(active_joints):
            raise ValueError(
                f"{base_link}->{target_link} needs {len(active_joints)} joint values, "
                f"got {len(q)}. Path joints: {', '.join(active_joints)}"
            )
        q_by_joint = dict(zip(active_joints, q))
    else:
        names = [str(name) for name in joint_names]
        if len(q) != len(names):
            raise ValueError(f"joint_names has {len(names)} names, but got {len(q)} values")
        q_by_joint = dict(zip(names, q))
        missing = [name for name in active_joints if name not in q_by_joint]
        if missing:
            raise ValueError(f"missing values for path joints: {', '.join(missing)}")

    if degrees:
        joint_type_by_name = {str(joint["name"]): str(joint["type"]) for joint, _ in path}
        q_by_joint = {
            name: math.radians(value)
            if joint_type_by_name.get(name) in {"revolute", "continuous"}
            else value
            for name, value in q_by_joint.items()
        }

    transform = _identity()
    for joint, direction in path:
        value = q_by_joint[str(joint["name"])] if _is_movable_joint(joint) else 0.0
        step = _joint_transform(joint, value)
        if direction < 0:
            step = _inverse_transform(step)
        transform = _matmul(transform, step)

    xyz = (transform[0][3], transform[1][3], transform[2][3])
    rpy = _rpy_from_matrix(transform)
    return {
        "matrix": transform,
        "xyz": xyz,
        "rpy": rpy,
        "rpy_deg": tuple(math.degrees(value) for value in rpy),
        "base_link": base_link,
        "target_link": target_link,
        "end_link": target_link,
        "joint_names": active_joints,
        "joint_values": tuple(q_by_joint[name] for name in active_joints),
    }


def fk(
    joint_angles: Sequence[float] | Iterable[float] | float,
    *more_angles: float,
    urdf_path: str | Path = DEFAULT_URDF,
    include_tool: bool = False,
    degrees: bool = True,
    base_link: str = DEFAULT_LEGACY_BASE_LINK,
    target_link: str | None = None,
    joint_names: Sequence[str] | None = None,
) -> dict[str, object]:
    """Backward-compatible FK wrapper.

    新代码建议直接用 ``fk_between(base_link, target_link, q...)``。
    """
    if target_link is None:
        target_link = DEFAULT_TOOL_LINK if include_tool else DEFAULT_LINK6_LINK
    return fk_between(
        base_link,
        target_link,
        joint_angles,
        *more_angles,
        urdf_path=urdf_path,
        degrees=degrees,
        joint_names=joint_names,
    )


def fk6(q1: float, q2: float, q3: float, q4: float, q5: float, q6: float) -> dict[str, object]:
    return fk(
        q1,
        q2,
        q3,
        q4,
        q5,
        q6,
        urdf_path=DEFAULT_CR7_URDF,
        base_link=DEFAULT_LEGACY_BASE_LINK,
        target_link=DEFAULT_LINK6_LINK,
    )


def _parse_joint_names(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part for part in value.replace(",", " ").split() if part]


def _is_float_text(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _default_base_link(args: argparse.Namespace) -> str:
    return DEFAULT_LEGACY_BASE_LINK if args.tool or args.link6 else DEFAULT_BASE_LINK


def _default_target_link(args: argparse.Namespace) -> str:
    if args.tool:
        return DEFAULT_TOOL_LINK
    if args.link6:
        return DEFAULT_LINK6_LINK
    return DEFAULT_TARGET_LINK


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute URDF forward kinematics",
        epilog=(
            "例子: python3 fk.py SJ r_tool --radians "
            "0 0 0 0 0 0 0"
        ),
    )
    parser.add_argument(
        "items",
        nargs="*",
        help="可写成: base_link target_link q...；或配合 --base/--link 只写 q...",
    )
    parser.add_argument("--base", dest="base_link", help="起始 link，例如 SJ")
    parser.add_argument("--link", "--target", dest="target_link", help="目标 link，例如 r_tool")
    parser.add_argument("--radians", action="store_true", help="input angles are radians")
    parser.add_argument("--joint-names", help="逗号或空格分隔的关节名；默认使用路径上的活动关节顺序")
    parser.add_argument("--list-chain", action="store_true", help="只打印 base->target 路径关节顺序")
    parser.add_argument("--tool", action="store_true", help="兼容旧用法：目标 link 使用 tool")
    parser.add_argument("--link6", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--urdf", default=str(DEFAULT_URDF), help="path to URDF/xacro file")
    parse_args = getattr(parser, "parse_intermixed_args", parser.parse_args)
    args = parse_args()

    items = list(args.items)
    if args.base_link or args.target_link:
        base_link = args.base_link or _default_base_link(args)
        target_link = args.target_link or _default_target_link(args)
        q_text = items
    elif len(items) >= 2 and not _is_float_text(items[0]) and not _is_float_text(items[1]):
        base_link, target_link = items[0], items[1]
        q_text = items[2:]
    else:
        base_link = _default_base_link(args)
        target_link = _default_target_link(args)
        q_text = items

    try:
        joint_angles = [float(value) for value in q_text]
    except ValueError as exc:
        parser.error(f"关节角必须是数字: {exc}")

    if args.list_chain:
        path = _load_chain(args.urdf, base_link=base_link, target_link=target_link)
        print(f"chain: {base_link} -> {target_link}")
        for joint, direction in path:
            arrow = "->" if direction > 0 else "<-"
            moving = "active" if _is_movable_joint(joint) else "fixed"
            print(
                f"  {joint['parent']} {arrow} {joint['child']}  "
                f"{joint['name']} ({joint['type']}, {moving})"
            )
        active = [str(joint["name"]) for joint, _ in path if _is_movable_joint(joint)]
        print(f"active joints ({len(active)}): {', '.join(active)}")
        return 0

    result = fk_between(
        base_link,
        target_link,
        joint_angles,
        urdf_path=args.urdf,
        degrees=not args.radians,
        joint_names=_parse_joint_names(args.joint_names),
    )

    print(f"transform: {result['base_link']} -> {result['target_link']}")
    print(f"active joints: {', '.join(result['joint_names'])}")
    print("matrix:")
    for row in result["matrix"]:
        print("  " + " ".join(f"{value: .9f}" for value in row))
    print(f"xyz [m]:     {tuple(round(value, 9) for value in result['xyz'])}")
    print(f"rpy [rad]:   {tuple(round(value, 9) for value in result['rpy'])}")
    print(f"rpy [deg]:   {tuple(round(value, 6) for value in result['rpy_deg'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
