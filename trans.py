"""根据改进 DH 参数生成 URDF 风格的相邻坐标系变换。

读取 `moveit_resources/g01_description/urdf/m_dh.json`，在关节角全为 0 时
按 Craig 改进 DH 约定计算每个相邻坐标系的变换，并只保存 `xyz` 和 `rpy`
到 `m_dh_transforms.json`。
"""

import json
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
PARAMS_PATH = BASE_DIR / "moveit_resources/g01_description/urdf/m_dh.json"
OUTPUT_PATH = BASE_DIR / "moveit_resources/g01_description/urdf/m_dh_transforms.json"
MDH_KEYS = ("a(i-1)", "d(i)", "alpha(i-1)", "theta(i)")


def modified_dh_matrix(a, alpha, d, theta):
    """Craig modified DH: Rx(alpha) * Tx(a) * Rz(theta) * Tz(d)."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)

    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def rpy_from_matrix(matrix):
    pitch = np.arctan2(-matrix[2, 0], np.hypot(matrix[0, 0], matrix[1, 0]))
    if abs(np.cos(pitch)) < 1e-9:
        roll = 0.0
        yaw = np.arctan2(-matrix[0, 1], matrix[1, 1])
    else:
        roll = np.arctan2(matrix[2, 1], matrix[2, 2])
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    return np.array([roll, pitch, yaw])


def fk_mdh(params, joint_angles_deg):
    a_values = np.array(params["a(i-1)"], dtype=float) / 1000.0
    d_values = np.array(params["d(i)"], dtype=float) / 1000.0
    alpha_values = np.deg2rad(np.array(params["alpha(i-1)"], dtype=float))
    theta_offsets = np.array(params["theta(i)"], dtype=float)

    if len(joint_angles_deg) != len(a_values):
        raise ValueError(f"需要 {len(a_values)} 个关节角，当前输入 {len(joint_angles_deg)} 个")

    theta_values = np.deg2rad(theta_offsets + np.array(joint_angles_deg, dtype=float))
    relative_transforms = {}

    for index, (a, alpha, d, theta) in enumerate(zip(a_values, alpha_values, d_values, theta_values), start=1):
        link_transform = modified_dh_matrix(a, alpha, d, theta)
        xyz = link_transform[:3, 3]
        rpy = rpy_from_matrix(link_transform)
        relative_transforms[f"frame_{index - 1}_to_frame_{index}"] = {
            "xyz": xyz.tolist(),
            "rpy": rpy.tolist(),
        }

    return relative_transforms


def is_mdh_param_block(value):
    return isinstance(value, dict) and all(key in value for key in MDH_KEYS)


with PARAMS_PATH.open("r", encoding="utf-8") as file:
    mdh_data = json.load(file)

joint_angles_deg = [0, 0, 0, 0, 0, 0]
if is_mdh_param_block(mdh_data):
    transforms_by_id = fk_mdh(mdh_data, joint_angles_deg)
else:
    transforms_by_id = {
        str(serial): fk_mdh(params, joint_angles_deg)
        for serial, params in mdh_data.items()
    }

with OUTPUT_PATH.open("w", encoding="utf-8") as file:
    json.dump(transforms_by_id, file, indent=2, ensure_ascii=False)

np.set_printoptions(precision=9, suppress=True)
if is_mdh_param_block(mdh_data):
    for name, transform in transforms_by_id.items():
        print(name)
        print("xyz [m]:", np.array(transform["xyz"]))
        print("rpy [rad]:", np.array(transform["rpy"]))
        print()
else:
    for serial, relative_transforms in transforms_by_id.items():
        print(serial)
        for name, transform in relative_transforms.items():
            print(name)
            print("xyz [m]:", np.array(transform["xyz"]))
            print("rpy [rad]:", np.array(transform["rpy"]))
        print()

print(f"Saved JSON: {OUTPUT_PATH}")
