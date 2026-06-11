import numpy as np


# 标准 DH / Standard DH 参数：
# 每一行按 A_i = Rz(theta_i) * Tz(d_i) * Tx(a_i) * Rx(alpha_i) 计算。
# a、d 单位从 mm 转为 m；alpha、theta 单位从 deg 转为 rad。
dh_params = [
    {"frame_name": "joint1", "joint_type": "r", "link_length": 0 / 1000, "twist": np.deg2rad(0), "offset": 147 / 1000, "theta": 0},
    {"frame_name": "joint2", "joint_type": "r", "link_length": -0.018278000876307487 / 1000, "twist": np.deg2rad(89.936286926269531), "offset": 0 / 1000, "theta": 0},
    {"frame_name": "joint3", "joint_type": "r", "link_length": -377.67788696289062 / 1000, "twist": np.deg2rad(-0.40807899832725525), "offset": 0 / 1000, "theta": 0},
    {"frame_name": "joint4", "joint_type": "r", "link_length": -307.05142211914062 / 1000, "twist": np.deg2rad(0.388377994298935), "offset": 141.59689331054688 / 1000, "theta": 0},
    {"frame_name": "joint5", "joint_type": "r", "link_length": -0.033348001539707184 / 1000, "twist": np.deg2rad(89.987556457519531), "offset": 116.02764129638672 / 1000, "theta": 0},
    {"frame_name": "joint6", "joint_type": "r", "link_length": 0.19047899544239044 / 1000, "twist": np.deg2rad(-89.9672393798828), "offset": 105 / 1000, "theta": 0},
]


def standard_dh_matrix(a, alpha, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)

    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def forward_kinematics_dh(params, joint_angles_deg):
    if len(params) != len(joint_angles_deg):
        raise ValueError(f"需要 {len(params)} 个关节角，当前输入 {len(joint_angles_deg)} 个")

    transform = np.eye(4)
    link_transforms = []

    for param, joint_angle_deg in zip(params, joint_angles_deg):
        theta = np.deg2rad(joint_angle_deg) + param["theta"]
        link_transform = standard_dh_matrix(
            param["link_length"],
            param["twist"],
            param["offset"],
            theta,
        )
        link_transforms.append(link_transform)
        transform = transform @ link_transform

    return transform, link_transforms


# 输入一组关节角度，单位：度
joint_angles_deg = [0, 0, 0, 0, 0, 0]
transform_matrix, link_transforms = forward_kinematics_dh(dh_params, joint_angles_deg)

np.set_printoptions(precision=9, suppress=True)
print("Standard-DH End-Effector Transformation Matrix:")
print(transform_matrix)
