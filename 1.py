import numpy as np
from mdh.kinematic_chain import KinematicChain

# MDH参数: [alpha, a, theta, d], 单位: 弧度, 米
dh = [
    [0, 0, 0, 0.147],
    [np.deg2rad(89.9363), -1.8278e-5, 0, 0],
    [np.deg2rad(-0.40808), -0.37768, 0, 0],
    [np.deg2rad(0.388378), -0.30705, 0, 0.1416],
    [np.deg2rad(89.9876), -3.3348e-5, 0, 0.11603],
    [np.deg2rad(-89.9672), 0.0001905, 0, 0.105]
]
# 注意：这里每个关节的type=1表示旋转关节
dh_dicts = [{'alpha': p[0], 'a': p[1], 'theta': p[2], 'd': p[3], 'type': 1} for p in dh]

kc = KinematicChain.from_parameters(dh_dicts)
T = kc.forward([0, 0, 0, 0, 0, 0])  # 零位验证
print(T)