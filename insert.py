"""使用 NumPy 实现三轴质量、阻尼、刚度导纳控制。"""

import numpy as np


# 对应 C++ 的 static 变量，多次调用共享同一份状态。
_x = np.zeros(3, dtype=float)
_dx = np.zeros(3, dtype=float)
_integral = np.zeros(3, dtype=float)


def admittance_ctrl_REI_classical(fd, fe, Be, dt):
    """更新一个周期，返回累计位置偏移的副本。

    fd、fe 为同一坐标系下的三维力向量，dt 为周期（秒）。
    与原代码一致，传入的 Be 会被固定值 [60, 60, 60] 覆盖。
    """
    global _x, _dx, _integral

    fd = np.asarray(fd, dtype=float)
    fe = np.asarray(fe, dtype=float)
    dt = float(dt)
    if fd.shape != (3,) or fe.shape != (3,):
        raise ValueError("fd 和 fe 必须为三维向量")
    if not np.all(np.isfinite(fd)) or not np.all(np.isfinite(fe)):
        raise ValueError("fd 和 fe 必须只包含有限数")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt 必须为大于 0 的有限数")

    Me = np.array([20.0, 20.0, 20.0])
    Be = np.array([60.0, 60.0, 60.0])  # 保留原代码覆盖 Be 的行为。
    Ke = np.array([45.0, 45.0, 45.0])
    ki = 0.5

    error = fe - fd
    _integral += error * dt
    ddx = (error + ki * _integral) / Me - Be * _dx / Me - Ke * _x / Me
    _dx += ddx * dt
    _x += _dx * dt
    return _x.copy()


def reset_admittance_ctrl_REI_classical():
    """开始新任务前，清零累计位置、速度和力误差积分。"""
    _x.fill(0.0)
    _dx.fill(0.0)
    _integral.fill(0.0)
