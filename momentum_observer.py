#!/usr/bin/env python3
"""右臂动量观测器，只打印估计值，不发送运动指令。

运行：python3 momentum_observer.py
订阅 /g01/right_arm/observer_state 和 /robot_description。
也可用 robot_description_file 参数指定展开后的 URDF。
不添加摩擦、力零偏、额外负载或安装方向补偿，保留模型重力项。
"""

import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
import PyKDL as kdl


JOINT_NAMES = [f'r_arm_joint{i}' for i in range(1, 7)]

# 单位和观测器参数集中在这里；ROS 启动参数可覆盖。
DEFAULTS = {
    'observer_gain': 20.0,           # 关节残差增益，1/s
    'print_hz': 20.0,
    'timestamp_scale': 0.001,        # 控制器时间戳若为 ms，则乘 0.001 得到 s
    'velocity_scale': np.pi / 180.,  # qd_actual 若为 deg/s；若已为 rad/s，改为 1.0
    'torque_scale': [1.0] * 6,       # 假设 m_actual 为关节侧 N*m，逐轴比例含符号
    'gravity': [0.0, 0.0, -9.81],    # 竖直安装，重力直接在 r_base_link 下表达
    'wrench_damping': 0.01,          # 力映射阻尼，避免奇异姿态直接求逆
    'max_sample_dt': 0.1,            # 超过此采样间隔，重置观测器
    'feedback_timeout': 0.3,
}


def vector(value, size, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (size,) or not np.all(np.isfinite(value)):
        raise ValueError(f'{name} 必须包含 {size} 个有限数')
    return value


def pose(element):
    xyz = vector(np.fromstring(element.get('xyz', '0 0 0'), sep=' '), 3, 'xyz') if element is not None else np.zeros(3)
    rpy = vector(np.fromstring(element.get('rpy', '0 0 0'), sep=' '), 3, 'rpy') if element is not None else np.zeros(3)
    return kdl.Frame(kdl.Rotation.RPY(*rpy), kdl.Vector(*xyz))


class ArmDynamics:
    """从真实 URDF 惯性构造 KDL 链；固定基座 r_base_link，末端 R6。"""

    def __init__(self, description, gravity):
        root = ET.fromstring(description)
        if any('xacro' in element.tag for element in root.iter()):
            raise ValueError('需要展开后的 URDF，不能直接输入 xacro 宏')
        links = {link.get('name'): link for link in root.findall('link')}
        parents = {joint.find('child').get('link'): joint for joint in root.findall('joint')}
        path, visited = [], set()
        child = 'R6'
        while child != 'r_base_link':
            if child in visited or child not in parents:
                raise ValueError('找不到有效的 r_base_link -> R6 链')
            visited.add(child)
            joint = parents[child]
            path.append((joint, links[child]))
            child = joint.find('parent').get('link')
        self.chain = kdl.Chain()
        names = []
        for joint, link in reversed(path):
            origin = pose(joint.find('origin'))
            kind = joint.get('type')
            if kind == 'fixed':
                kdl_joint = kdl.Joint(joint.get('name'), kdl.Joint.Fixed)
            elif kind in ('revolute', 'continuous'):
                axis_element = joint.find('axis')
                axis = vector(np.fromstring(axis_element.get('xyz', '1 0 0')
                              if axis_element is not None else '1 0 0', sep=' '), 3, 'axis')
                if np.linalg.norm(axis) < 1e-12:
                    raise ValueError('关节轴为零')
                axis /= np.linalg.norm(axis)
                kdl_joint = kdl.Joint(joint.get('name'), origin.p,
                                      origin.M * kdl.Vector(*axis), kdl.Joint.RotAxis)
                names.append(joint.get('name'))
            else:
                raise ValueError(f'不支持的右臂关节类型：{kind}')
            inertial = link.find('inertial')
            if inertial is None:
                if kind != 'fixed':
                    raise ValueError(f'{link.get("name")} 缺少惯性参数')
                inertia = kdl.RigidBodyInertia(0.)
            else:
                inertial_pose = pose(inertial.find('origin'))
                mass = float(inertial.find('mass').get('value'))
                tensor = inertial.find('inertia')
                ixx, iyy, izz, ixy, ixz, iyz = [float(tensor.get(key))
                                               for key in ('ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz')]
                matrix = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
                if not np.isfinite(mass) or mass <= 0 or not np.all(np.isfinite(matrix)):
                    raise ValueError(f'{link.get("name")} 惯性参数无效')
                if np.min(np.linalg.eigvalsh(matrix)) < -1e-10:
                    raise ValueError(f'{link.get("name")} 惯性张量不是半正定')
                rotation = np.array([[inertial_pose.M[i, j] for j in range(3)] for i in range(3)])
                # 惯性张量旋转到 link 坐标系；质心 xyz 已在 link 下，不能再旋转。
                matrix = rotation @ matrix @ rotation.T
                inertia = kdl.RigidBodyInertia(mass, inertial_pose.p, kdl.RotationalInertia(
                    matrix[0, 0], matrix[1, 1], matrix[2, 2],
                    matrix[0, 1], matrix[0, 2], matrix[1, 2]))
            self.chain.addSegment(kdl.Segment(link.get('name'), kdl_joint, origin, inertia))
        if names != JOINT_NAMES:
            raise ValueError('动力学链关节必须按 r_arm_joint1 到 r_arm_joint6 排列')
        self.dynamics = kdl.ChainDynParam(self.chain, kdl.Vector(*gravity))
        self.fk = kdl.ChainFkSolverPos_recursive(self.chain)
        self.jacobian = kdl.ChainJntToJacSolver(self.chain)

    def evaluate(self, q, dq):
        q = vector(q, 6, 'q')
        dq = vector(dq, 6, 'dq')
        q_kdl, dq_kdl = kdl.JntArray(6), kdl.JntArray(6)
        for i in range(6):
            q_kdl[i], dq_kdl[i] = q[i], dq[i]
        mass = kdl.JntSpaceInertiaMatrix(6)
        gravity, coriolis = kdl.JntArray(6), kdl.JntArray(6)
        jacobian, frame = kdl.Jacobian(6), kdl.Frame()
        results = [self.dynamics.JntToMass(q_kdl, mass),
                   self.dynamics.JntToGravity(q_kdl, gravity),
                   self.dynamics.JntToCoriolis(q_kdl, dq_kdl, coriolis),
                   self.jacobian.JntToJac(q_kdl, jacobian), self.fk.JntToCart(q_kdl, frame)]
        if any(result < 0 for result in results):
            raise ValueError(f'KDL 计算失败：{results}')
        M = np.array([[mass[i, j] for j in range(6)] for i in range(6)])
        G = np.array([gravity[i] for i in range(6)])
        Cq = np.array([coriolis[i] for i in range(6)])
        J = np.array([[jacobian[i, j] for j in range(6)] for i in range(6)])
        R = np.array([[frame.M[i, j] for j in range(3)] for i in range(3)])
        if not all(np.all(np.isfinite(v)) for v in (M, G, Cq, J, R)):
            raise ValueError('动力学结果包含无效数值')
        return M, G, Cq, J, R


class MomentumObserver:
    """外力矩符号：M*ddq+Cq+G=tau+tau_ext；不额外补偿摩擦。"""

    def __init__(self, gain=20., timestamp_scale=0.001, max_dt=0.1):
        self.gain, self.timestamp_scale, self.max_dt = gain, timestamp_scale, max_dt
        self.previous_timestamp = None
        self.previous_mass = None
        self.sigma = None
        self.residual = np.zeros(6)

    def reset(self):
        self.previous_timestamp = self.previous_mass = self.sigma = None
        self.residual = np.zeros(6)

    def update(self, M, dq, tau, G, Cq, timestamp):
        momentum = M @ dq
        if self.previous_timestamp is None:
            self.previous_timestamp = timestamp
            self.previous_mass = M.copy()
            self.sigma = momentum.copy()  # 初始化动量，避免非零初速度产生假外力。
            return None
        delta = timestamp - self.previous_timestamp
        if delta == 0:
            return None  # 重复帧不积分。
        dt = delta * self.timestamp_scale
        if dt <= 0 or dt > self.max_dt:
            self.reset()
            return self.update(M, dq, tau, G, Cq, timestamp)
        CTq = ((M - self.previous_mass) / dt) @ dq - Cq
        drive = tau + CTq - G
        # 将残差反馈用后向 Euler 离散，避免 k*dt 较大时显式反馈振荡。
        self.residual = self.gain * (momentum-self.sigma-dt*drive) / (1+self.gain*dt)
        self.sigma = momentum - self.residual/self.gain
        self.previous_mass = M.copy()
        self.previous_timestamp = timestamp
        return self.residual.copy()


def estimate_wrench(J, residual, R, damping):
    # tau_ext=J.T*w_base；正则求解，不显式求逆。
    wrench_base = np.linalg.solve(J @ J.T + damping**2*np.eye(6), J @ residual)
    # 相同参考点 R6，仅变换表达方向，不移动力矩参考点。
    return np.concatenate((R.T @ wrench_base[:3], R.T @ wrench_base[3:]))


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from std_msgs.msg import String

    class ObserverNode(Node):
        def __init__(self):
            super().__init__('right_arm_momentum_observer')

            def parameter(name, default):
                return self.declare_parameter(name, default).value

            def positive(name):
                value = float(parameter(name, DEFAULTS[name]))
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f'{name} 必须为正的有限数')
                return value

            self.gravity = vector(parameter('gravity', DEFAULTS['gravity']), 3, 'gravity')
            self.velocity_scale = positive('velocity_scale')
            self.torque_scale = vector(parameter('torque_scale', DEFAULTS['torque_scale']), 6, 'torque_scale')
            self.timeout = positive('feedback_timeout')
            self.wrench_damping = positive('wrench_damping')
            self.observer = MomentumObserver(positive('observer_gain'), positive('timestamp_scale'),
                                             positive('max_sample_dt'))
            self.model = None
            self.wrench = None
            self.last_feedback = self.last_estimate = None
            description_file = parameter('robot_description_file', '')
            if description_file:
                self.model = ArmDynamics(Path(description_file).read_text(), self.gravity)
            else:
                qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 reliability=ReliabilityPolicy.RELIABLE)
                self.create_subscription(String, '/robot_description', self.on_description, qos)
            self.create_subscription(String, '/g01/right_arm/observer_state', self.on_feedback, 10)
            self.create_timer(1/positive('print_hz'), self.print_wrench)
            self.get_logger().info(
                '等待模型与同帧反馈；输出 R6 坐标系 [Fx,Fy,Fz](N)、[Mx,My,Mz](N*m)。'
                f'速度比例={self.velocity_scale:g}，扭矩比例={self.torque_scale.tolist()}，'
                f'时间戳比例={self.observer.timestamp_scale:g}，重力={self.gravity.tolist()}。'
                '未进行摩擦、零偏或额外负载补偿；这些单位假设必须与固件一致。')

        def on_description(self, msg):
            if self.model is not None:
                return
            try:
                self.model = ArmDynamics(msg.data, self.gravity)
                self.get_logger().info('右臂 KDL 动力学模型就绪。')
            except (ValueError, TypeError, AttributeError, KeyError, ET.ParseError) as exc:
                self.get_logger().error(f'URDF 模型错误：{exc}')

        def on_feedback(self, msg):
            if self.model is None:
                return
            try:
                data = json.loads(msg.data)
                names = data['joint_names']
                if len(names) != 6 or len(set(names)) != 6 or set(names) != set(JOINT_NAMES):
                    raise ValueError('joint_names 与右臂六关节不匹配')
                order = [names.index(name) for name in JOINT_NAMES]
                q = vector(data['position_rad'], 6, 'position_rad')[order]
                dq = vector(data['qd_actual_raw'], 6, 'qd_actual_raw')[order] * self.velocity_scale
                tau = vector(data['m_actual_raw'], 6, 'm_actual_raw')[order] * self.torque_scale
                timestamp = data['controller_time_stamp']
                if type(timestamp) is not int or not 0 <= timestamp < 2**64:
                    raise ValueError('控制器时间戳必须为 uint64 整数')
                now = time.monotonic()
                if self.last_feedback is not None and now-self.last_feedback > self.timeout:
                    self.observer.reset()
                    self.wrench = None
                self.last_feedback = now
                if timestamp == self.observer.previous_timestamp:
                    return
                M, G, Cq, J, R = self.model.evaluate(q, dq)
                residual = self.observer.update(M, dq, tau, G, Cq, timestamp)
                if residual is None:
                    self.wrench = None
                    return
                self.wrench = estimate_wrench(J, residual, R, self.wrench_damping)
                if not np.all(np.isfinite(self.wrench)):
                    raise ValueError('估计力包含非有限数')
                self.last_estimate = now
            except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
                self.observer.reset()
                self.wrench = None
                self.get_logger().warning(f'跳过反馈并重置观测器：{exc}', throttle_duration_sec=2.)

        def print_wrench(self):
            if self.wrench is None or self.last_estimate is None:
                self.get_logger().info('等待有效模型/反馈，或正在初始化观测器。', throttle_duration_sec=2.)
                return
            if time.monotonic()-self.last_estimate > self.timeout:
                self.get_logger().warning('反馈或估计未更新，不打印旧估计值。', throttle_duration_sec=2.)
                return
            fx, fy, fz, mx, my, mz = self.wrench
            print(f'[R6] Fx={fx:+9.3f} Fy={fy:+9.3f} Fz={fz:+9.3f} N | '
                  f'Mx={mx:+9.4f} My={my:+9.4f} Mz={mz:+9.4f} N*m', flush=True)

    rclpy.init(args=args)
    node = None
    try:
        node = ObserverNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
