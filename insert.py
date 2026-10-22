"""沿初始 R6 Z 轴直线运动 5cm，叠加 X/Y/Rz 导纳。运行：python3 insert.py。"""

from pathlib import Path
from datetime import datetime, timezone
import os
import time
import xml.etree.ElementTree as ET
import numpy as np

RIGHT_JOINTS = [f"r_arm_joint{i}" for i in range(1, 7)]

# 每次启动先直线到达这些关节角对应的 R6 位姿，单位 rad。
START_JOINTS = [
 2.4064290523529053,
 -0.3589716851711273,
 -1.2858946323394775,
 0.056008487939834595,
 1.5611138343811035,
 -1.213338851928711,
]
PREPARE_DEFAULTS = {
    'linear_speed': 0.01,       # m/s，准备阶段最大期望线速度
    'angular_speed': 0.1,       # rad/s
    'position_tolerance': 0.001,
    'orientation_tolerance': 0.01,
    'joint_tolerance': 0.02,    # 也确认到达指定的关节构型
    'settle_time': 0.3,
    'extra_timeout': 30.0,      # 超过期望准备时长后的额外跟随时间
}

# 期望轨迹参数：距离正值沿初始 R6 +Z，负值沿 -Z。
TRAJECTORY_DEFAULTS = {
    'distance': 0.05,             # m
    'duration':7.0,             # s；五次曲线，起止速度/加速度为零，结束后退出程序
    'position_gain': 2.0,         # 位姿参考跟踪增益，1/s
    'orientation_gain': 2.0,
    'max_linear_speed': 0.03,     # 合成后的末端速度范数上限，m/s
    'max_angular_speed': 0.2,     # 合成后的末端角速度范数上限，rad/s
}

# 导纳调参区：修改这里的默认值，重启节点生效；ROS 启动参数可覆盖。
ADMITTANCE_DEFAULTS = {
    'xyz': {
        #                   X       Y       Z（当前任务只使用 X/Y，Z 参数保留）
        'M':              [20.0,   20.0,   20.0],
        'B':              [5000.0, 5000.0, 5000.0],
        'K':              [0.0,    0.0,    0.0],
        'ki': 0.0,
        'max_offset': 0.3,       # 每轴累计位移限幅，m
        'max_velocity': 0.1,     # 每轴线速度限幅，m/s
    },
    'rz': {
        'M': 1.0,               # 虚拟转动惯量，kg*m^2
        'B': 10.0,              # 旋转阻尼，N*m*s/rad
        'K': 0.0,               # 旋转刚度，N*m/rad
        'ki': 0.0,
        'max_angle_deg': 10.0,   # 累计转角限幅，度；ROS max_angle_rz 仍用 rad
        'max_velocity': 0.07,    # 绕 Z 角速度限幅，rad/s
    },
    'joint': {
        'max_velocity': 0.07,    # 每关节速度限幅，rad/s；会限制平移和旋转
        'max_tracking_error': 0.5,  # 关节目标与反馈的允许偏差，rad
    },
}


def save_force_plot(samples, directory, insertion_start=None, show=True):
    """控制结束后保存原始/低通 Fx,Fy,Fz,Mz 的 CSV 和曲线图。"""
    if not samples:
        print('没有运动期间的有效力反馈，未生成力曲线。', flush=True)
        return None
    import matplotlib
    if not show or not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        matplotlib.use('Agg')
        show = False
    import matplotlib.pyplot as plt

    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%fUTC')
    base = directory / f'insert_forces_{stamp}'
    data = np.asarray(samples, dtype=float)
    csv_path = base.with_suffix('.csv')
    np.savetxt(csv_path, data, delimiter=',', comments='', header=(
        'time_s,Fx_raw_N,Fy_raw_N,Fz_raw_N,Mz_raw_Nm,'
        'Fx_filtered_N,Fy_filtered_N,Fz_filtered_N,Mz_filtered_Nm,phase_0_prepare_1_insert'))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for index, (axis, name) in enumerate(zip(axes.flat, ('Fx', 'Fy', 'Fz', 'Mz'))):
        axis.plot(data[:, 0], data[:, index+1], color='0.65', alpha=0.65,
                  linewidth=0.7, label='Raw')
        axis.plot(data[:, 0], data[:, index+5], color='tab:blue',
                  linewidth=1.1, label='Low-pass')
        if insertion_start is not None:
            axis.axvspan(0., insertion_start, color='tab:orange', alpha=0.08)
            axis.axvline(insertion_start, color='tab:orange', linestyle='--',
                         linewidth=1., label='Insertion start')
        axis.set_title(f'{name} in R6')
        axis.set_ylabel('Torque (N m)' if name == 'Mz' else 'Force (N)')
        axis.set_xlabel('Time since preparation start (s)')
        axis.grid(True, alpha=0.3)
        axis.legend(loc='best')
    fig.suptitle('R6 forces and Z torque during preparation / insertion')
    fig.tight_layout()
    png_path = base.with_suffix('.png')
    fig.savefig(png_path, dpi=160)
    print(f'力曲线已保存：{png_path}\n原始与滤波数据已保存：{csv_path}', flush=True)
    try:
        if show:
            plt.show()
    finally:
        plt.close(fig)
    return png_path, csv_path


def z_reference(elapsed, distance, duration):
    """五次时间标度：返回沿初始 R6 Z 轴的位移及速度。"""
    u = float(np.clip(elapsed / duration, 0.0, 1.0))
    position = distance * (10*u**3 - 15*u**4 + 6*u**5)
    velocity = distance / duration * (30*u**2 - 60*u**3 + 30*u**4)
    return position, velocity


def limit_norm(value, maximum):
    return value / max(1.0, float(np.linalg.norm(value)) / maximum)


def rotation_vector(rotation):
    """SO(3) 对数，返回父坐标系下的最短旋转向量。"""
    angle = float(np.arccos(np.clip((np.trace(rotation)-1)/2, -1., 1.)))
    skew = np.array([rotation[2, 1]-rotation[1, 2],
                     rotation[0, 2]-rotation[2, 0], rotation[1, 0]-rotation[0, 1]])
    if angle < 1e-7:
        return 0.5*skew
    if np.pi-angle < 1e-5:
        _, vectors = np.linalg.eigh(0.5*(rotation+rotation.T))
        axis = vectors[:, -1]
        if np.dot(axis, skew) < 0:
            axis = -axis
        return axis*angle
    return skew*angle/(2*np.sin(angle))


class CartesianStartReference:
    """从当前 R6 到目标关节构型对应位姿，位置直线、姿态最短旋转插值。"""

    def __init__(self, start, target, duration):
        self.start, self.target, self.duration = start.copy(), target.copy(), duration
        self.displacement = target[:3, 3]-start[:3, 3]
        self.rotvec = rotation_vector(target[:3, :3] @ start[:3, :3].T)

    def step(self, elapsed):
        phase, phase_velocity = z_reference(elapsed, 1., self.duration)
        angle = np.linalg.norm(self.rotvec)
        delta_rotation = (np.eye(3) if angle < 1e-12 else
                          axis_rotation(self.rotvec/angle, angle*phase))
        return (self.start[:3, 3]+self.displacement*phase,
                delta_rotation @ self.start[:3, :3],
                self.displacement*phase_velocity, self.rotvec*phase_velocity)


class InsertionReference:
    """直线期望轨迹 + 沿当前 R6 X/Y 的导纳偏移 + 绕初始 Z 的转角。"""

    def __init__(self, initial_transform, distance, duration):
        self.initial = initial_transform.copy()
        self.distance, self.duration = distance, duration
        self.xy_offset_base = np.zeros(3)

    def step(self, elapsed, dt, rotation, velocity_xy, yaw, omega_z):
        z, vz = z_reference(elapsed, self.distance, self.duration)
        z_axis = self.initial[:3, 2]
        # 将当前局部横向增量累计为期望轨迹上的柔顺偏移。
        velocity_xy_base = rotation @ np.array([velocity_xy[0], velocity_xy[1], 0.0])
        # 数值姿态误差不能让横向导纳改变期望的轴向插入深度。
        velocity_xy_base -= z_axis * np.dot(z_axis, velocity_xy_base)
        self.xy_offset_base += velocity_xy_base * dt
        position = self.initial[:3, 3] + z_axis*z + self.xy_offset_base
        orientation = self.initial[:3, :3] @ axis_rotation([0, 0, 1], yaw)
        velocity = z_axis*vz + velocity_xy_base
        angular_velocity = z_axis*omega_z
        return position, orientation, velocity, angular_velocity


def axis_rotation(axis, angle):
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle)*skew + (1-np.cos(angle))*(skew @ skew)


class RightArmKinematics:
    """读取展开后的 URDF，计算 r_base_link -> R6 几何雅可比。"""

    def __init__(self, description):
        root = ET.fromstring(description)
        if any('xacro' in e.tag for e in root.iter()):
            raise ValueError('需要展开后的 URDF，请先用 xacro 展开')
        parents = {j.find('child').get('link'): j for j in root.findall('joint')}
        chain, visited = [], set()
        child = 'R6'
        while child != 'r_base_link':
            if child in visited or child not in parents:
                raise ValueError('URDF 缺少有效的 r_base_link -> R6 链')
            visited.add(child)
            joint = parents[child]
            chain.append(joint)
            child = joint.find('parent').get('link')
        self.chain = []
        self.lower = np.full(6, -np.inf)
        self.upper = np.full(6, np.inf)
        self.velocity_limits = np.full(6, np.inf)
        names = []
        for joint in reversed(chain):
            kind = joint.get('type')
            if kind not in ('fixed', 'revolute', 'continuous'):
                raise ValueError(f'不支持的关节类型：{kind}')
            origin, axis_element = joint.find('origin'), joint.find('axis')
            xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
            rpy = np.fromstring(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0', sep=' ')
            axis = np.fromstring(axis_element.get('xyz', '1 0 0') if axis_element is not None else '1 0 0', sep=' ')
            if any(v.shape != (3,) or not np.all(np.isfinite(v)) for v in (xyz, rpy, axis)):
                raise ValueError('URDF 中存在无效关节变换')
            if np.linalg.norm(axis) < 1e-12:
                raise ValueError('URDF 中存在零关节轴')
            axis /= np.linalg.norm(axis)
            transform = np.eye(4)
            transform[:3, :3] = (axis_rotation([0, 0, 1], rpy[2])
                                  @ axis_rotation([0, 1, 0], rpy[1])
                                  @ axis_rotation([1, 0, 0], rpy[0]))
            transform[:3, 3] = xyz
            index = None
            if kind != 'fixed':
                name = joint.get('name')
                if name not in RIGHT_JOINTS:
                    raise ValueError(f'R6 链中包含额外活动关节：{name}')
                index = RIGHT_JOINTS.index(name)
                names.append(name)
                limit = joint.find('limit')
                if kind == 'revolute':
                    if limit is None:
                        raise ValueError(f'关节 {name} 缺少位置限位')
                    self.lower[index] = float(limit.get('lower'))
                    self.upper[index] = float(limit.get('upper'))
                    if not self.lower[index] < self.upper[index]:
                        raise ValueError(f'关节 {name} 的位置限位无效')
                # 本项目 URDF 的 velocity=0 是占位值，只使用正速度限值。
                if limit is not None and float(limit.get('velocity', '0')) > 0:
                    self.velocity_limits[index] = float(limit.get('velocity'))
            self.chain.append((transform, axis, index))
        if names != RIGHT_JOINTS:
            raise ValueError('URDF 右臂关节顺序必须为 r_arm_joint1 到 r_arm_joint6')

    def evaluate(self, q):
        """返回 T_base_R6 和基座系表达的 [线速度; 角速度] 雅可比。"""
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError('关节位置必须为六维有限数')
        transform = np.eye(4)
        origins, axes = {}, {}
        for origin, axis, index in self.chain:
            transform = transform @ origin
            if index is not None:
                origins[index] = transform[:3, 3].copy()
                axes[index] = transform[:3, :3] @ axis
                rotation = np.eye(4)
                rotation[:3, :3] = axis_rotation(axis, q[index])
                transform = transform @ rotation
        jacobian = np.zeros((6, 6))
        for index in range(6):
            jacobian[:3, index] = np.cross(axes[index], transform[:3, 3]-origins[index])
            jacobian[3:, index] = axes[index]
        return transform, jacobian


def joint_velocity(jacobian, velocity_r6, rotation, damping, angular_velocity_r6=None):
    """R6 线速度、角速度转关节速度，阻尼抑制奇异放大。"""
    if angular_velocity_r6 is None:
        angular_velocity_r6 = np.zeros(3)
    twist = np.concatenate((rotation @ velocity_r6, rotation @ angular_velocity_r6))
    return jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + damping**2*np.eye(6), twist)


class AdmittanceController:
    """M*a+B*v+K*x=e+ki*integral，可用于平移或旋转局部轴。"""

    def __init__(self, mass=20., damping=120., stiffness=0., ki=0.,
                 max_offset=0.3, max_velocity=0.1, dimensions=3):
        def xyz(value, name, allow_zero=False):
            value = np.asarray(value, dtype=float)
            if value.ndim == 0:
                value = np.full(dimensions, float(value))
            if (value.shape != (dimensions,) or not np.all(np.isfinite(value))
                    or np.any(value < 0 if allow_zero else value <= 0)):
                raise ValueError(f'{name} 必须为 {dimensions} 个有效的' + ('非负数' if allow_zero else '正数'))
            return value.copy()

        self.mass = xyz(mass, 'mass')
        self.damping = xyz(damping, 'damping')
        self.stiffness = xyz(stiffness, 'stiffness', allow_zero=True)
        self.ki = ki
        self.max_offset, self.max_velocity = max_offset, max_velocity
        self.x = np.zeros(dimensions)
        self.velocity = np.zeros(dimensions)
        self.integral = np.zeros(dimensions)

    def step(self, fd, fe, dt):
        self.integral = np.clip(self.integral+(fe-fd)*dt, -1000., 1000.)
        drive = fe-fd+self.ki*self.integral-self.stiffness*self.x
        # 周期内保持 drive 不变，精确离散 M*dv/dt+B*v=drive。
        # 当前 K=0、ki=0 时，这是恒定力输入下速度的精确解。
        # expm1 避免 dt 很小时计算 1-exp(-B*dt/M) 的精度损失。
        alpha = -np.expm1(-self.damping*dt/self.mass)
        velocity = self.velocity+alpha*(drive/self.damping-self.velocity)
        self.velocity = np.clip(velocity,
                                -self.max_velocity, self.max_velocity)
        candidate = self.x+self.velocity*dt
        next_x = np.clip(candidate, -self.max_offset, self.max_offset)
        # 边界处最后一步只输出剩余位移所允许的速度。
        actual_velocity = (next_x-self.x)/dt
        limited = candidate != next_x
        self.x = next_x
        self.velocity[limited] = 0.
        self.integral[limited] = 0.
        return actual_velocity


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float32MultiArray, String

    class RightArmAdmittance(Node):
        def __init__(self):
            super().__init__('right_arm_admittance')

            def parameter(name, default):
                return self.declare_parameter(name, default).value

            def positive(name, default):
                value = float(parameter(name, default))
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f'{name} 必须为大于零的有限数')
                return value

            def nonnegative(name, default):
                value = float(parameter(name, default))
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f'{name} 必须为非负有限数')
                return value

            self.dt = positive('dt', 0.03)
            self.timeout = positive('feedback_timeout', 0.2)
            self.force_filter_cutoff_hz = positive('force_filter_cutoff_hz', 5.)
            self.jacobian_damping = positive('jacobian_damping', 0.01)
            xyz_defaults = ADMITTANCE_DEFAULTS['xyz']
            rz_defaults = ADMITTANCE_DEFAULTS['rz']
            joint_defaults = ADMITTANCE_DEFAULTS['joint']
            self.max_joint_velocity = positive('max_joint_velocity', joint_defaults['max_velocity'])
            self.max_tracking_error = positive('max_tracking_error', joint_defaults['max_tracking_error'])
            self.start_joints = np.asarray(parameter('start_joints', START_JOINTS), dtype=float)
            if self.start_joints.shape != (6,) or not np.all(np.isfinite(self.start_joints)):
                raise ValueError('start_joints 必须为六个有限的关节角，单位 rad')
            self.prepare_options = {
                name: positive('prepare_'+name, value) for name, value in PREPARE_DEFAULTS.items()
            }
            self.trajectory_distance = float(parameter('trajectory_distance', TRAJECTORY_DEFAULTS['distance']))
            if not np.isfinite(self.trajectory_distance):
                raise ValueError('trajectory_distance 必须为有限数')
            self.trajectory_duration = positive('trajectory_duration', TRAJECTORY_DEFAULTS['duration'])
            self.position_gain = positive('position_gain', TRAJECTORY_DEFAULTS['position_gain'])
            self.orientation_gain = positive('orientation_gain', TRAJECTORY_DEFAULTS['orientation_gain'])
            self.max_task_linear_speed = positive('max_task_linear_speed', TRAJECTORY_DEFAULTS['max_linear_speed'])
            self.max_task_angular_speed = positive('max_task_angular_speed', TRAJECTORY_DEFAULTS['max_angular_speed'])
            # R6 局部 X、Y、Z 三轴分别设置 M、B、K。
            mass = np.array([
                positive('mass_x', xyz_defaults['M'][0]),
                positive('mass_y', xyz_defaults['M'][1]),
                positive('mass_z', xyz_defaults['M'][2]),
            ])
            damping = np.array([
                positive('damping_x', xyz_defaults['B'][0]),
                positive('damping_y', xyz_defaults['B'][1]),
                positive('damping_z', xyz_defaults['B'][2]),
            ])
            stiffness = np.array([
                nonnegative('stiffness_x', xyz_defaults['K'][0]),
                nonnegative('stiffness_y', xyz_defaults['K'][1]),
                nonnegative('stiffness_z', xyz_defaults['K'][2]),
            ])
            ki = nonnegative('ki', xyz_defaults['ki'])
            self.controller = AdmittanceController(
                mass[:2], damping[:2], stiffness[:2], ki,
                positive('max_offset', xyz_defaults['max_offset']),
                positive('max_velocity', xyz_defaults['max_velocity']), dimensions=2)
            # R6 绕 Z 轴：Mz(N*m) -> 角速度(rad/s)，M 为虚拟转动惯量。
            mass_rz = positive('mass_rz', rz_defaults['M'])
            damping_rz = positive('damping_rz', rz_defaults['B'])
            stiffness_rz = nonnegative('stiffness_rz', rz_defaults['K'])
            self.rotation_controller = AdmittanceController(
                mass_rz, damping_rz, stiffness_rz, nonnegative('ki_rz', rz_defaults['ki']),
                positive('max_angle_rz', float(np.deg2rad(rz_defaults['max_angle_deg']))),
                positive('max_angular_velocity_rz', rz_defaults['max_velocity']), dimensions=1)
            self.model = None
            self.q = self.force = self.fd = self.q_command = None
            self.mz = self.initial_mz = None
            self.q_time = self.force_time = 0.
            self.last_update = None
            self.trajectory = None
            self.trajectory_start = None
            self.trajectory_finished = False
            self.force_history = []
            self.recording_start = None
            self.plot_directory = parameter('plot_directory', str(Path(__file__).resolve().parent/'plots'))
            self.show_force_plot = parameter('show_force_plot', True)
            self.preparing = True
            self.prepare_reference = None
            self.prepare_start = self.prepare_settled_since = None
            self.fault = False
            self.command_pub = self.create_publisher(JointState, '/g01/joint_commands', 1)
            self.create_subscription(Float32MultiArray, '/g01/right_arm/state', self.on_state, 1)
            self.create_subscription(Float32MultiArray, '/g01/right_arm/actual_TCP_force', self.on_force, 1)
            description_file = parameter('robot_description_file', '')
            if description_file:
                self.model = RightArmKinematics(Path(description_file).read_text())
            else:
                qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 reliability=ReliabilityPolicy.RELIABLE)
                self.create_subscription(String, '/robot_description', self.on_description, qos)
            self.create_timer(self.dt, self.tick)
            self.get_logger().info(
                f'等待 URDF、右臂关节和 R6 力；周期 {self.dt:g}s，力低通 {self.force_filter_cutoff_hz:g}Hz。'
                f'导纳 XY：M={mass[:2].tolist()}, B={damping[:2].tolist()}, K={stiffness[:2].tolist()}, ki={ki:g}；Z 力不参与导纳。'
                f'绕 R6 Z：M={mass_rz:g}, B={damping_rz:g}, K={stiffness_rz:g}；由第六项 Mz 驱动。'
                f'沿初始 R6 Z 轴期望位移 {self.trajectory_distance:g}m，时长 {self.trajectory_duration:g}s。'
                f'开始前先直线到达 start_joints={self.start_joints.tolist()} 对应位姿并停稳。'
                '使用雅可比跟踪叠加导纳后的位姿，轨迹结束后停止发布并退出程序。')

        def on_description(self, msg):
            if self.model is not None or self.fault:
                return
            try:
                self.model = RightArmKinematics(msg.data)
            except (ValueError, TypeError, ET.ParseError) as exc:
                self.stop(f'无法读取右臂模型：{exc}')

        def stop(self, reason):
            if not self.fault:
                self.get_logger().error(f'停止发布：{reason}；修复后重启节点。')
            self.fault = True

        def on_state(self, msg):
            q = np.asarray(msg.data, dtype=float)
            if q.shape != (6,) or not np.all(np.isfinite(q)):
                self.stop('关节反馈无效')
                return
            self.q = q.copy()
            self.q_time = time.monotonic()

        def on_force(self, msg):
            force = np.asarray(msg.data, dtype=float)
            if force.ndim != 1 or force.size < 6 or not np.all(np.isfinite(force[[0, 1, 2, 5]])):
                self.stop('六维反馈缺少有效的 XYZ 力或 Mz 力矩')
                return
            now = time.monotonic()
            if self.force is None:
                self.force = force[:3].copy()
                self.mz = float(force[5])
            else:
                alpha = -np.expm1(-2*np.pi*self.force_filter_cutoff_hz*max(0., now-self.force_time))
                self.force += alpha*(force[:3]-self.force)
                self.mz += alpha*(float(force[5])-self.mz)
            self.force_time = now
            if self.recording_start is not None:
                # 每帧同时记录原始值与滤波值；phase=0 准备，phase=1 插入。
                self.force_history.append((
                    now-self.recording_start, *force[[0, 1, 2, 5]].tolist(),
                    *self.force.tolist(), self.mz, 0 if self.preparing else 1))

        def tick(self):
            if self.trajectory_finished or self.fault or self.model is None or self.q is None:
                return
            now = time.monotonic()
            if now-self.q_time > self.timeout:
                if self.q_command is not None:
                    self.stop('关节反馈超时')
                return
            if self.preparing and self.prepare_reference is None:
                if np.any(self.start_joints < self.model.lower) or np.any(self.start_joints > self.model.upper):
                    self.stop('start_joints 超出 URDF 关节限位')
                    return
                start, _ = self.model.evaluate(self.q)
                goal, _ = self.model.evaluate(self.start_joints)
                # 五次曲线最大归一化速度为 1.875；时长兼顾位置、姿态和关节距离。
                duration = max(
                    3., 1.875*np.linalg.norm(goal[:3, 3]-start[:3, 3])/self.prepare_options['linear_speed'],
                    1.875*np.linalg.norm(rotation_vector(goal[:3, :3] @ start[:3, :3].T))/self.prepare_options['angular_speed'],
                    1.875*np.max(np.abs(self.start_joints-self.q))/self.max_joint_velocity)
                self.prepare_reference = CartesianStartReference(start, goal, duration)
                self.prepare_start = self.last_update = now
                self.recording_start = now
                self.q_command = self.q.copy()
                self.get_logger().info(f'开始右臂准备直线运动，期望时长 {duration:.2f}s；此阶段不叠加导纳。')
                return
            if not self.preparing:
                if self.force is None or self.mz is None or now-self.force_time > self.timeout:
                    if self.q_command is not None:
                        self.stop('力反馈超时')
                    return
            if self.q_command is None:
                self.fd = self.force.copy()
                self.initial_mz = self.mz
                self.q_command = self.q.copy()
                transform, _ = self.model.evaluate(self.q)
                self.trajectory = InsertionReference(
                    transform, self.trajectory_distance, self.trajectory_duration)
                self.trajectory_start = now
                self.last_update = now
                self.get_logger().info(
                    f'初始力 desired_force={self.fd.tolist()} N，初始 Mz={self.initial_mz:g} N*m，开始速度导纳。')
                return
            dt = now-self.last_update
            self.last_update = now
            if dt <= 0 or dt > min(self.timeout, 3*self.dt):
                self.stop('控制周期异常')
                return
            if np.any(np.abs(self.q_command-self.q) > self.max_tracking_error):
                self.stop('关节目标与反馈偏差超过 max_tracking_error')
                return
            transform, jacobian = self.model.evaluate(self.q)
            if self.preparing:
                elapsed = now-self.prepare_start
                if elapsed > self.prepare_reference.duration+self.prepare_options['extra_timeout']:
                    self.stop('准备直线运动未能到达指定构型，结束程序')
                    self.trajectory_finished = True
                    return
                position, orientation, feedforward_v, feedforward_w = self.prepare_reference.step(elapsed)
            else:
                velocity_xy = self.controller.step(self.fd[:2], self.force[:2], dt)
                omega_z = self.rotation_controller.step(
                    np.array([self.initial_mz]), np.array([self.mz]), dt)[0]
                elapsed = now - self.trajectory_start
                position, orientation, feedforward_v, feedforward_w = self.trajectory.step(
                    elapsed, dt, transform[:3, :3], velocity_xy,
                    self.rotation_controller.x[0], omega_z)
            # 对连续关节目标对应的位姿做参考跟踪，避免将实际跟随滞后反复积分。
            command_transform, _ = self.model.evaluate(self.q_command)
            orientation_error = 0.5 * sum(
                (np.cross(command_transform[:3, i], orientation[:, i]) for i in range(3)),
                np.zeros(3))
            linear_limit = self.max_task_linear_speed
            angular_limit = self.max_task_angular_speed
            if self.preparing:
                linear_limit = min(linear_limit, self.prepare_options['linear_speed'])
                angular_limit = min(angular_limit, self.prepare_options['angular_speed'])
            velocity_base = limit_norm(
                feedforward_v + self.position_gain*(position-command_transform[:3, 3]),
                linear_limit)
            angular_velocity_base = limit_norm(
                feedforward_w + self.orientation_gain*orientation_error,
                angular_limit)
            # joint_velocity 接收局部速度，先将合成的基座系速度转回当前 R6。
            velocity_r6 = transform[:3, :3].T @ velocity_base
            angular_velocity_r6 = transform[:3, :3].T @ angular_velocity_base
            try:
                dq = joint_velocity(jacobian, velocity_r6, transform[:3, :3],
                                    self.jacobian_damping, angular_velocity_r6)
            except np.linalg.LinAlgError:
                self.stop('雅可比求解失败')
                return
            limits = np.minimum(self.max_joint_velocity, self.model.velocity_limits)
            # 整体缩放，保留各关节速度比例。
            dq /= max(1., float(np.max(np.abs(dq)/limits)))
            target = self.q_command+dq*dt
            if not np.all(np.isfinite(target)):
                self.stop('关节目标无效')
                return
            if np.any(target < self.model.lower) or np.any(target > self.model.upper):
                self.stop('关节目标超出 URDF 位置限位')
                return
            self.q_command = target
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = list(RIGHT_JOINTS)
            msg.position = target.tolist()
            msg.velocity = dq.tolist()
            self.command_pub.publish(msg)
            if self.preparing:
                goal = self.prepare_reference.target
                arrived = (
                    elapsed >= self.prepare_reference.duration
                    and np.linalg.norm(transform[:3, 3]-goal[:3, 3]) <= self.prepare_options['position_tolerance']
                    and np.linalg.norm(rotation_vector(goal[:3, :3] @ transform[:3, :3].T)) <= self.prepare_options['orientation_tolerance']
                    and np.max(np.abs(self.q-self.start_joints)) <= self.prepare_options['joint_tolerance'])
                if not arrived:
                    self.prepare_settled_since = None
                elif self.prepare_settled_since is None:
                    self.prepare_settled_since = now
                elif now-self.prepare_settled_since >= self.prepare_options['settle_time']:
                    self.preparing = False
                    self.q_command = None
                    # 丢弃准备运动中的滤波力，下一帧重新采集初始力与 Mz。
                    self.force = self.mz = None
                    self.force_time = 0.
                    self.get_logger().info(
                        f'准备位姿已到达并停稳，等待新力反馈后开始 {self.trajectory_duration:g} 秒导纳轨迹。')
            elif elapsed >= self.trajectory_duration and not self.trajectory_finished:
                self.trajectory_finished = True
                self.get_logger().info(
                    f'{self.trajectory_distance*100:g}cm 期望轨迹时间结束，停止发布并退出程序。')

    rclpy.init(args=args)
    node = None
    try:
        node = RightArmAdmittance()
        while rclpy.ok() and not node.trajectory_finished and not node.fault:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        # 先关闭 ROS 控制，再绘图；正常完成、故障和 Ctrl+C 都保存已采集的数据。
        if node is not None:
            insertion_start = (None if node.trajectory_start is None or node.recording_start is None
                               else node.trajectory_start-node.recording_start)
            try:
                save_force_plot(node.force_history, node.plot_directory,
                                insertion_start, node.show_force_plot)
            except (ImportError, OSError, ValueError, RuntimeError) as exc:
                print(f'保存力曲线失败：{exc}', flush=True)


if __name__ == '__main__':
    main()
