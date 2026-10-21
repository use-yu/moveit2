"""R6 力导纳 -> 雅可比关节速度 -> 关节角积分。运行：python3 insert.py。"""

from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np

RIGHT_JOINTS = [f"r_arm_joint{i}" for i in range(1, 7)]


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


def joint_velocity(jacobian, velocity_r6, rotation, damping):
    """R6 线速度转关节速度，目标角速度为零，阻尼抑制奇异放大。"""
    twist = np.concatenate((rotation @ velocity_r6, np.zeros(3)))
    return jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + damping**2*np.eye(6), twist)


class AdmittanceController:
    """M*a+B*v+K*x=e+ki*integral，状态分量沿 R6 局部轴定义。"""

    def __init__(self, mass=20., damping=120., stiffness=0., ki=0.,
                 max_offset=0.3, max_velocity=0.1):
        self.mass, self.damping, self.stiffness, self.ki = mass, damping, stiffness, ki
        self.max_offset, self.max_velocity = max_offset, max_velocity
        self.x = np.zeros(3)
        self.velocity = np.zeros(3)
        self.integral = np.zeros(3)

    def step(self, fd, fe, dt):
        self.integral = np.clip(self.integral+(fe-fd)*dt, -1000., 1000.)
        acceleration = (fe-fd+self.ki*self.integral
                        - self.damping*self.velocity-self.stiffness*self.x)/self.mass
        self.velocity = np.clip(self.velocity+acceleration*dt,
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

            self.dt = positive('dt', 0.03)
            self.timeout = positive('feedback_timeout', 0.2)
            self.force_filter_cutoff_hz = positive('force_filter_cutoff_hz', 5.)
            self.jacobian_damping = positive('jacobian_damping', 0.01)
            self.max_joint_velocity = positive('max_joint_velocity', 0.05)
            self.max_tracking_error = positive('max_tracking_error', 0.5)
            mass = positive('mass', 20.)
            damping = positive('damping', 1000.)
            stiffness = float(parameter('stiffness', 0.))
            ki = float(parameter('ki', 0.0))
            if not np.isfinite(stiffness) or stiffness < 0 or not np.isfinite(ki) or ki < 0:
                raise ValueError('stiffness 和 ki 必须为非负有限数')
            self.controller = AdmittanceController(
                mass, damping, stiffness, ki,
                positive('max_offset', 0.3), positive('max_velocity', 0.1))
            self.model = None
            self.q = self.force = self.fd = self.q_command = None
            self.q_time = self.force_time = 0.
            self.last_update = None
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
                f'导纳 M={mass:g}, B={damping:g}, K={stiffness:g}, ki={ki:g}。'
                '使用雅可比关节速度积分，不调用 FK/IK 服务。')

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
            if force.size < 3 or not np.all(np.isfinite(force[:3])):
                self.stop('力反馈 XYZ 无效')
                return
            now = time.monotonic()
            if self.force is None:
                self.force = force[:3].copy()
            else:
                alpha = -np.expm1(-2*np.pi*self.force_filter_cutoff_hz*max(0., now-self.force_time))
                self.force += alpha*(force[:3]-self.force)
            self.force_time = now

        def tick(self):
            if self.fault or self.model is None or self.q is None or self.force is None:
                return
            now = time.monotonic()
            if now-self.q_time > self.timeout or now-self.force_time > self.timeout:
                if self.q_command is not None:
                    self.stop('关节或力反馈超时')
                return
            if self.q_command is None:
                self.fd = self.force.copy()
                self.q_command = self.q.copy()
                self.last_update = now
                self.get_logger().info(f'初始力 desired_force={self.fd.tolist()} N，开始速度导纳。')
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
            velocity_r6 = self.controller.step(self.fd, self.force, dt)
            try:
                dq = joint_velocity(jacobian, velocity_r6, transform[:3, :3], self.jacobian_damping)
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

    rclpy.init(args=args)
    node = None
    try:
        node = RightArmAdmittance()
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
