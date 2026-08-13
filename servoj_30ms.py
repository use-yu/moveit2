# 两个臂连接
# 两个臂 EnableRobot()
# 两个臂先 servoj_2(..., 0) 复位
# sleep(3)
# rclpy.spin(state_node) 等待 /g01/joint_commands
# 控制和臂位置是角度
# tcp读取状态100hz，/g01/joint_commands 200hz；每30ms取最新完整6关节值发送一次ServoJ
# 只有完整左臂6关节
# → 只更新左臂最新目标
# → 30 ms定时器只向左臂发送 ServoJ

# 只有完整右臂6关节
# → 只更新右臂最新目标
# → 只向右臂发送 ServoJ

# 同时有左右臂完整关节
# → 分别更新左右臂最新目标
# → 左右臂各发送一次 ServoJ

# 没有手臂关节或关节不完整
# → 不发送 ServoJ
# /g01/servoj_control 可按臂 stop/resume；停止时关闭指令入口并清轨迹缓存，
# 处理结果发布到 /g01/servoj_control_state。
# /g01/left/ft_sensor_commands 和 /g01/right/ft_sensor_commands 使用
# dobot_msgs_v4/srv/EnableFTSensor，通过 status=1/0 开启/关闭对应力传感器。
import json
import socket
from time import sleep
import time
import threading
import logging
from logging.handlers import RotatingFileHandler
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, String
from dobot_msgs_v4.srv import EnableFTSensor, SetToolPower

MyType = np.dtype([('len', np.int64,), ('digital_input_bits', np.uint64,), ('digital_output_bits',
                                                                            np.uint64,), ('robot_mode', np.uint64,),
                   ('time_stamp', np.uint64,), ('time_stamp_reserve_bit', np.uint64,),
                   ('test_value', np.uint64,), ('test_value_keep_bit', np.float64,), ('speed_scaling', np.float64,),
                   ('linear_momentum_norm', np.float64,),
                   ('v_main', np.float64,), ('v_robot', np.float64,), ('i_robot', np.float64,),
                   ('i_robot_keep_bit1', np.float64,), ('i_robot_keep_bit2', np.float64,),
                   ('tool_accelerometer_values', np.float64, (3,)),
                   ('elbow_position', np.float64, (3,)),
                   ('elbow_velocity', np.float64, (3,)),
                   ('q_target', np.float64, (6,)),
                   ('qd_target', np.float64, (6,)),
                   ('qdd_target', np.float64, (6,)),
                   ('i_target', np.float64, (6,)),
                   ('m_target', np.float64, (6,)),
                   ('q_actual', np.float64, (6,)),
                   ('qd_actual', np.float64, (6,)),
                   ('i_actual', np.float64, (6,)),
                   ('actual_TCP_force', np.float64, (6,)),
                   ('tool_vector_actual', np.float64, (6,)),
                   ('TCP_speed_actual', np.float64, (6,)),
                   ('TCP_force', np.float64, (6,)),
                   ('Tool_vector_target', np.float64, (6,)),
                   ('TCP_speed_target', np.float64, (6,)),
                   ('motor_temperatures', np.float64, (6,)),
                   ('joint_modes', np.float64, (6,)),
                   ('v_actual', np.float64, (6,)),
                   ('hand_type', np.byte, (4,)),
                   ('user', np.byte,),
                   ('tool', np.byte,),
                   ('run_queued_cmd', np.byte,),
                   ('pause_cmd_flag', np.byte,),
                   ('velocity_ratio', np.int8,),
                   ('acceleration_ratio', np.int8,),
                   ('jerk_ratio', np.int8,),
                   ('xyz_velocity_ratio', np.int8,),
                   ('r_velocity_ratio', np.int8,),
                   ('xyz_acceleration_ratio', np.int8,),
                   ('r_acceleration_ratio', np.int8,),
                   ('xyz_jerk_ratio', np.int8,),
                   ('r_jerk_ratio', np.int8,),
                   ('brake_status', np.int8,),
                   ('enable_status', np.int8,),
                   ('drag_status', np.int8,),
                   ('running_status', np.int8,),
                   ('error_status', np.int8,),
                   ('jog_status', np.int8,),
                   ('robot_type', np.int8,),
                   ('drag_button_signal', np.int8,),
                   ('enable_button_signal', np.int8,),
                   ('record_button_signal', np.int8,),
                   ('reappear_button_signal', np.int8,),
                   ('jaw_button_signal', np.int8,),
                   ('six_force_online', np.int8,),
                   ('reserve2', np.int8, (82,)),
                   ('m_actual', np.float64, (6,)),
                   ('load', np.float64,),
                   ('center_x', np.float64,),
                   ('center_y', np.float64,),
                   ('center_z', np.float64,),
                   ('user1', np.float64, (6,)),
                   ('Tool1', np.float64, (6,)),
                   ('trace_index', np.float64,),
                   ('six_force_value', np.float64, (6,)),
                   ('target_quaternion', np.float64, (4,)),
                   ('actual_quaternion', np.float64, (4,)),
                   ('reserve3', np.int8, (24,))
                   ])

JOINT_ORDER = [
    "base_joint1",
    "base_joint2",
    "body_joint1",
    "body_joint2",
    "l_arm_joint1",
    "l_arm_joint2",
    "l_arm_joint3",
    "l_arm_joint4",
    "l_arm_joint5",
    "l_arm_joint6",
    "r_arm_joint1",
    "r_arm_joint2",
    "r_arm_joint3",
    "r_arm_joint4",
    "r_arm_joint5",
    "r_arm_joint6",
]
LEFT_JOINTS = JOINT_ORDER[4:10]
RIGHT_JOINTS = JOINT_ORDER[10:16]
ARM_COMMAND_SIZE = 6
SERVOJ_SEND_PERIOD_SEC = 0.03
COMMAND_TOPIC = "/g01/joint_commands"
SERVOJ_CONTROL_TOPIC = "/g01/servoj_control"
SERVOJ_CONTROL_STATE_TOPIC = "/g01/servoj_control_state"
SERVOJ_RESUME_GUARD_SEC = 0.1
LEFT_TOOL_COMMAND_SERVICE = "/g01/left/tool_commands"
RIGHT_TOOL_COMMAND_SERVICE = "/g01/right/tool_commands"
LEFT_FT_SENSOR_COMMAND_SERVICE = "/g01/left/ft_sensor_commands"
RIGHT_FT_SENSOR_COMMAND_SERVICE = "/g01/right/ft_sensor_commands"
# DEFAULT_PAYLOAD = (1.1, 0.0, 0.0, 45.0)
# TOOL_POWER_ON_PAYLOAD = (4.85, 0.0, 0.0, 87.0)
DEFAULT_PAYLOAD = (1.1, 0.0, 0.0, 82.0)
TOOL_POWER_ON_PAYLOAD = (4.85, 0.0, 0.0, 124.0)
FEEDBACK_FRAME_SIZE = MyType.itemsize
FEEDBACK_TEST_VALUE = 0x0123456789ABCDEF
FEEDBACK_TEST_VALUE_OFFSET = MyType.fields["test_value"][1]


class fankuis():
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.socket_feedback = 0
        self._recv_buffer = bytearray()
        self._last_error_log_time = 0.0

        if self.port == 30005 or self.port == 30004:
            self.socket_feedback = socket.socket()
            self.socket_feedback.settimeout(1.0)
            self.socket_feedback.connect((self.ip, self.port))
        else:
            raise ValueError("反馈端口只支持 30004 或 30005")

    def _log_feedback_error(self, error):
        # 网络短暂抖动时避免 100 Hz 循环刷屏，但不能再静默吞掉解析错误。
        now = time.monotonic()
        if now - self._last_error_log_time >= 2.0:
            print(f"{self.ip}:{self.port} 反馈接收解析失败：{error}")
            self._last_error_log_time = now

    def _pop_valid_frame(self):
        """从 TCP 字节流中取一帧，并在错位时利用 test_value 重新同步。"""
        while len(self._recv_buffer) >= FEEDBACK_FRAME_SIZE:
            test_value = int.from_bytes(
                self._recv_buffer[
                    FEEDBACK_TEST_VALUE_OFFSET:FEEDBACK_TEST_VALUE_OFFSET + 8
                ],
                byteorder="little",
                signed=False,
            )
            if test_value == FEEDBACK_TEST_VALUE:
                frame = bytes(self._recv_buffer[:FEEDBACK_FRAME_SIZE])
                del self._recv_buffer[:FEEDBACK_FRAME_SIZE]
                return frame

            # recv() 可能从半帧开始；逐字节寻找下一处合法帧头。
            del self._recv_buffer[0]

        return None

    def _read_latest_frame(self):
        while True:
            frame = self._pop_valid_frame()
            if frame is not None:
                # 一次 recv() 可能带回多帧。丢弃积压的旧帧，只发布最新完整帧。
                while True:
                    newer_frame = self._pop_valid_frame()
                    if newer_frame is None:
                        return frame
                    frame = newer_frame

            chunk = self.socket_feedback.recv(FEEDBACK_FRAME_SIZE * 8)
            if not chunk:
                raise ConnectionError("机器人已关闭反馈连接")
            self._recv_buffer.extend(chunk)

    def feed(self):
        try:
            frame = self._read_latest_frame()
            feedback = np.frombuffer(frame, dtype=MyType, count=1)[0]
            tool_v = feedback['tool_vector_actual'].copy()
            tool_j = feedback['q_actual'].copy()
            tool_i = feedback['i_actual'].copy()
            tcp_force = feedback['TCP_force'].copy()
            ft_sensor = feedback['six_force_value'].copy()
            ft_sensor_online = int(feedback['six_force_online'])
            return [tool_v, tool_j, tool_i, tcp_force, ft_sensor, ft_sensor_online]
        except (OSError, ConnectionError, ValueError) as error:
            self._log_feedback_error(error)
            return ["NG"]

class ArmRosBridge(Node):
    def __init__(self, clients):
        super().__init__("dobot_arm_ros_bridge")
        self.arm_clients = clients
        self.left_pub = self.create_publisher(Float32MultiArray, "/g01/left_arm/state", 10)
        self.right_pub = self.create_publisher(Float32MultiArray, "/g01/right_arm/state", 10)
        self.left_i_pub = self.create_publisher(Float32MultiArray, "/g01/left_arm/i_actual", 10)
        self.right_i_pub = self.create_publisher(Float32MultiArray, "/g01/right_arm/i_actual", 10)
        self.left_tcp_force_pub = self.create_publisher(Float32MultiArray, "/g01/left_arm/TCP_force", 10)
        self.right_tcp_force_pub = self.create_publisher(Float32MultiArray, "/g01/right_arm/TCP_force", 10)
        self.left_ft_sensor_pub = self.create_publisher(Float32MultiArray, "/g01/left/FTSensor", 10)
        self.right_ft_sensor_pub = self.create_publisher(Float32MultiArray, "/g01/right/FTSensor", 10)
        # 200Hz输入只保留最新一条，30ms定时器按固定频率下发最新完整目标。
        self._latest_arm_commands = {"left": None, "right": None}
        self._new_arm_commands = {"left": False, "right": False}
        self._arm_paused = {"left": False, "right": False}
        self._accept_commands_after = {"left": 0.0, "right": 0.0}
        self._arm_command_lock = threading.Lock()
        self.create_subscription(JointState, COMMAND_TOPIC, self._on_joint_commands, 1)
        self.create_subscription(
            String,
            SERVOJ_CONTROL_TOPIC,
            self._on_servoj_control,
            10,
        )
        self._servoj_control_state_pub = self.create_publisher(
            String,
            SERVOJ_CONTROL_STATE_TOPIC,
            10,
        )
        self.create_timer(SERVOJ_SEND_PERIOD_SEC, self._send_latest_joint_commands)
        self.create_service(SetToolPower, LEFT_TOOL_COMMAND_SERVICE, self._on_left_tool_command)
        self.create_service(SetToolPower, RIGHT_TOOL_COMMAND_SERVICE, self._on_right_tool_command)
        self.create_service(
            EnableFTSensor,
            LEFT_FT_SENSOR_COMMAND_SERVICE,
            self._on_left_ft_sensor_command,
        )
        self.create_service(
            EnableFTSensor,
            RIGHT_FT_SENSOR_COMMAND_SERVICE,
            self._on_right_ft_sensor_command,
        )

    def publish_arm_state(self, side, q):
        msg = Float32MultiArray()
        msg.data = [math.radians(float(num)) for num in q]
        if side == "left":
            self.left_pub.publish(msg)
        elif side == "right":
            self.right_pub.publish(msg)

    def publish_arm_i_actual(self, side, i_actual):
        msg = Float32MultiArray()
        msg.data = [float(num) for num in i_actual]
        if side == "left":
            self.left_i_pub.publish(msg)
        elif side == "right":
            self.right_i_pub.publish(msg)

    def publish_arm_tcp_force(self, side, tcp_force):
        msg = Float32MultiArray()
        msg.data = [float(num) for num in tcp_force]
        if side == "left":
            self.left_tcp_force_pub.publish(msg)
        elif side == "right":
            self.right_tcp_force_pub.publish(msg)

    def publish_ft_sensor(self, side, force):
        msg = Float32MultiArray()
        msg.data = [float(num) for num in force]
        if side == "left":
            self.left_ft_sensor_pub.publish(msg)
        elif side == "right":
            self.right_ft_sensor_pub.publish(msg)

    def _extract_arm_command(self, name_to_pos, joints):
        out = []
        for joint in joints:
            if joint not in name_to_pos:
                return None
            out.append(float(name_to_pos[joint]))
        return out if len(out) == ARM_COMMAND_SIZE else None

    def _on_joint_commands(self, msg):
        name_to_pos = {}
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                name_to_pos[str(name)] = float(msg.position[i])

        left_cmd = self._extract_arm_command(name_to_pos, LEFT_JOINTS)
        right_cmd = self._extract_arm_command(name_to_pos, RIGHT_JOINTS)

        # 高频回调只覆盖缓存，不直接发送；同一个30ms周期内的旧目标会被新目标替换。
        now = time.monotonic()
        with self._arm_command_lock:
            if (
                left_cmd is not None
                and not self._arm_paused["left"]
                and now >= self._accept_commands_after["left"]
            ):
                self._latest_arm_commands["left"] = left_cmd
                self._new_arm_commands["left"] = True
            if (
                right_cmd is not None
                and not self._arm_paused["right"]
                and now >= self._accept_commands_after["right"]
            ):
                self._latest_arm_commands["right"] = right_cmd
                self._new_arm_commands["right"] = True

    def _publish_servoj_control_state(
        self,
        side,
        state,
        request_id,
        success=True,
        message="",
    ):
        reply = String()
        reply.data = json.dumps(
            {
                "side": side,
                "state": state,
                "request_id": request_id,
                "success": bool(success),
                "message": str(message),
            },
            ensure_ascii=False,
        )
        self._servoj_control_state_pub.publish(reply)

    def _on_servoj_control(self, msg):
        """按臂停止/恢复 ServoJ，并用带 request_id 的状态话题确认。"""
        try:
            request = json.loads(msg.data)
            side = str(request["side"])
            command = str(request["command"])
            request_id = int(request["request_id"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self.get_logger().error(
                f"{SERVOJ_CONTROL_TOPIC} 命令格式错误：{msg.data!r}; {exc}"
            )
            return

        if side not in ("left", "right") or command not in ("stop", "resume"):
            self.get_logger().error(
                f"{SERVOJ_CONTROL_TOPIC} 命令非法：side={side!r}, "
                f"command={command!r}"
            )
            return

        if command == "stop":
            # 先在同一回调内关闸并清除缓存。之后到达的旧 MoveIt 轨迹点也会被丢弃。
            with self._arm_command_lock:
                self._arm_paused[side] = True
                self._latest_arm_commands[side] = None
                self._new_arm_commands[side] = False

            self.get_logger().warning(
                f"{side} ServoJ 已停止接收指令并清除轨迹缓存，"
                f"request_id={request_id}"
            )
            self._publish_servoj_control_state(side, "stopped", request_id)
            return

        # 恢复时仍清一次缓存，并短暂过滤可能已排队的旧轨迹消息；下一条新规划
        # 至少要经过服务规划和 action 握手，不会被这 100 ms 防陈旧窗口误丢弃。
        with self._arm_command_lock:
            self._arm_paused[side] = False
            self._latest_arm_commands[side] = None
            self._new_arm_commands[side] = False
            self._accept_commands_after[side] = (
                time.monotonic() + SERVOJ_RESUME_GUARD_SEC
            )
        self.get_logger().info(
            f"{side} ServoJ 已恢复，request_id={request_id}"
        )
        self._publish_servoj_control_state(side, "running", request_id)

    def _send_latest_joint_commands(self):
        pending_commands = []
        with self._arm_command_lock:
            for side, client_index in (("left", 0), ("right", 1)):
                if self._arm_paused[side]:
                    self._latest_arm_commands[side] = None
                    self._new_arm_commands[side] = False
                    continue
                command = self._latest_arm_commands[side]
                if not self._new_arm_commands[side] or command is None:
                    continue
                pending_commands.append((self.arm_clients[client_index], list(command)))
                self._new_arm_commands[side] = False

        # 每30ms最多向每侧发送一次；这段时间没有新目标则不重复发送旧目标。
        # 把数据写入内核发送缓冲区，没 recv(200)。
        for client, command in pending_commands:
            ServoJ(client, [math.degrees(value) for value in command])

    def _on_tool_command(self, request, response, side, client):
        status = int(request.status)
        if status not in (0, 1):
            response.res = -1
            self.get_logger().error(f"SetToolPower status 非法：{status}，只允许 0/1")
            return response

        response.res = send_set_tool_power(client, status)
        payload = TOOL_POWER_ON_PAYLOAD if status == 1 else DEFAULT_PAYLOAD
        payload_res = send_set_payload(client, *payload)
        if response.res == 0:
            response.res = payload_res
        self.get_logger().info(
            f"/g01/{side}/tool_commands: SetToolPower({status}), "
            f"SetPayload(load={payload[0]}, x={payload[1]}, y={payload[2]}, z={payload[3]}), "
            f"res={response.res}"
        )
        return response

    def _on_left_tool_command(self, request, response):
        return self._on_tool_command(request, response, "left", self.arm_clients[0])

    def _on_right_tool_command(self, request, response):
        return self._on_tool_command(request, response, "right", self.arm_clients[1])

    def _on_ft_sensor_command(self, request, response, side, client):
        """通过对应机械臂的 29999 端口开启或关闭六维力传感器。"""
        status = int(request.status)
        if status not in (0, 1):
            response.res = -1
            self.get_logger().error(
                f"EnableFTSensor status 非法：{status}，只允许 0/1"
            )
            return response

        try:
            response.res = send_enable_ft_sensor(client, status)
        except OSError as exc:
            response.res = -1
            self.get_logger().error(
                f"/g01/{side}/ft_sensor_commands: "
                f"EnableFTSensor({status}) 发送失败：{exc}"
            )
            return response

        self.get_logger().info(
            f"/g01/{side}/ft_sensor_commands: "
            f"EnableFTSensor({status}), res={response.res}"
        )
        return response

    def _on_left_ft_sensor_command(self, request, response):
        return self._on_ft_sensor_command(
            request,
            response,
            "left",
            self.arm_clients[0],
        )

    def _on_right_ft_sensor_command(self, request, response):
        return self._on_ft_sensor_command(
            request,
            response,
            "right",
            self.arm_clients[1],
        )


q_actual = {"left": [], "right": []}
i_actual = {"left": [], "right": []}
TCP_force = {"left": [], "right": []}
FT_sensor = {"left": [], "right": []}
def joint(feed_v, side, state_node):
    global q_actual, i_actual, TCP_force, FT_sensor
    last_ft_state = None
    while stop:
        actual = feed_v.feed()
        if actual == ["NG"]:
            time.sleep(0.01)
            continue

        q_actual[side] = [round(num, 6) for num in actual[1]]
        i_actual[side] = [round(num, 6) for num in actual[2]]
        TCP_force[side] = [float(num) for num in actual[3]]
        state_node.publish_arm_state(side, q_actual[side])
        state_node.publish_arm_i_actual(side, i_actual[side])
        state_node.publish_arm_tcp_force(side, TCP_force[side])

        ft_sensor_status = int(actual[5])
        force_values = [float(num) for num in actual[4]]
        force_valid = len(force_values) == 6 and all(
            math.isfinite(value) for value in force_values
        )
        ft_state = (ft_sensor_status, force_valid)

        if ft_state != last_ft_state:
            if ft_sensor_status == 1 and force_valid:
                state_node.get_logger().info(
                    f"{side} 力传感器已在线，恢复发布 /g01/{side}/FTSensor"
                )
            else:
                reason = (
                    f"six_force_online={ft_sensor_status}"
                    if ft_sensor_status != 1
                    else "六维力数据无效"
                )
                state_node.get_logger().warning(
                    f"{side} 力传感器不可用（{reason}），停止发布 /g01/{side}/FTSensor"
                )
            last_ft_state = ft_state

        # 传感器下电时控制器可能继续重复最后一帧；必须用在线状态拦截。
        if ft_sensor_status == 1 and force_valid:
            FT_sensor[side] = force_values
            state_node.publish_ft_sensor(side, FT_sensor[side])
        else:
            # 清空内部缓存，避免其他代码误把下电前的最后一帧当作实时数据。
            FT_sensor[side] = []

        # feed() 本身按控制器约 8 ms 的反馈周期阻塞，不再额外 sleep，避免旧帧积压。

def setup_logger(log_file='app.log'):
    # 创建日志记录器
    logger = logging.getLogger('my_app')
    logger.setLevel(logging.DEBUG)
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 创建旋转文件处理器
    # maxBytes=50MB (50 * 1024 * 1024), backupCount=5
    handler = RotatingFileHandler(
        log_file,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=5,  # 保留5个备份文件
        encoding='utf-8'
    )
    handler.setFormatter(formatter)
    # 添加处理器到日志记录器
    logger.addHandler(handler)
    return logger
def ServoP(client, pt_floats):
    tcp_command = f"ServoP({pt_floats[0]},{pt_floats[1]},{pt_floats[2]},{pt_floats[3]},{pt_floats[4]},{pt_floats[5]}, t=0.004, aheadtime=80, gain=500)"
    tcp_command = tcp_command.encode()
    client.send(tcp_command)

def ServoJ(client, joints):
    tcp_command = "servoj(%f,%f,%f,%f,%f,%f,t=0.03,aheadtime=26, gain=200)" % tuple(joints)
    tcp_command = tcp_command.encode()
    client.sendall(tcp_command)
    # buf = client.recv(200).decode()  # 接收反馈信息的长度
    # print(buf)

def send_set_tool_power(client, status):
    tcp_command = "SetToolPower(%d)" % int(status)
    tcp_command = tcp_command.encode()
    client.sendall(tcp_command)
    return 0

def send_enable_ft_sensor(client, status):
    tcp_command = "EnableFTSensor(%d)" % int(status)
    client.sendall(tcp_command.encode())
    return 0

DASHBOARD_COMMAND_TIMEOUT = 5.0
DASHBOARD_POWERON_TIMEOUT = 15.0
DASHBOARD_RECV_BUFFERS = {}

def send_set_payload(client, load, x, y, z, wait_response=False):
    tcp_command = "SetPayload(%s,%s,%s,%s)" % (
        float(load),
        float(x),
        float(y),
        float(z),
    )
    client.sendall(tcp_command.encode())
    if not wait_response:
        return 0
    response_text = recv_dashboard_response(client, expected_command=tcp_command)
    print(response_text)
    return parse_dashboard_result(response_text)

def servoj_2(client, J1_angle):
    tcp_command = "servoj(%f,0,0,0,0,0,t=2,aheadtime=50, gain=500)" % (J1_angle)
    tcp_command = tcp_command.encode()
    print(tcp_command)
    client.send(tcp_command)

def parse_dashboard_result(text):
    try:
        return int(text.split(",", 1)[0].strip())
    except (ValueError, IndexError):
        return -1

def parse_dashboard_payload(text):
    start = text.find("{")
    end = text.find("}", start + 1)
    if start < 0 or end < 0:
        return ""
    return text[start + 1:end].strip()

def dashboard_command_name(command):
    return command.split("(", 1)[0].strip()

def dashboard_response_command(response_text):
    response_text = response_text.strip().rstrip(";")
    payload_end = response_text.find("}")
    if payload_end >= 0:
        command_text = response_text[payload_end + 1:].strip()
        if command_text.startswith(","):
            command_text = command_text[1:].strip()
        return command_text
    parts = response_text.split(",", 2)
    if len(parts) < 3:
        return ""
    return parts[2].strip()

def dashboard_response_matches(response_text, command):
    expected_name = dashboard_command_name(command)
    response_command = dashboard_response_command(response_text)
    return response_command.startswith(f"{expected_name}(")

def dashboard_timeout_for(command):
    if dashboard_command_name(command) == "PowerOn":
        return DASHBOARD_POWERON_TIMEOUT
    return DASHBOARD_COMMAND_TIMEOUT

def recv_dashboard_response(client, timeout=DASHBOARD_COMMAND_TIMEOUT, expected_command=None, name=""):
    old_timeout = client.gettimeout()
    deadline = time.time() + timeout
    recv_buffer = DASHBOARD_RECV_BUFFERS.get(client, "")
    last_unmatched = None
    try:
        while True:
            while ";" not in recv_buffer:
                remaining = deadline - time.time()
                if remaining <= 0:
                    if last_unmatched is not None and expected_command:
                        raise TimeoutError(
                            f"等待 dashboard 返回超时，未收到 {expected_command} 的返回，最后收到：{last_unmatched}"
                        )
                    raise TimeoutError("等待 dashboard 返回超时")
                client.settimeout(remaining)
                data = client.recv(1024)
                if not data:
                    raise ConnectionError("dashboard 连接断开")
                recv_buffer += data.decode(errors="ignore")

            response_text, recv_buffer = recv_buffer.split(";", 1)
            DASHBOARD_RECV_BUFFERS[client] = recv_buffer
            response_text = response_text.strip() + ";"
            if expected_command is None or dashboard_response_matches(response_text, expected_command):
                return response_text

            last_unmatched = response_text
            prefix = f"{name} " if name else ""
            print(f"{prefix}跳过非当前命令返回：{response_text}")
    except socket.timeout as e:
        if last_unmatched is not None and expected_command:
            raise TimeoutError(
                f"等待 dashboard 返回超时，未收到 {expected_command} 的返回，最后收到：{last_unmatched}"
            ) from e
        raise TimeoutError("等待 dashboard 返回超时") from e
    finally:
        DASHBOARD_RECV_BUFFERS[client] = recv_buffer
        client.settimeout(old_timeout)

DASHBOARD_ERROR_HINTS = {
    -2: "当前仍是报警状态，需要 ClearError 或处理无法清除的报警",
    -3: "急停仍被按下，需要释放实体急停后再清错",
    -4: "本体下电，需要 PowerOn",
    -5: "脚本运行中，需要 Stop",
    -7: "脚本暂停中，需要 Stop",
    -8: "机器人认证过期，TCP 不可用",
}

ROBOT_MODE_NAMES = {
    1: "INIT",
    2: "BRAKE_OPEN",
    3: "POWEROFF",
    4: "DISABLED",
    5: "ENABLE",
    6: "BACKDRIVE",
    7: "RUNNING",
    8: "SINGLE_MOVE",
    9: "ERROR",
    10: "PAUSE",
    11: "COLLISION",
}

def dashboard_error_hint(result):
    return DASHBOARD_ERROR_HINTS.get(result, "请查看控制器返回码")

def dashboard_command(client, command, name="", required=False, timeout=None):
    if timeout is None:
        timeout = dashboard_timeout_for(command)
    client.sendall(command.encode())
    response_text = recv_dashboard_response(client, timeout=timeout, expected_command=command, name=name)
    result = parse_dashboard_result(response_text)
    prefix = f"{name} " if name else ""
    print(f"{prefix}{command} 返回：{response_text}")
    if required and result != 0:
        hint = dashboard_error_hint(result)
        raise RuntimeError(f"{prefix}{command} 失败：{response_text}，{hint}")
    return result, response_text

def safe_dashboard_command(client, command, name="", timeout=None):
    try:
        return dashboard_command(client, command, name=name, required=False, timeout=timeout)
    except (ConnectionError, TimeoutError, OSError) as e:
        prefix = f"{name} " if name else ""
        print(f"{prefix}{command} 发送失败：{e}")
        return None, ""

def get_robot_mode(client, name=""):
    result, response_text = dashboard_command(client, "RobotMode()", name=name, required=True)
    payload = parse_dashboard_payload(response_text)
    try:
        mode = int(payload.split(",", 1)[0].strip())
    except (ValueError, IndexError):
        raise RuntimeError(f"{name} RobotMode 返回解析失败：{response_text}")
    return mode

def robot_mode_name(mode):
    return ROBOT_MODE_NAMES.get(mode, "UNKNOWN")

def robot_mode_text(mode):
    return f"{mode}({robot_mode_name(mode)})"

def print_robot_mode(client, name, label):
    mode = get_robot_mode(client, name)
    print(f"{name} {label}：RobotMode={robot_mode_text(mode)}")
    return mode

def wait_robot_mode(client, name, expected_mode=5, timeout=3.0):
    time.sleep(min(timeout, 0.5))
    mode = get_robot_mode(client, name)
    if mode == expected_mode:
        return mode
    raise RuntimeError(f"{name} 初始化后状态不是 ENABLE(5)，当前 RobotMode={robot_mode_text(mode)}")

def wait_robot_mode_in(client, name, expected_modes, timeout, label):
    expected_modes = set(expected_modes)
    time.sleep(min(timeout, 0.5))
    mode = get_robot_mode(client, name)
    if mode in expected_modes:
        return mode
    expected_text = "/".join(robot_mode_text(mode_value) for mode_value in sorted(expected_modes))
    raise RuntimeError(f"{name} {label} 后状态异常，期望 {expected_text}，当前 RobotMode={robot_mode_text(mode)}")

def get_error_id(client, name):
    result, response_text = dashboard_command(client, "GetErrorID()", name=name, required=False)
    if result == 0:
        print(f"{name} 当前报警信息：{parse_dashboard_payload(response_text)}")
    return result, response_text

def EnableRobot(client):
    result, _ = dashboard_command(client, "EnableRobot()", required=True)
    return result

def request_tcp_control(client, name, required=True):
    try:
        result, response_text = dashboard_command(client, "RequestControl()", name=name, required=False)
    except (ConnectionError, TimeoutError, OSError) as e:
        if required:
            raise RuntimeError(f"{name} RequestControl() 发送失败：{e}") from e
        print(f"{name} RequestControl() 发送失败：{e}，先读取 RobotMode() 再决定后续动作")
        return False

    if result == 0:
        return True

    if required:
        raise RuntimeError(
            f"{name} RequestControl() 失败：{response_text}。"
            "请确认控制器处于未上电/下使能状态、未开启手自动模式，且没有其它客户端占用控制权"
        )
    print(f"{name} RequestControl() 未成功，先读取 RobotMode() 再决定后续动作")
    return False

def prepare_robot_for_servoj(client, name):
    print(f"{name} 初始化：先切 TCP，再读取状态，按需上电/使能")
    tcp_ready = request_tcp_control(client, name, required=False)
    mode = print_robot_mode(client, name, "当前状态")

    if mode == 5:
        print(f"{name} 已是 ENABLE 可运行状态，跳过 Stop/PowerOn/EnableRobot")
        return

    if mode == 1:
        mode = wait_robot_mode_in(client, name, (3, 4, 5), timeout=10.0, label="等待初始化完成")
        print(f"{name} 初始化状态结束：RobotMode={robot_mode_text(mode)}")
        if mode == 5:
            print(f"{name} 已是 ENABLE 可运行状态，跳过 Stop/PowerOn/EnableRobot")
            return

    if mode in (6, 7, 8, 10):
        print(f"{name} 当前处于运动/暂停/拖拽状态，先 Stop()")
        safe_dashboard_command(client, "StopDrag()", name=name)
        dashboard_command(client, "Stop()", name=name, required=True)
        mode = wait_robot_mode_in(client, name, (3, 4, 5), timeout=5.0, label="等待 Stop 后空闲")
        if mode == 5:
            print(f"{name} Stop 后已是 ENABLE 可运行状态")
            return

    if mode in (9, 11):
        print(f"{name} 当前为错误/碰撞状态，读取报警并清错")
        get_error_id(client, name)
        for command in ("EmergencyStop(0)", "Stop()", "ClearError()"):
            safe_dashboard_command(client, command, name=name)
            time.sleep(0.15)
        mode = wait_robot_mode_in(client, name, (3, 4, 5), timeout=5.0, label="等待清错完成")
        if mode == 5:
            print(f"{name} 清错后已是 ENABLE 可运行状态")
            return

    if mode == 5:
        print(f"{name} 已是 ENABLE 可运行状态")
        return

    if mode not in (3, 4):
        raise RuntimeError(f"{name} 无法进入允许 RequestControl 的状态，当前 RobotMode={robot_mode_text(mode)}")

    if not tcp_ready:
        request_tcp_control(client, name, required=True)

    if mode == 3:
        dashboard_command(client, "PowerOn()", name=name, required=True)
        mode = wait_robot_mode_in(client, name, (4, 5), timeout=5.0, label="等待上电完成")

    if mode == 4:
        dashboard_command(client, "EnableRobot()", name=name, required=True)
    elif mode != 5:
        raise RuntimeError(f"{name} 上电后状态异常：RobotMode={robot_mode_text(mode)}")

    mode = wait_robot_mode(client, name, expected_mode=5, timeout=3.0)
    print(f"{name} 初始化完成：RobotMode={robot_mode_text(mode)}")

def iter_dashboard_responses(text):
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if chunk:
            yield parse_dashboard_result(chunk), chunk

def send_dashboard_once(ip, port, command):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(3.0)
    try:
        sock.connect((ip, port))
        sock.sendall(command.encode())
        return parse_dashboard_result(sock.recv(200).decode(errors="ignore"))
    except OSError as e:
        print(f"{ip}:{port} {command} 失败：{e}")
        return -1
    finally:
        sock.close()

stop = True
SERVER_ADDRESSES = ['192.168.1.1', '192.168.1.2']
SERVER_PORT = 29999

def create_client_socket(ip, port):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 设置TCP_NODELAY禁用Nagle算法
    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 100000)
    client_socket.connect((ip, port))
    print(f'成功连接到服务端 {ip}:{port}')
    return client_socket

client_sockets = [create_client_socket(ip, SERVER_PORT) for ip in SERVER_ADDRESSES]
def worker(client_socket, name):  #子线程接受模式
    global stop
    recv_buffer = DASHBOARD_RECV_BUFFERS.pop(client_socket, "")
    while stop:
        try:
            data = client_socket.recv(1024).decode()
            if data:
                recv_buffer += data
                responses = recv_buffer.split(";")
                recv_buffer = responses.pop()
                for result, response_text in iter_dashboard_responses(";".join(responses)):
                    print(f'{name} 返回：{response_text};')
                    if result != 0:
                        stop = False
                        print(f'{name} 返回异常({result})：{response_text};，停止运动')
                        break
            else:
                stop = False
                print(f'{name} 连接断开，停止运动')
                break
            if not stop:
                break
        except BlockingIOError:
            time.sleep(0.001)
        except OSError as e:
            if stop:
                stop = False
                print(f'{name} 接收失败：{e}，停止运动')
            break
time_h = time.time()
state_node = None
try:
    rclpy.init()
    state_node = ArmRosBridge(client_sockets)
    for index, client_socket in enumerate(client_sockets, start=1):
        arm_name = f'第{index}个机械臂'
        prepare_robot_for_servoj(client_socket, arm_name)
        payload_result = send_set_payload(client_socket, *DEFAULT_PAYLOAD, wait_response=True)
        if payload_result != 0:
            raise RuntimeError(f"{arm_name} SetPayload 失败，返回码：{payload_result}")

    feedback_connections = {}
    for ip, side in zip(SERVER_ADDRESSES, ["left", "right"]):
        feed_v = fankuis(ip, 30004)
        feedback_connections[side] = feed_v

    print("力传感器无需在初始化时在线，开始给双臂末端下电")
    for index, client_socket in enumerate(client_sockets, start=1):
        dashboard_command(
            client_socket,
            "SetToolPower(0)",
            name=f'第{index}个机械臂',
            required=True,
        )
    print("左右机械臂末端已下电")

    for side in ["left", "right"]:
        feed_thread1 = threading.Thread(
            target=joint,
            args=(feedback_connections[side], side, state_node))  # 机器状态反馈线程
        feed_thread1.daemon = True
        feed_thread1.start()
    for index, client_socket in enumerate(client_sockets, start=1):
        feed_thread = threading.Thread(
            target=worker,
            args=(client_socket, f'第{index}个机械臂'))  # 机器状态反馈线程
        feed_thread.daemon = True
        feed_thread.start()
    # 复位
    # for client_socket in client_sockets:
    #     servoj_2(client_socket, 0)
    sleep(1)
    rclpy.spin(state_node)
except KeyboardInterrupt:
    print("用户中断操作。")
    print((time.time()-time_h)/60)
except Exception as e:
    print("发生异常：", e)
    print((time.time() - time_h) / 60)
finally:
    stop = False
    if state_node is not None:
        state_node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

print((time.time()-time_h)/60)
