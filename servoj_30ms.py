# 两个臂连接
# 两个臂 EnableRobot()
# 两个臂先 servoj_2(..., 0) 复位
# sleep(3)
# rclpy.spin(state_node) 等待 /g01/joint_commands
# 控制和臂位置是角度
# tcp读取状态100hz,发布指令33hz

import socket
from time import sleep
import time
import threading
import logging
from logging.handlers import RotatingFileHandler
import socket
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
from dobot_msgs_v4.srv import SetToolPower

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
COMMAND_TOPIC = "/g01/joint_commands"
LEFT_TOOL_COMMAND_SERVICE = "/g01/left/tool_commands"
RIGHT_TOOL_COMMAND_SERVICE = "/g01/right/tool_commands"
DEFAULT_PAYLOAD = (1.1, 0.0, 0.0, 45.0)
TOOL_POWER_ON_PAYLOAD = (4.85, 0.0, 0.0, 87.0)


class fankuis():
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.socket_feedback = 0

        if self.port == 30005 or self.port == 30004:
            self.socket_feedback = socket.socket()
            self.socket_feedback.settimeout(1)
            self.socket_feedback.connect((self.ip, self.port))
        else:
            print("Connect to feedback server need use port 30003 !")

    def feed(self):
        try:
            self.socket_feedback.setblocking(True)  # 需要先设置为非阻塞, 使用select超时机制清空
            self.all = self.socket_feedback.recv(10240)
            data = self.all[0:1440]
            # print(data)
            a = np.frombuffer(data, dtype=MyType)
            if hex((a['test_value'][0])) == '0x123456789abcdef':
                tool_v = a['tool_vector_actual'][0]
                tool_j = a['q_actual'][0]
            return [tool_v, tool_j]
        except:
            return ["NG"]
            print("反馈接收解析失败")

class ArmRosBridge(Node):
    def __init__(self, clients):
        super().__init__("dobot_arm_ros_bridge")
        self.arm_clients = clients
        self.left_pub = self.create_publisher(Float32MultiArray, "/g01/left_arm/state", 10)
        self.right_pub = self.create_publisher(Float32MultiArray, "/g01/right_arm/state", 10)
        self.create_subscription(JointState, COMMAND_TOPIC, self._on_joint_commands, 10)
        self.create_service(SetToolPower, LEFT_TOOL_COMMAND_SERVICE, self._on_left_tool_command)
        self.create_service(SetToolPower, RIGHT_TOOL_COMMAND_SERVICE, self._on_right_tool_command)

    def publish_arm_state(self, side, q):
        msg = Float32MultiArray()
        msg.data = [math.radians(float(num)) for num in q]
        if side == "left":
            self.left_pub.publish(msg)
        elif side == "right":
            self.right_pub.publish(msg)

    def _extract_arm_command(self, name_to_pos, joints):
        out = []
        for joint in joints:
            if joint not in name_to_pos:
                return None
            out.append(float(name_to_pos[joint]))
        return out

    def _on_joint_commands(self, msg):
        name_to_pos = {}
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                name_to_pos[str(name)] = float(msg.position[i])
    
        left_cmd = self._extract_arm_command(name_to_pos, LEFT_JOINTS)
        right_cmd = self._extract_arm_command(name_to_pos, RIGHT_JOINTS)

        # 把数据写入内核发送缓冲区，没 recv(200)
        if left_cmd is not None:
            ServoJ(self.arm_clients[0], [math.degrees(value) for value in left_cmd])
        if right_cmd is not None:
            ServoJ(self.arm_clients[1], [math.degrees(value) for value in right_cmd])

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


q_actual = {"left": [], "right": []}
def joint(ip, side, state_node):
    global q_actual
    feed_v = fankuis(ip, 30004)
    while stop:
        actual = feed_v.feed()
        #print(actual)
        if actual != ["NG"]:
           q_actual[side] = [round(num, 6) for num in actual[1]]
           state_node.publish_arm_state(side, q_actual[side])
           #print(q_actual[side])  # 输出: [3.141593, 2.718282, 1.414214]
        time.sleep(0.01)

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

def send_set_payload(client, load, x, y, z, wait_response=False):
    tcp_command = "SetPayload(%s,%s,%s,%s)" % (
        float(load),
        float(x),
        float(y),
        float(z),
    )
    tcp_command = tcp_command.encode()
    client.sendall(tcp_command)
    if not wait_response:
        return 0
    response_text = recv_dashboard_response(client)
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

def recv_dashboard_response(client, timeout=3.0):
    old_timeout = client.gettimeout()
    client.settimeout(timeout)
    response_text = ""
    try:
        while ";" not in response_text:
            data = client.recv(1024)
            if not data:
                raise ConnectionError("dashboard 连接断开")
            response_text += data.decode(errors="ignore")
    except socket.timeout as e:
        raise TimeoutError("等待 dashboard 返回超时") from e
    finally:
        client.settimeout(old_timeout)
    return response_text.split(";", 1)[0].strip() + ";"

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

def dashboard_command(client, command, name="", required=False, timeout=3.0):
    client.sendall(command.encode())
    response_text = recv_dashboard_response(client, timeout=timeout)
    result = parse_dashboard_result(response_text)
    prefix = f"{name} " if name else ""
    print(f"{prefix}{command} 返回：{response_text}")
    if required and result != 0:
        hint = dashboard_error_hint(result)
        raise RuntimeError(f"{prefix}{command} 失败：{response_text}，{hint}")
    return result, response_text

def get_robot_mode(client, name=""):
    result, response_text = dashboard_command(client, "RobotMode()", name=name, required=True)
    payload = parse_dashboard_payload(response_text)
    try:
        mode = int(payload.split(",", 1)[0].strip())
    except (ValueError, IndexError):
        raise RuntimeError(f"{name} RobotMode 返回解析失败：{response_text}")
    return mode

def wait_robot_mode(client, name, expected_mode=5, timeout=3.0):
    deadline = time.time() + timeout
    last_mode = None
    while time.time() < deadline:
        mode = get_robot_mode(client, name)
        last_mode = mode
        if mode == expected_mode:
            return mode
        time.sleep(0.2)
    mode_name = ROBOT_MODE_NAMES.get(last_mode, "UNKNOWN")
    raise RuntimeError(f"{name} 初始化后状态不是 ENABLE(5)，当前 {last_mode}({mode_name})")

def EnableRobot(client):
    result, _ = dashboard_command(client, "EnableRobot()", required=True)
    return result

def prepare_robot_for_servoj(client, name):
    print(f"{name} 初始化：释放急停/停止脚本/清错/切 TCP/上使能")

    # 这些命令用于清理控制器残留状态，失败时继续向后恢复，最终由关键步骤兜底。
    for command in ("EmergencyStop(0)", "Stop()", "ClearError()", "PowerOn()", "DisableRobot()"):
        try:
            dashboard_command(client, command, name=name, required=False)
        except (ConnectionError, TimeoutError, OSError) as e:
            print(f"{name} {command} 发送失败：{e}")
        time.sleep(0.1)

    result, response_text = dashboard_command(client, "RequestControl()", name=name, required=False)
    if result != 0:
        print(f"{name} 第一次切 TCP 模式失败：{response_text}，再清一次状态后重试")
        for command in ("EmergencyStop(0)", "Stop()", "ClearError()", "PowerOn()", "DisableRobot()"):
            dashboard_command(client, command, name=name, required=False)
            time.sleep(0.1)
        dashboard_command(client, "RequestControl()", name=name, required=True)

    for command in ("Stop()", "EmergencyStop(0)", "ClearError()", "PowerOn()"):
        dashboard_command(client, command, name=name, required=False)
        time.sleep(0.1)

    dashboard_command(client, "EnableRobot()", name=name, required=True)
    mode = wait_robot_mode(client, name, expected_mode=5, timeout=3.0)
    mode_name = ROBOT_MODE_NAMES.get(mode, "UNKNOWN")
    print(f"{name} 初始化完成：RobotMode={mode}({mode_name})")

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
SERVER_ADDRESSES = ['192.168.5.1', '192.168.5.2']
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
    recv_buffer = ""
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
    for ip, side in zip(SERVER_ADDRESSES, ["left", "right"]):
        feed_thread1 = threading.Thread(
            target=joint,
            args=(ip, side, state_node))  # 机器状态反馈线程
        feed_thread1.daemon = True
        feed_thread1.start()
    for index, client_socket in enumerate(client_sockets, start=1):
        prepare_robot_for_servoj(client_socket, f'第{index}个机械臂')
        payload_result = send_set_payload(client_socket, *DEFAULT_PAYLOAD, wait_response=True)
        if payload_result != 0:
            raise RuntimeError(f"第{index}个机械臂 SetPayload 失败，返回码：{payload_result}")
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
