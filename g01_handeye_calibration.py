#!/usr/bin/env python3
"""G01 固定头部相机的眼在手外标定，支持 left/right 两条六轴臂。

功能
  1. 从当前末端位姿出发，生成多轴平移、转动及组合采样位姿。
  2. MoveIt 先做带碰撞检查的 IK，再从上一段的实际规划终点规划下一段。
     跳过无解或规划失败的点。全部规划完毕后逐段执行，每段末尾停车。
  3. 实机用 D435i 彩色图像及其 CameraInfo。12×9 格、边长 0.015 m 的
     棋盘有 11×8 个内角点。每次停车后配对图像和关节状态，通过 MoveIt FK
     和 solvePnP 得到一对位姿，保存到 JSON。相机、头部和躯干须固定，
     棋盘必须刚性安装。实机前在 MoveIt 规划场景中配置板、支架和障碍物。
  4. --mode sim 只规划并执行轨迹，跳过相机、采样与求解。

公式（列向量，^aT_b 将 b 系坐标变换到 a 系，平移单位 m）
  b=选中机械臂基座，e=第六轴 L6/R6，c=D435i 彩色光学坐标系，t=棋盘。
  F_i = ^bT_e(q_i)；C_i = ^cT_t(i)，由像素 u ~ project(K, D, C_i P_t) 求得。
  X = ^bT_c 是所求外参；Y = ^eT_t 是固定的棋盘安装变换。
  F_i Y = X C_i  =>  F_i^-1 X C_i = Y。
  两帧相消： (F_j F_i^-1) X = X (C_j C_i^-1)。
  沿用 realsense-hand-eye-calibration/handeye_solver.py 的四种 OpenCV 方法，
  但 gripper2base 输入 F_i^-1，target2cam 输入 C_i，使其输出 X。
  用 Y_i = F_i^-1 X C_i 的平移及旋转离散程度选解，并输出残差。

用法（先 source ROS 2 Humble 和工作区）
  ros2 launch g01_moveit_config demo.launch.py use_real_hardware:=false
  python3 g01_handeye_calibration.py --arm left --mode sim

  实机先启动 G01 驱动、RealSense ROS 驱动和彩色流，然后启动：
  ros2 launch g01_moveit_config demo.launch.py use_real_hardware:=true
  将棋盘移入画面，再执行：
  python3 g01_handeye_calibration.py --arm right --mode real
  默认订阅 /camera/camera/color/{image_raw,camera_info}，可用 --image-topic、
  --info-topic 覆盖。建议使用原始彩色图像以及对应的 CameraInfo。
"""

import argparse
from collections import deque
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from action_msgs.msg import GoalStatus
from controller_manager_msgs.srv import ListHardwareComponents
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CameraInfo, Image, JointState


PATTERN = (11, 8)
SQUARE_M = 0.015
# 偏移量乘以用户指定的平移/旋转幅度。先试纯旋转，再覆盖位置及组合。
OFFSETS = (
    ((0, 0, 0), (1, 0, 0)), ((0, 0, 0), (-1, 0, 0)),
    ((0, 0, 0), (0, 1, 0)), ((0, 0, 0), (0, -1, 0)),
    ((0, 0, 0), (0, 0, 1)), ((0, 0, 0), (0, 0, -1)),
    ((1, 0, 0), (0, 0, 0)), ((-1, 0, 0), (0, 0, 0)),
    ((0, 1, 0), (0, 0, 0)), ((0, -1, 0), (0, 0, 0)),
    ((0, 0, 1), (0, 0, 0)), ((0, 0, -1), (0, 0, 0)),
    ((1, 0, 0), (0, 1, 0)), ((-1, 0, 0), (0, -1, 0)),
    ((0, 1, 0), (-1, 0, 0)), ((0, -1, 0), (1, 0, 0)),
    ((0, 0, 1), (1, 1, 0)), ((0, 0, -1), (-1, -1, 0)),
    ((1, 1, 0), (1, -1, 0)), ((-1, -1, 0), (-1, 1, 0)),
    ((0, 1, 1), (0, 1, 1)), ((0, -1, -1), (0, -1, -1)),
    ((1, 0, -1), (1, 0, -1)), ((-1, 0, 1), (-1, 0, 1)),
)


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def pose_matrix(pose):
    p, q = pose.position, pose.orientation
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    matrix[:3, 3] = [p.x, p.y, p.z]
    return matrix


def target_pose(frame, matrix):
    pose = PoseStamped()
    pose.header.frame_id = frame
    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = matrix[:3, 3].tolist()
    q = Rotation.from_matrix(matrix[:3, :3]).as_quat().tolist()
    pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = q
    return pose


def robot_state(joints):
    state = RobotState(is_diff=True)
    state.joint_state.name = list(joints)
    state.joint_state.position = list(joints.values())
    return state


def solve_handeye(samples, cv):
    fk = np.asarray([sample["T_base_ee"] for sample in samples])
    camera = np.asarray([sample["T_camera_board"] for sample in samples])
    inverse_fk = np.linalg.inv(fk)
    relative = fk[0, :3, :3].T @ fk[:, :3, :3]
    singular = np.linalg.svd(Rotation.from_matrix(relative).as_rotvec(), compute_uv=False)
    if singular[1] < math.radians(8) or singular[1] < 0.1 * singular[0]:
        raise RuntimeError("有效样本的转动方向过于单一，无法可靠标定。")

    results = []
    for name, method in (
        ("Tsai-Lenz", cv.CALIB_HAND_EYE_TSAI),
        ("Park", cv.CALIB_HAND_EYE_PARK),
        ("Horaud", cv.CALIB_HAND_EYE_HORAUD),
        ("Daniilidis", cv.CALIB_HAND_EYE_DANIILIDIS),
    ):
        try:
            rotation, translation = cv.calibrateHandEye(
                inverse_fk[:, :3, :3], inverse_fk[:, :3, 3],
                camera[:, :3, :3], camera[:, :3, 3], method=method,
            )
        except cv.error:
            continue
        if (not np.isfinite(rotation).all() or not np.isfinite(translation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)):
            continue
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = rotation, translation.ravel()
        boards = inverse_fk @ transform @ camera
        positions = boards[:, :3, 3]
        rotations = Rotation.from_matrix(boards[:, :3, :3])
        trans_errors = np.linalg.norm(positions - positions.mean(axis=0), axis=1)
        angle_errors = (rotations.mean().inv() * rotations).magnitude()
        rmse_m = float(np.sqrt(np.mean(trans_errors**2)))
        rmse_rad = float(np.sqrt(np.mean(angle_errors**2)))
        # 用棋盘对角线将角度误差转换为同量纲的长度，比较四种算法。
        score = math.hypot(rmse_m, math.hypot(12, 9) * SQUARE_M * rmse_rad)
        results.append({
            "method": name, "T_base_camera": transform.tolist(),
            "translation_m": translation.ravel().tolist(),
            "quaternion_xyzw": Rotation.from_matrix(rotation).as_quat().tolist(),
            "board_translation_rmse_mm": 1000 * rmse_m,
            "board_rotation_rmse_deg": math.degrees(rmse_rad),
            "score_m": score,
        })
    if not results:
        raise RuntimeError("四种手眼算法都未得到有效解。")
    results.sort(key=lambda result: result["score_m"])
    return {"best": results[0], "methods": results}


class ChessboardCamera:
    def __init__(self, node, args):
        import cv2

        self.cv = cv2
        self.info = None
        self.images = deque(maxlen=6)
        self.points = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
        self.points[:, :2] = np.mgrid[:PATTERN[0], :PATTERN[1]].T.reshape(-1, 2) * SQUARE_M
        node.create_subscription(CameraInfo, args.info_topic, self.on_info, qos_profile_sensor_data)
        node.create_subscription(Image, args.image_topic, self.on_image, qos_profile_sensor_data)

    def on_info(self, msg):
        self.info = msg

    def on_image(self, msg):
        self.images.append(msg)

    def detect(self, image, info):
        cv = self.cv
        if ((image.width, image.height) != (info.width, info.height)
                or not image.header.frame_id or image.header.frame_id != info.header.frame_id
                or info.distortion_model not in ("plumb_bob", "rational_polynomial", "")
                or info.k[0] <= 0 or info.k[4] <= 0
                or info.binning_x > 1 or info.binning_y > 1
                or info.roi.x_offset or info.roi.y_offset
                or info.roi.width not in (0, info.width)
                or info.roi.height not in (0, info.height)):
            raise RuntimeError("彩色图像和 CameraInfo 不匹配，或相机内参/畸变模型不支持。")
        channels = {"mono8": 1, "rgb8": 3, "bgr8": 3}.get(image.encoding)
        if channels is None:
            raise RuntimeError(f"不支持图像编码 {image.encoding}。")
        rows = np.frombuffer(bytes(image.data), dtype=np.uint8).reshape(image.height, image.step)
        pixels = rows[:, :image.width * channels]
        gray = pixels if channels == 1 else cv.cvtColor(
            pixels.reshape(image.height, image.width, channels),
            cv.COLOR_RGB2GRAY if image.encoding == "rgb8" else cv.COLOR_BGR2GRAY,
        )
        found, corners = cv.findChessboardCornersSB(gray, PATTERN, cv.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            return None
        grid = corners.reshape(PATTERN[1], PATTERN[0], 2)
        # 对角内格是反色（11×8 内角点）。固定棋盘的第一个内格为黑色，
        # 使角点顺序在检测器发生 180° 翻转时仍指向同一实体坐标。
        first = grid[:2, :2].mean(axis=(0, 1))
        last = grid[-2:, -2:].mean(axis=(0, 1))
        a, b = (float(cv.getRectSubPix(gray, (3, 3), tuple(point)).mean())
                for point in (first, last))
        if abs(a - b) < 20:
            return None
        if a > b:
            corners = np.ascontiguousarray(corners[::-1])
        k, d = np.asarray(info.k).reshape(3, 3), np.asarray(info.d)
        ok, rvec, tvec = cv.solvePnP(self.points, corners, k, d)
        if not ok or not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            return None
        rotation = cv.Rodrigues(rvec)[0]
        if np.min((self.points @ rotation.T + tvec.ravel())[:, 2]) <= 0:
            return None
        projected = cv.projectPoints(self.points, rvec, tvec, k, d)[0]
        rmse = float(np.sqrt(np.mean(np.sum((projected - corners)**2, axis=-1))))
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = rotation, tvec.ravel()
        return transform, rmse


class RosResponseTimeout(RuntimeError):
    pass


class Calibration(Node):
    def __init__(self, args):
        super().__init__("g01_handeye_calibration")
        self.args = args
        side = args.arm[0]
        self.base, self.tip, self.group = f"{side}_base_link", f"{side.upper()}6", f"{args.arm}_arm"
        self.arm = [f"{side}_arm_joint{i}" for i in range(1, 7)]
        self.required = ["base_joint1", "base_joint2", "body_joint1", "body_joint2", "head_joint"]
        self.required += [f"{s}_arm_joint{i}" for s in ("l", "r") for i in range(1, 7)]
        self.fixed = [name for name in self.required if name not in self.arm]
        self.history = deque(maxlen=300)
        self.create_subscription(JointState, args.joint_topic, self.on_joints, qos_profile_sensor_data)
        self.fk_client = self.create_client(GetPositionFK, "/compute_fk")
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")
        self.hardware_client = self.create_client(
            ListHardwareComponents, "/controller_manager/list_hardware_components"
        )
        self.move_client = ActionClient(self, MoveGroup, "/move_action")
        self.execute_client = ActionClient(self, ExecuteTrajectory, "/execute_trajectory")
        self.camera = ChessboardCamera(self, args) if args.mode == "real" else None

    def on_joints(self, msg):
        joints = dict(zip(msg.name, msg.position))
        if all(name in joints and math.isfinite(joints[name]) for name in self.required):
            self.history.append((time.monotonic(), stamp_ns(msg.header.stamp),
                                 {name: joints[name] for name in self.required}))

    def await_future(self, future, timeout, operation):
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done():
            raise RosResponseTimeout(f"{operation} 在 {timeout:g} 秒内未响应。")
        return future.result()

    def query(self, client, request, *, detail=""):
        # 只对无副作用的查询重试，避免重复发送轨迹或动作目标。
        operation = f"服务 {client.srv_name}{detail}"
        for attempt in range(1, 4):
            future = client.call_async(request)
            try:
                return self.await_future(future, 5.0, operation)
            except RosResponseTimeout:
                client.remove_pending_request(future)
                if attempt == 3:
                    raise RosResponseTimeout(f"{operation} 连续 {attempt} 次未响应（每次等待 5 秒）。") from None
                self.get_logger().warning(f"{operation} 第 {attempt} 次查询超时，正在重试。")

    def wait_until(self, predicate, timeout):
        deadline = time.monotonic() + timeout
        while rclpy.ok():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            value = predicate()
            if value is not None:
                return value
            rclpy.spin_once(self, timeout_sec=min(remaining, 0.05))
        return None

    def current(self):
        since = time.monotonic()
        value = self.wait_until(
            lambda: self.history[-1][2] if self.history and self.history[-1][0] >= since else None,
            5.0,
        )
        if value is None:
            raise RuntimeError(f"{self.args.joint_topic} 未收到完整的 G01 关节反馈。")
        return value

    def require_close(self, actual, expected, names, angle_tol, label):
        changed = [name for name in names
                   if abs(actual[name] - expected[name]) > (0.002 if name in
                      ("base_joint1", "body_joint1") else angle_tol)]
        if changed:
            raise RuntimeError(f"{label}：{', '.join(changed)}。停止执行。")

    def connect(self):
        for client in (self.hardware_client, self.fk_client, self.ik_client):
            if not client.wait_for_service(timeout_sec=30.0):
                raise RuntimeError(f"服务 {client.srv_name} 不可用。")
        result = self.query(self.hardware_client, ListHardwareComponents.Request())
        expected = ("mock_components/GenericSystem" if self.args.mode == "sim"
                    else "g01_topic_hardware/G01TopicSystem")
        components = [item for item in result.component if item.name == "G01System"]
        if len(components) != 1 or components[0].class_type != expected:
            raise RuntimeError(f"--mode {self.args.mode} 要求 ros2_control 使用 {expected}。")
        for client in (self.move_client, self.execute_client):
            if not client.wait_for_server(timeout_sec=10.0):
                raise RuntimeError("MoveIt 规划或执行 action 不可用。")

    def fk(self, joints, frame):
        request = GetPositionFK.Request()
        request.header.frame_id = frame
        request.fk_link_names = [self.tip]
        request.robot_state = robot_state(joints)
        response = self.query(self.fk_client, request, detail=f"（{self.tip} → {frame}）")
        if (response.error_code.val != MoveItErrorCodes.SUCCESS
                or len(response.pose_stamped) != 1 or response.pose_stamped[0].header.frame_id != frame):
            raise RuntimeError(f"MoveIt FK 失败，坐标系 {frame}，错误码 {response.error_code.val}。")
        return pose_matrix(response.pose_stamped[0].pose)

    def action(self, client, goal, timeout, operation):
        send = client.send_goal_async(goal)
        handle = None
        try:
            handle = self.await_future(send, 10, f"{operation}的目标确认")
            if not handle.accepted:
                raise RuntimeError("MoveIt 拒绝 action 目标。")
            return self.await_future(handle.get_result_async(), timeout, f"{operation}的结果")
        except (RuntimeError, KeyboardInterrupt):
            if handle is None:
                # 给还在途中的目标确认一次机会，以便取消已接受的动作。
                rclpy.spin_until_future_complete(self, send, timeout_sec=2)
                if send.done() and send.exception() is None:
                    handle = send.result()
            if handle is not None and handle.accepted:
                cancel = handle.cancel_goal_async()
                rclpy.spin_until_future_complete(self, cancel, timeout_sec=3)
            raise

    def plan(self, seed, pose, index):
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name, ik.ik_link_name = self.group, self.tip
        ik.pose_stamped, ik.robot_state = pose, robot_state(seed)
        ik.avoid_collisions = True
        ik.timeout.nanosec = 250_000_000
        response = self.query(self.ik_client, request, detail=f"（候选点 {index}）")
        code = response.error_code.val
        if code in (MoveItErrorCodes.NO_IK_SOLUTION, MoveItErrorCodes.GOAL_IN_COLLISION,
                    MoveItErrorCodes.TIMED_OUT):
            return None, code
        if code != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"MoveIt IK 请求失败，错误码 {code}。")
        joints = response.solution.joint_state
        solution = dict(zip(joints.name, joints.position))
        if not all(name in solution and math.isfinite(solution[name]) for name in self.arm):
            raise RuntimeError("MoveIt IK 返回的关节解不完整。")
        if max(abs(seed[name] - solution[name]) for name in self.arm) < 0.005:
            return None, "位姿重复"

        goal = MoveGroup.Goal()
        goal.planning_options.plan_only = True
        goal.request.group_name = self.group
        goal.request.start_state = robot_state(seed)
        goal.request.num_planning_attempts = 3
        goal.request.allowed_planning_time = 3.0
        goal.request.max_velocity_scaling_factor = self.args.speed
        goal.request.max_acceleration_scaling_factor = self.args.speed
        goal.request.goal_constraints = [Constraints(joint_constraints=[
            JointConstraint(joint_name=name, position=solution[name],
                            tolerance_above=0.001, tolerance_below=0.001, weight=1.0)
            for name in self.arm
        ])]
        action = self.action(self.move_client, goal, 15, f"候选点 {index} 的 MoveIt 规划")
        code = action.result.error_code.val
        if code in (MoveItErrorCodes.PLANNING_FAILED, MoveItErrorCodes.GOAL_IN_COLLISION,
                    MoveItErrorCodes.INVALID_MOTION_PLAN, MoveItErrorCodes.TIMED_OUT,
                    MoveItErrorCodes.GOAL_VIOLATES_PATH_CONSTRAINTS,
                    MoveItErrorCodes.GOAL_CONSTRAINTS_VIOLATED):
            return None, code
        if action.status != GoalStatus.STATUS_SUCCEEDED or code != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"MoveIt 规划请求失败，错误码 {code}。")
        trajectory = action.result.planned_trajectory
        jt = trajectory.joint_trajectory
        if (not jt.points or set(jt.joint_names) != set(self.arm)
                or trajectory.multi_dof_joint_trajectory.points):
            raise RuntimeError("MoveIt 返回的轨迹没有覆盖且仅覆盖所选六轴。")
        start = dict(zip(jt.joint_names, jt.points[0].positions))
        self.require_close(start, seed, self.arm, 0.015, "规划轨迹起点与种子不一致")
        return trajectory, code

    def collect(self, index, reference, expected):
        after = self.get_clock().now().nanoseconds + int(self.args.settle * 1e9)
        seen = 0
        best_rmse = None

        def synchronized_frame():
            info = self.camera.info
            if info is None or not self.history:
                return None
            skew = int(self.args.max_skew * 1e9)
            for image in reversed(self.camera.images):
                stamp = stamp_ns(image.header.stamp)
                if stamp <= max(seen, after) or stamp > self.history[-1][1]:
                    continue
                before = next((item for item in reversed(self.history)
                               if after <= item[1] <= stamp), None)
                later = next((item for item in self.history if item[1] >= stamp), None)
                if before is not None and later is not None:
                    if max(stamp - before[1], later[1] - stamp) <= skew:
                        return image, info, before, later
            return None

        deadline = time.monotonic() + self.args.capture_timeout
        while time.monotonic() < deadline:
            pair = self.wait_until(synchronized_frame, deadline - time.monotonic())
            if pair is None:
                break
            image, info, before, later = pair
            seen = stamp_ns(image.header.stamp)
            for frame in (before, later):
                self.require_close(frame[2], reference, self.fixed, 0.005, "头部、躯干或另一条臂发生移动")
            if max(abs(before[2][name] - later[2][name]) for name in self.arm) > 0.002:
                continue
            nearest = min((before, later), key=lambda item: abs(item[1] - seen))
            self.require_close(nearest[2], expected, self.arm, 0.025, "末端未到达轨迹终点")
            detection = self.camera.detect(image, info)
            if detection is None:
                continue
            matrix, rmse = detection
            best_rmse = rmse if best_rmse is None else min(best_rmse, rmse)
            if rmse <= self.args.max_reprojection:
                return {
                    "index": index, "image_stamp_ns": seen, "joint_stamp_ns": nearest[1],
                    "joint_positions_rad_or_m": nearest[2], "camera_frame": image.header.frame_id,
                    "T_base_ee": self.fk(nearest[2], self.base).tolist(),
                    "T_camera_board": matrix.tolist(), "reprojection_rmse_px": rmse,
                }
        if best_rmse is not None:
            self.get_logger().warning(f"点 {index} 的最佳重投影误差 {best_rmse:.2f} px。")
        return None

    def save(self, data):
        output = self.args.output or Path(f"g01_handeye_{self.args.arm}.json")
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output)
        return output

    def run(self):
        self.connect()
        reference = self.current()
        initial = self.fk(reference, "world")
        data = None
        if self.camera is not None:
            first = self.collect(0, reference, reference)
            if first is None:
                raise RuntimeError("起始位姿没有合格的棋盘观测。检查棋盘可见性、内参和图像/关节时钟。")
            data = {"arm": self.args.arm, "base_frame": self.base, "end_effector_frame": self.tip,
                    "camera_frame": first["camera_frame"], "head_joint_rad": reference["head_joint"],
                    "pattern_inner_corners": list(PATTERN), "square_size_m": SQUARE_M,
                    "samples": [first]}
            self.save(data)

        segments = []
        seed = reference.copy()
        rotation_scale = math.radians(self.args.rotation_deg)
        for index, (offset, angles) in enumerate(OFFSETS, 1):
            target = initial.copy()
            target[:3, 3] += initial[:3, :3] @ (self.args.translation_m * np.asarray(offset))
            target[:3, :3] = initial[:3, :3] @ Rotation.from_euler(
                "xyz", rotation_scale * np.asarray(angles)
            ).as_matrix()
            trajectory, code = self.plan(seed, target_pose("world", target), index)
            if trajectory is None:
                self.get_logger().warning(f"跳过候选点 {index}，IK 或规划返回 {code}。")
                continue
            segments.append((index, trajectory))
            jt = trajectory.joint_trajectory
            seed.update(zip(jt.joint_names, jt.points[-1].positions))
            self.get_logger().info(f"候选点 {index} 规划通过。")

        needed = self.args.min_samples - 1 if self.camera is not None else 1
        if len(segments) < needed:
            raise RuntimeError(f"仅有 {len(segments)} 段可执行轨迹，需要至少 {needed} 段。")
        self.get_logger().info(f"开始执行 {len(segments)} 段 MoveIt 轨迹。")
        for index, trajectory in segments:
            jt = trajectory.joint_trajectory
            actual = self.current()
            self.require_close(actual, reference, self.fixed, 0.005, "固定关节发生移动")
            self.require_close(actual, dict(zip(jt.joint_names, jt.points[0].positions)),
                               self.arm, 0.025, "实测关节偏离下一段轨迹起点")
            duration = stamp_ns(jt.points[-1].time_from_start) / 1e9
            result = self.action(self.execute_client, ExecuteTrajectory.Goal(trajectory=trajectory),
                                 duration * 2 + 15, f"候选点 {index} 的轨迹执行")
            code = result.result.error_code.val
            if result.status != GoalStatus.STATUS_SUCCEEDED or code != MoveItErrorCodes.SUCCESS:
                raise RuntimeError(f"第 {index} 段执行失败，MoveIt 错误码 {code}。")
            expected = dict(zip(jt.joint_names, jt.points[-1].positions))
            self.get_logger().info(f"候选点 {index} 执行完成。")
            if data is None:
                self.require_close(self.current(), expected, self.arm, 0.025, "仿真未到达轨迹终点")
                continue
            sample = self.collect(index, reference, expected)
            if sample is None:
                self.get_logger().warning(f"跳过点 {index} 的采样，未获得合格的同步棋盘观测。")
                continue
            if sample["camera_frame"] != data["camera_frame"]:
                raise RuntimeError("采样过程中相机光学坐标系发生变化。")
            data["samples"].append(sample)
            self.save(data)
            self.get_logger().info(f"已采样 {len(data['samples'])} 帧，重投影误差 "
                                   f"{sample['reprojection_rmse_px']:.2f} px。")

        if data is not None:
            if len(data["samples"]) < self.args.min_samples:
                raise RuntimeError(f"只有 {len(data['samples'])} 个有效样本，需至少 "
                                   f"{self.args.min_samples} 个。已保存采集数据。")
            data["calibration"] = solve_handeye(data["samples"], self.camera.cv)
            output = self.save(data)
            best = data["calibration"]["best"]
            self.get_logger().info(f"{best['method']}：{data['camera_frame']} → {self.base}，"
                                   f"棋盘一致性 RMSE {best['board_translation_rmse_mm']:.2f} mm / "
                                   f"{best['board_rotation_rmse_deg']:.2f}°。已保存到 {output}。")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=("left", "right"), required=True)
    parser.add_argument("--mode", choices=("sim", "real"), default="sim")
    parser.add_argument("--joint-topic", default="/g01/joint_states")
    parser.add_argument("--image-topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--info-topic", default="/camera/camera/color/camera_info")
    parser.add_argument("--translation-m", type=float, default=0.035)
    parser.add_argument("--rotation-deg", type=float, default=18.0)
    parser.add_argument("--speed", type=float, default=0.10)
    parser.add_argument("--settle", type=float, default=0.6)
    parser.add_argument("--capture-timeout", type=float, default=5.0)
    parser.add_argument("--max-skew", type=float, default=0.12, help="图像与每侧关节反馈的最大时间差（秒）")
    parser.add_argument("--max-reprojection", type=float, default=1.0, help="最大重投影 RMSE（像素）")
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args, ros_args = parser.parse_known_args()
    if ros_args and ros_args[0] != "--ros-args":
        parser.error(f"未知参数 {' '.join(ros_args)}")
    for name in ("translation_m", "rotation_deg", "speed", "settle",
                 "capture_timeout", "max_skew", "max_reprojection"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须为正数")
    if args.speed > 1 or not 5 <= args.min_samples <= len(OFFSETS) + 1:
        parser.error("--speed 须在 (0, 1]，--min-samples 须在 [5, 25]")
    if args.capture_timeout <= args.settle:
        parser.error("--capture-timeout 必须大于 --settle")

    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = Calibration(args)
        node.run()
        return 0
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, ValueError, OSError, ImportError) as error:
        print(f"标定流程停止：{error}", flush=True)
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
