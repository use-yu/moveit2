#!/usr/bin/env python3
"""执行 G01 双臂的指定关节姿态，再用关节反馈和 MoveIt FK 报告到位误差。"""

import argparse
import math
import time

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetPositionFK
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState


BASE_FRAME = "moveit_base_link"
ARMS = (
    ("左臂", "l", "l_base_link", "l_tool", (37, -63, -55, -48, 88, 39)),
    ("右臂", "r", "r_base_link", "r_tool", (-37, 63, 55, 48, -88, -39)),
)
TARGET_JOINTS = {
    f"{side}_arm_joint{axis}": math.radians(degrees)
    for _, side, _, _, angles in ARMS
    for axis, degrees in enumerate(angles, 1)
}
REQUIRED_JOINTS = (*TARGET_JOINTS, "base_joint1", "base_joint2", "body_joint1", "body_joint2", "head_joint")
FK_LINKS = [BASE_FRAME, "l_base_link", "r_base_link", "l_tool", "r_tool"]


def robot_state(joints):
    return RobotState(
        is_diff=True,
        joint_state=JointState(name=list(joints), position=list(joints.values())),
    )


def pose_matrix(pose):
    p, q = pose.position, pose.orientation
    matrix = np.eye(4)
    matrix[:3, 3] = [p.x, p.y, p.z]
    matrix[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    return matrix


def format_vector(values, digits):
    return "(" + ", ".join(f"{value:.{digits}f}" for value in values) + ")"


def print_pose(label, pose):
    quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
    print(f"  {label} xyz (m) {format_vector(pose[:3, 3], 6)}  "
          f"quaternion xyzw {format_vector(quat, 6)}")


def print_report(target_poses, actual_poses, actual_joints):
    for label, side, arm_base, tool, _ in ARMS:
        print(f"\n{label} {tool}")
        for frame in (BASE_FRAME, arm_base):
            target = np.linalg.solve(target_poses[frame], target_poses[tool])
            actual = np.linalg.solve(actual_poses[frame], actual_poses[tool])
            print(f"  坐标系 {frame}")
            print_pose("目标", target)
            print_pose("实测关节 FK", actual)
            delta_mm = (actual[:3, 3] - target[:3, 3]) * 1000
            relative_rotation = target[:3, :3].T @ actual[:3, :3]
            angle_deg = math.degrees(Rotation.from_matrix(relative_rotation).magnitude())
            print(f"  位置偏差 Δxyz (mm) {format_vector(delta_mm, 3)}  "
                  f"距离 {np.linalg.norm(delta_mm):.3f} mm  姿态偏差 {angle_deg:.4f}°")

        print("  关节        目标 (°)     反馈 (°)     反馈−目标 (°)")
        for axis in range(1, 7):
            name = f"{side}_arm_joint{axis}"
            target_deg = math.degrees(TARGET_JOINTS[name])
            actual_deg = math.degrees(actual_joints[name])
            print(f"  J{axis}       {target_deg:10.4f}   {actual_deg:10.4f}   {actual_deg - target_deg:+12.4f}")


class DualArmPoseCheck(Node):
    def __init__(self, args):
        super().__init__("g01_dual_arm_pose_check")
        self.args = args
        self.feedback = None
        self.create_subscription(JointState, args.joint_topic, self.on_joints, qos_profile_sensor_data)
        self.move = ActionClient(self, MoveGroup, "/move_action")
        self.fk = self.create_client(GetPositionFK, "/compute_fk")

    def on_joints(self, msg):
        joints = dict(zip(msg.name, msg.position))
        if all(name in joints and math.isfinite(joints[name]) for name in REQUIRED_JOINTS):
            stamp = msg.header.stamp
            self.feedback = (stamp.sec * 1_000_000_000 + stamp.nanosec,
                             {name: joints[name] for name in REQUIRED_JOINTS})

    def fresh_joints(self, delay=0.0):
        after = self.get_clock().now().nanoseconds + round(delay * 1e9)
        ready = time.monotonic() + delay
        deadline = ready + 5.0
        while rclpy.ok() and time.monotonic() < deadline:
            if time.monotonic() >= ready and self.feedback is not None and self.feedback[0] > after:
                return self.feedback[1]
            rclpy.spin_once(self, timeout_sec=0.05)
        raise RuntimeError(f"未从 {self.args.joint_topic} 收到新的完整关节反馈。")

    def wait_result(self, future, timeout, operation):
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done():
            raise RuntimeError(f"{operation}超时。")
        result = future.result()
        if result is None:
            raise RuntimeError(f"{operation}未返回结果。")
        return result

    def forward_poses(self, joints):
        request = GetPositionFK.Request(fk_link_names=FK_LINKS, robot_state=robot_state(joints))
        # 让服务返回机器人模型坐标系中的 FK。之后从同一关节快照计算各参考系中的相对位姿。
        response = self.wait_result(self.fk.call_async(request), 5.0, "MoveIt FK")
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"MoveIt FK 失败，错误码 {response.error_code.val}。")
        frames = {pose.header.frame_id for pose in response.pose_stamped}
        if (len(response.pose_stamped) != len(FK_LINKS)
                or set(response.fk_link_names) != set(FK_LINKS)
                or len(frames) != 1 or "" in frames):
            raise RuntimeError("MoveIt FK 返回的连杆或坐标系不完整。")
        return {link: pose_matrix(pose.pose)
                for link, pose in zip(response.fk_link_names, response.pose_stamped)}

    def move_arms(self, start):
        goal = MoveGroup.Goal()
        goal.request.group_name = "dual_arm"
        goal.request.start_state = robot_state(start)
        goal.request.num_planning_attempts = 5
        goal.request.allowed_planning_time = 10.0
        goal.request.max_velocity_scaling_factor = self.args.speed
        goal.request.max_acceleration_scaling_factor = self.args.speed
        goal.request.goal_constraints = [Constraints(joint_constraints=[
            JointConstraint(joint_name=name, position=position, weight=1.0,
                            tolerance_above=0.001, tolerance_below=0.001)
            for name, position in TARGET_JOINTS.items()
        ])]
        goal.planning_options.plan_only = self.args.plan_only
        self.get_logger().info("开始规划双臂目标姿态。" if self.args.plan_only else "开始规划及执行双臂目标姿态。")

        send = self.move.send_goal_async(goal)
        handle = None
        try:
            handle = self.wait_result(send, 10.0, "确认运动目标")
            if not handle.accepted:
                raise RuntimeError("MoveIt 拒绝运动目标。")
            result = self.wait_result(handle.get_result_async(), 120.0, "双臂规划或执行")
        except (RuntimeError, KeyboardInterrupt):
            if handle is None:
                rclpy.spin_until_future_complete(self, send, timeout_sec=2.0)
                if send.done() and not send.cancelled() and send.exception() is None:
                    handle = send.result()
            if handle is not None and handle.accepted:
                cancel = handle.cancel_goal_async()
                rclpy.spin_until_future_complete(self, cancel, timeout_sec=3.0)
            raise
        if result.status != GoalStatus.STATUS_SUCCEEDED or result.result.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"双臂规划或执行失败，动作状态 {result.status}，"
                               f"MoveIt 错误码 {result.result.error_code.val}。")

    def run(self):
        if not self.move.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("MoveIt /move_action 不可用。")
        if not self.fk.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("MoveIt /compute_fk 不可用。")
        start = self.fresh_joints()
        target_poses = self.forward_poses({**start, **TARGET_JOINTS})
        self.move_arms(start)
        if self.args.plan_only:
            print("双臂规划通过。")
            return
        actual_joints = self.fresh_joints(delay=self.args.settle)
        actual_poses = self.forward_poses(actual_joints)
        print_report(target_poses, actual_poses, actual_joints)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-topic", default="/g01/joint_states")
    parser.add_argument("--speed", type=float, default=0.10, help="MoveIt 速度及加速度比例（默认 0.10）")
    parser.add_argument("--settle", type=float, default=0.6, help="执行完成后等待反馈的时间（秒）")
    parser.add_argument("--plan-only", action="store_true", help="只检查 MoveIt 规划")
    args, ros_args = parser.parse_known_args()
    if ros_args and ros_args[0] != "--ros-args":
        parser.error(f"未知参数 {' '.join(ros_args)}")
    if not 0 < args.speed <= 1:
        parser.error("--speed 须在 (0, 1]")
    if not math.isfinite(args.settle) or args.settle < 0:
        parser.error("--settle 须为有限的非负秒数")

    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = DualArmPoseCheck(args)
        node.run()
        return 0
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        print(f"双臂到位检查失败：{error}", flush=True)
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
