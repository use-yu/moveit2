# 测试箱子姿态：
# ros2 run tf2_ros tf2_echo moveit_base_link aruco_33

#!/usr/bin/env python3
"""Add the ArUco-tagged box to MoveIt and publish tag and grasp TFs."""

from math import isfinite
from queue import Empty, SimpleQueue
import sys
from threading import Thread
from time import monotonic

import rclpy
from geometry_msgs.msg import Point, Pose, Quaternion, TransformStamped, Vector3
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Float64MultiArray
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener


SOURCE_FRAME = "head"
SCENE_FRAME = "moveit_base_link"
TARGET_FRAMES = (SCENE_FRAME, "SJ", "YB")
BOX_ID = "aruco_box"
TF_PUBLISH_PERIOD_SEC = 0.1
#箱子尺寸
BOX_SIZE_M = (0.593, 0.312, 0.395)
# 抓取点位于箱体局部 +Y 边缘之外。
GRASP_Y_OUTSIDE_M = 0.095 #0.099
GRASP_X_OUTSIDE_M = 0.025 #0.02
# 标定时箱体和左右工具在 moveit_base_link 中的四元数（xyzw）。
REFERENCE_BOX_XYZW = (-0.4911, 0.5049, 0.5147, -0.4888)
GRASP_REFERENCES = (
    ("l_box_grasp", -1, (0.634, -0.308, 0.661, -0.257)),
    ("r_box_grasp", 1, (0.672, 0.243, 0.645, 0.270)),
)
#加的保护尺寸
PADDING_PER_SIDE_M = 0.02
BOX_RPY_DEG = (-3, 0.0, -1.0)  # 箱体相对识别到的二维码，绕局部 X、Y、Z 轴旋转（度）
# 二维码位于 X 中线，距箱体 +Y 边缘 26 cm、+Z 边缘 1.6 cm。
MARKER_TO_CENTER_M = (0.0, 0.260 - BOX_SIZE_M[1] / 2, 0.016 - BOX_SIZE_M[2] / 2)
TIMEOUT_SEC = 3.0
GREEN = "\033[32m"
RESET = "\033[0m"


def make_pose(position, rotation):
    x, y, z = map(float, position)
    qx, qy, qz, qw = rotation.as_quat().tolist()
    return Pose(position=Point(x=x, y=y, z=z), orientation=Quaternion(x=qx, y=qy, z=qz, w=qw))


class BoxNode(Node):
    def __init__(self):
        super().__init__("aruco_box_obstacle")
        self.marker_id = self.declare_parameter("marker_id", 33).value
        self.tag_frame = f"aruco_{self.marker_id}"
        self.tf_buffer = Buffer(node=self)
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.frame_poses = {}
        self.box_rotation_in_tag = Rotation.from_euler("XYZ", BOX_RPY_DEG, degrees=True)
        self.box_center_in_tag = self.box_rotation_in_tag.apply(MARKER_TO_CENTER_M)
        # 将标定工具姿态从基座坐标系转换到箱体坐标系。
        reference_to_box = Rotation.from_quat(REFERENCE_BOX_XYZW).inv()
        grasp_x = BOX_SIZE_M[0] / 2 + GRASP_X_OUTSIDE_M
        grasp_y = BOX_SIZE_M[1] / 2 + GRASP_Y_OUTSIDE_M
        self.grasps_in_box = {
            frame: (
                (side * grasp_x, grasp_y, 0.0),
                reference_to_box * Rotation.from_quat(tool_xyzw),
            )
            for frame, side, tool_xyzw in GRASP_REFERENCES
        }
        self.tf_timer = self.create_timer(TF_PUBLISH_PERIOD_SEC, self.publish_tfs)
        self.scene_client = self.create_client(ApplyPlanningScene, "apply_planning_scene")

    def read_marker(self):
        """Read one valid detection after the command."""
        detection = None

        def on_poses(msg):
            nonlocal detection
            if len(msg.data) % 8:
                return
            # /aruco_poses packs [x, y, z, qx, qy, qz, qw, id] for each tag.
            for start in range(0, len(msg.data), 8):
                *pose, marker_id = msg.data[start:start + 8]
                if marker_id != self.marker_id or not all(map(isfinite, pose)):
                    continue
                try:
                    rotation = Rotation.from_quat(pose[3:])
                except ValueError:
                    continue
                detection = (self.get_clock().now(), pose[:3], rotation)
                break

        subscription = self.create_subscription(Float64MultiArray, "/aruco_poses", on_poses, 1)
        try:
            deadline = monotonic() + TIMEOUT_SEC
            while detection is None and rclpy.ok():
                remaining = deadline - monotonic()
                if remaining <= 0:
                    break
                rclpy.spin_once(self, timeout_sec=remaining)
            return detection
        finally:
            self.destroy_subscription(subscription)

    def box_poses(self, stamp, marker_xyz, marker_rotation):
        poses = {}
        deadline = monotonic() + TIMEOUT_SEC
        for frame in TARGET_FRAMES:
            while rclpy.ok():
                try:
                    transform = self.tf_buffer.lookup_transform(frame, SOURCE_FRAME, stamp).transform
                    break
                except TransformException as exc:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        self.get_logger().error(f"无法获取检测时刻的 TF {frame} ← {SOURCE_FRAME}：{exc}")
                        return None
                    rclpy.spin_once(self, timeout_sec=remaining)
            else:
                return None

            q = transform.rotation
            t = transform.translation
            rotation = Rotation.from_quat([q.x, q.y, q.z, q.w])
            tag_xyz = rotation.apply(marker_xyz) + (t.x, t.y, t.z)
            tag_rotation = rotation * marker_rotation
            center = tag_xyz + tag_rotation.apply(self.box_center_in_tag)
            poses[frame] = (
                make_pose(tag_xyz, tag_rotation),
                make_pose(center, tag_rotation * self.box_rotation_in_tag),
            )
        return poses

    def add_box(self):
        if not self.scene_client.wait_for_service(timeout_sec=TIMEOUT_SEC):
            self.get_logger().error("MoveIt 的 apply_planning_scene 服务不可用")
            return

        detection = self.read_marker()
        if detection is None:
            self.get_logger().error(f"未收到 /aruco_poses 中的新标签 {self.marker_id} 位姿")
            return

        poses = self.box_poses(*detection)
        if poses is None:
            return
        for frame, (marker_pose, pose) in poses.items():
            tag = marker_pose.position
            p = pose.position
            q = pose.orientation
            print(
                f"{GREEN}[{frame}] 标签 {self.marker_id} xyz (m) "
                f"({tag.x:.4f}, {tag.y:.4f}, {tag.z:.4f})，"
                f"箱中心 xyz (m) ({p.x:.4f}, {p.y:.4f}, {p.z:.4f})，"
                f"箱体四元数 xyzw ({q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f}){RESET}"
            )

        if self.apply_box(*poses[SCENE_FRAME]):
            print(f"{GREEN}已更新箱体 {BOX_ID}，发布 TF {SCENE_FRAME} → "
                  f"{', '.join(self.frame_poses)}{RESET}")

    def apply_box(self, marker_pose, box_pose):
        collision = CollisionObject(id=BOX_ID, operation=CollisionObject.ADD)
        collision.header.frame_id = SCENE_FRAME
        collision.pose = marker_pose
        dimensions = [round(size + 2 * PADDING_PER_SIDE_M, 3) for size in BOX_SIZE_M]
        collision.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=dimensions)]
        collision.primitive_poses = [make_pose(self.box_center_in_tag, self.box_rotation_in_tag)]

        if not self.apply_scene(collision):
            return False
        p = box_pose.position
        q = box_pose.orientation
        center = (p.x, p.y, p.z)
        box_rotation = Rotation.from_quat([q.x, q.y, q.z, q.w])
        self.frame_poses = {self.tag_frame: marker_pose}
        for frame, (position, rotation) in self.grasps_in_box.items():
            self.frame_poses[frame] = make_pose(
                center + box_rotation.apply(position), box_rotation * rotation
            )
        self.publish_tfs()
        return True

    def publish_tfs(self):
        if not self.frame_poses:
            return
        stamp = self.get_clock().now().to_msg()
        transforms = []
        for frame, pose in self.frame_poses.items():
            position = pose.position
            transform = TransformStamped(child_frame_id=frame)
            transform.header.frame_id = SCENE_FRAME
            transform.header.stamp = stamp
            transform.transform.translation = Vector3(x=position.x, y=position.y, z=position.z)
            transform.transform.rotation = pose.orientation
            transforms.append(transform)
        self.tf_broadcaster.sendTransform(transforms)

    def clear_scene(self):
        if not self.scene_client.wait_for_service(timeout_sec=TIMEOUT_SEC):
            self.get_logger().error("MoveIt 的 apply_planning_scene 服务不可用")
            return

        # MoveIt interprets REMOVE with an empty ID as clearing all world collision objects.
        if self.apply_scene(CollisionObject(operation=CollisionObject.REMOVE)):
            frames = ", ".join(self.frame_poses)
            self.frame_poses.clear()
            print(f"{GREEN}已清除世界碰撞物体，停止发布箱体相关 TF {frames}{RESET}")

    def apply_scene(self, collision):
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        scene.world.collision_objects = [collision]
        future = self.scene_client.call_async(ApplyPlanningScene.Request(scene=scene))
        rclpy.spin_until_future_complete(self, future, timeout_sec=TIMEOUT_SEC)
        if not future.done():
            future.cancel()
            self.get_logger().error("等待 MoveIt 确认场景更新超时")
        elif future.exception() is not None or not future.result().success:
            self.get_logger().error("MoveIt 场景更新失败")
        else:
            return True
        return False


def main():
    rclpy.init()
    node = BoxNode()
    commands = SimpleQueue()

    def read_commands():
        for line in sys.stdin:
            commands.put(line.strip().lower())
        commands.put("q")

    # Keep the executor running while the terminal is waiting for the next command.
    Thread(target=read_commands, daemon=True).start()
    try:
        while rclpy.ok():
            print("输入 1 单次识别并添加箱体、发布二维码和抓取 TF，0 清除，q 退出：", end="", flush=True)
            while rclpy.ok():
                try:
                    command = commands.get_nowait()
                    break
                except Empty:
                    rclpy.spin_once(node, timeout_sec=TF_PUBLISH_PERIOD_SEC)
            else:
                break
            if command == "1":
                node.add_box()
            elif command == "0":
                node.clear_scene()
            elif command == "q":
                break
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
