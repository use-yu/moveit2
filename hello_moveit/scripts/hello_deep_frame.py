#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向 MoveIt 规划场景添加半透明「深框」碰撞体（仅场景更新，不做运动规划）。

使用前提：
  1. 已启动 move_group（例如：ros2 launch g01_moveit_config demo.launch.py）
  2. 本节点与 move_group 在同一 ROS 域

编译与运行：
  colcon build --packages-select hello_moveit
  source install/setup.bash
  ros2 run hello_moveit hello_deep_frame.py

在 RViz 的 Motion Planning 插件中勾选 Scene Geometry / Planning Scene 即可查看深框。
按回车键退出时自动从规划场景中移除深框。

说明：障碍物写入全局 Planning Scene，对 left_arm、left、left_body 等所有规划组均生效。
"""

from __future__ import annotations

import sys

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, ObjectColor, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import ColorRGBA

# G01 规划坐标系（与 SRDF virtual_joint 的 parent_frame 一致）
PLANNING_FRAME = "world"

# 深框外形尺寸 [m]：长 L × 宽 W × 高 H（开口朝上，无顶盖）
FRAME_LENGTH = 0.8
FRAME_WIDTH = 0.8
FRAME_HEIGHT = 0.7
# 板厚 [m]
WALL_THICKNESS = 0.02

# 深框底面中心在规划坐标系下的位置 [m]
BASE_X = 2.0
BASE_Y = 0.0
BASE_Z = 0.0

COLLISION_OBJECT_ID = "深框"

# RViz 显示用半透明颜色（alpha 越小越透明；碰撞检测仍按几何体计算）
DISPLAY_COLOR = ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.5)

APPLY_PLANNING_SCENE_SERVICE = "apply_planning_scene"


def _make_pose(x: float, y: float, z: float) -> Pose:
    pose = Pose()
    pose.orientation.w = 1.0
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    return pose


def _add_wall(
    collision_object: CollisionObject,
    dx: float,
    dy: float,
    dz: float,
    x: float,
    y: float,
    z: float,
) -> None:
    """在深框碰撞对象中追加一块 BOX 墙板。"""
    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = [dx, dy, dz]

    pose = _make_pose(BASE_X + x, BASE_Y + y, BASE_Z + z)
    collision_object.primitives.append(primitive)
    collision_object.primitive_poses.append(pose)


def make_deep_frame_collision_object(frame_id: str) -> CollisionObject:
    """构造深框 CollisionObject：底板 + 四面侧墙，顶部敞开。"""
    L = FRAME_LENGTH
    W = FRAME_WIDTH
    H = FRAME_HEIGHT
    tb = WALL_THICKNESS

    obj = CollisionObject()
    obj.header.frame_id = frame_id
    obj.id = COLLISION_OBJECT_ID

    _add_wall(obj, L, W, tb, 0.0, 0.0, tb / 2.0)
    _add_wall(obj, tb, W, H, L / 2.0 - tb / 2.0, 0.0, H / 2.0)
    _add_wall(obj, tb, W, H, -(L / 2.0 - tb / 2.0), 0.0, H / 2.0)
    _add_wall(obj, L - 2.0 * tb, tb, H, 0.0, W / 2.0 - tb / 2.0, H / 2.0)
    _add_wall(obj, L - 2.0 * tb, tb, H, 0.0, -(W / 2.0 - tb / 2.0), H / 2.0)

    obj.operation = CollisionObject.ADD
    return obj


def make_remove_collision_object() -> CollisionObject:
    """构造用于从场景中删除深框的 CollisionObject。"""
    obj = CollisionObject()
    obj.id = COLLISION_OBJECT_ID
    obj.operation = CollisionObject.REMOVE
    return obj


def make_display_object_color() -> ObjectColor:
    """RViz 中深框的半透明显示颜色。"""
    oc = ObjectColor()
    oc.id = COLLISION_OBJECT_ID
    oc.color = DISPLAY_COLOR
    return oc


class HelloDeepFrame(Node):
    """通过 apply_planning_scene 服务向 move_group 写入/删除碰撞体。"""

    def __init__(self) -> None:
        super().__init__("hello_deep_frame")
        self._apply_scene_client = self.create_client(
            ApplyPlanningScene, APPLY_PLANNING_SCENE_SERVICE
        )

    def _apply_planning_scene(
        self,
        collision_objects: list[CollisionObject],
        object_colors: list[ObjectColor] | None = None,
    ) -> bool:
        if not self._apply_scene_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(
                f"服务 {APPLY_PLANNING_SCENE_SERVICE} 不可用，请先启动 move_group。"
            )
            return False

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.world.collision_objects.extend(collision_objects)
        if object_colors:
            scene.object_colors.extend(object_colors)

        request = ApplyPlanningScene.Request()
        request.scene = scene
        future = self._apply_scene_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if not future.done() or future.result() is None:
            self.get_logger().error("调用 apply_planning_scene 超时或失败。")
            return False
        if not future.result().success:
            self.get_logger().error("apply_planning_scene 返回 success=False。")
            return False
        return True

    def add_deep_frame(self, collision_object: CollisionObject) -> bool:
        return self._apply_planning_scene(
            [collision_object], [make_display_object_color()]
        )

    def remove_deep_frame(self) -> bool:
        return self._apply_planning_scene([make_remove_collision_object()])


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node = HelloDeepFrame()
    logger = node.get_logger()
    exit_code = 1
    added = False

    try:
        logger.info(f"规划坐标系: {PLANNING_FRAME}")

        deep_frame = make_deep_frame_collision_object(PLANNING_FRAME)
        logger.info(
            f"正在向规划场景添加半透明碰撞体「{COLLISION_OBJECT_ID}」"
            f"（{len(deep_frame.primitives)} 块 BOX，alpha={DISPLAY_COLOR.a}）…"
        )

        if not node.add_deep_frame(deep_frame):
            logger.error("添加深框失败。请确认 move_group 已启动。")
            return 1

        added = True
        logger.info(
            f"深框已添加。尺寸: {FRAME_LENGTH:.2f} × {FRAME_WIDTH:.2f} × {FRAME_HEIGHT:.2f} m, "
            f"板厚 {WALL_THICKNESS:.3f} m, 底面中心 ({BASE_X:.2f}, {BASE_Y:.2f}, {BASE_Z:.2f})"
        )
        logger.info("在 RViz 中可规划避障路径。按回车键结束并移除深框…")
        try:
            input()
        except EOFError:
            pass

        exit_code = 0
    finally:
        if added:
            logger.info(f"正在移除碰撞体「{COLLISION_OBJECT_ID}」…")
            if node.remove_deep_frame():
                logger.info("深框已从规划场景中移除。")
            else:
                logger.error("移除深框失败。")
                exit_code = 1
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
