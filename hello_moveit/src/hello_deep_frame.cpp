/**
 * @file hello_deep_frame.cpp
 * @brief 向 MoveIt 规划场景添加「深框」碰撞体（仅场景更新，不做运动规划）
 *
 * 使用前提：
 *   1. 已启动 move_group（例如：ros2 launch g01_moveit_config demo.launch.py）
 *   2. 本节点与 move_group 使用同一 ROS 域
 *
 * 编译与运行：
 *   colcon build --packages-select hello_moveit
 *   source install/setup.bash
 *   ros2 run hello_moveit hello_deep_frame
 *
 * 在 RViz 的 Motion Planning 插件中勾选 Scene Geometry / Planning Scene 即可查看深框。
 */

#include <chrono>
#include <memory>
#include <thread>

#include <geometry_msgs/msg/pose.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <rclcpp/rclcpp.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

namespace
{
// G01 左臂规划组；若换机器人请改为 SRDF 中对应的 group 名称
constexpr char kMoveGroupName[] = "left_arm";

// 深框外形尺寸 [m]：长 L × 宽 W × 高 H（开口朝上，无顶盖）
constexpr double kFrameLength = 0.7;
constexpr double kFrameWidth = 0.7;
constexpr double kFrameHeight = 0.6;
// 板厚 [m]：底板和四面墙的厚度
constexpr double kWallThickness = 0.01;

// 深框底面中心在规划坐标系下的位置 [m]（规划坐标系一般为 SRDF virtual_joint 的 parent_frame，G01 为 world）
constexpr double kBaseX = 2;
constexpr double kBaseY = 0.0;
constexpr double kBaseZ = 0.0;

/**
 * @brief 在深框碰撞对象中追加一块长方体（BOX）墙板
 * @param collision_object  待填充的 CollisionObject（可含多块 primitive）
 * @param dx, dy, dz        盒子在 x/y/z 方向的边长 [m]
 * @param x, y, z           盒子中心相对「深框底面中心」(base_x, base_y, base_z) 的偏移 [m]
 */
void addWall(
  moveit_msgs::msg::CollisionObject & collision_object,
  double dx, double dy, double dz, double x, double y, double z)
{
  shape_msgs::msg::SolidPrimitive primitive;
  primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
  primitive.dimensions = {dx, dy, dz};

  geometry_msgs::msg::Pose pose;
  pose.orientation.w = 1.0;  // 无旋转，墙板轴与规划坐标系平行
  pose.position.x = kBaseX + x;
  pose.position.y = kBaseY + y;
  pose.position.z = kBaseZ + z;

  collision_object.primitives.push_back(primitive);
  collision_object.primitive_poses.push_back(pose);
}

/**
 * @brief 构造「深框」CollisionObject：底板 + 四面侧墙，顶部敞开
 * @param frame_id  碰撞体所在的坐标系（应使用 MoveGroup 的 planning frame，如 world）
 */
moveit_msgs::msg::CollisionObject makeDeepFrameCollisionObject(const std::string & frame_id)
{
  const double L = kFrameLength;
  const double W = kFrameWidth;
  const double H = kFrameHeight;
  const double tb = kWallThickness;

  moveit_msgs::msg::CollisionObject collision_object;
  collision_object.header.frame_id = frame_id;
  collision_object.id = "深框";

  // 底板：铺在 z = base_z 上，中心抬高 tb/2
  addWall(collision_object, L, W, tb, 0.0, 0.0, tb / 2.0);

  // 前墙（+X 侧）：中心在 x = +(L/2 - tb/2)
  addWall(collision_object, tb, W, H, L / 2.0 - tb / 2.0, 0.0, H / 2.0);
  // 后墙（-X 侧）
  addWall(collision_object, tb, W, H, -(L / 2.0 - tb / 2.0), 0.0, H / 2.0);
  // 左墙（+Y 侧）：长度略短，避免与前后墙角重叠
  addWall(collision_object, L - 2.0 * tb, tb, H, 0.0, W / 2.0 - tb / 2.0, H / 2.0);
  // 右墙（-Y 侧）
  addWall(collision_object, L - 2.0 * tb, tb, H, 0.0, -(W / 2.0 - tb / 2.0), H / 2.0);

  collision_object.operation = moveit_msgs::msg::CollisionObject::ADD;
  return collision_object;
}

}  // namespace

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  // 允许通过 launch 或命令行覆盖参数（如 move_group 命名空间等）
  auto const node = std::make_shared<rclcpp::Node>(
    "hello_deep_frame",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  auto const logger = rclcpp::get_logger("hello_deep_frame");

  // MoveGroupInterface 仅用于查询规划坐标系名称，不参与规划
  using moveit::planning_interface::MoveGroupInterface;
  MoveGroupInterface move_group(node, kMoveGroupName);

  const std::string planning_frame = move_group.getPlanningFrame();
  RCLCPP_INFO(
    logger, "规划组: %s, 规划坐标系: %s", kMoveGroupName, planning_frame.c_str());

  moveit_msgs::msg::CollisionObject deep_frame =
    makeDeepFrameCollisionObject(planning_frame);

  // PlanningSceneInterface 通过 move_group 的 planning scene 服务把碰撞体写入场景
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

  RCLCPP_INFO(
    logger,
    "正在向规划场景添加碰撞体「%s」（%zu 块 BOX）…",
    deep_frame.id.c_str(), deep_frame.primitives.size());

  const bool applied = planning_scene_interface.applyCollisionObject(deep_frame);

  if (applied)
  {
    RCLCPP_INFO(
      logger,
      "深框已添加。尺寸: %.2f × %.2f × %.2f m, 板厚 %.3f m, 底面中心 (%.2f, %.2f, %.2f)",
      kFrameLength, kFrameWidth, kFrameHeight, kWallThickness, kBaseX, kBaseY, kBaseZ);
  }
  else
  {
    RCLCPP_ERROR(
      logger,
      "添加深框失败。请确认 move_group 已启动且命名空间与参数一致。");
  }

  // 短暂 spin，确保服务调用与话题发布完成后再退出
  rclcpp::sleep_for(std::chrono::milliseconds(500));
  rclcpp::shutdown();
  return applied ? 0 : 1;
}
