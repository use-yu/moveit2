// colcon build --packages-select hello_moveit

#include <chrono>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

namespace
{
constexpr char kPlanningGroup[] = "body";  // 快速切换: "body" | "dual_arm"

constexpr unsigned int kParallelPlanningAttempts = 100;

constexpr char kPlanningFrame[] = "world";
constexpr char kCollisionObjectId[] = "深框";
constexpr double kFrameLength = 0.8;
constexpr double kFrameWidth = 0.8;
constexpr double kFrameHeight = 0.7;
constexpr double kWallThickness = 0.02;
constexpr double kFrameBaseX = 2.5;
constexpr double kFrameBaseY = 0.0;
constexpr double kFrameBaseZ = 0.0;

using JointMap = std::map<std::string, double>;

const std::map<std::string, JointMap> kConfigs{
  {"body",
   {
     {"base_joint1", 1.0},
     {"base_joint2", 0.0},
     {"body_joint1", -0.1},
     {"body_joint2", 0.6},
   }},
  {"dual_arm",
   {
     {"base_joint1", 0.2},
     {"base_joint2", 0.0},
     {"body_joint1", -0.1},
     {"body_joint2", 1.1},
     {"l_arm_joint1", 0.0},
     {"l_arm_joint2", 0.0},
     {"l_arm_joint3", 0.0},
     {"l_arm_joint4", 0.0},
     {"l_arm_joint5", 0.0},
     {"l_arm_joint6", 0.0},
     {"r_arm_joint1", 0.0},
     {"r_arm_joint2", 0.0},
     {"r_arm_joint3", 0.0},
     {"r_arm_joint4", 0.0},
     {"r_arm_joint5", 0.0},
     {"r_arm_joint6", 0.0},
   }},
};

JointMap filterToGroup(const moveit::core::JointModelGroup * jmg, const JointMap & joints)
{
  JointMap filtered;
  for (const std::string & name : jmg->getVariableNames())
  {
    auto it = joints.find(name);
    if (it != joints.end())
    {
      filtered[name] = it->second;
    }
  }
  return filtered;
}

std::optional<JointMap> readStartFromJointStates(
  const rclcpp::Node::SharedPtr & node,
  const moveit::core::JointModelGroup * jmg,
  double timeout_sec = 10.0)
{
  sensor_msgs::msg::JointState::SharedPtr latest;
  std::mutex mtx;
  std::condition_variable cv;
  bool received = false;

  auto sub = node->create_subscription<sensor_msgs::msg::JointState>(
    "joint_states", rclcpp::SensorDataQoS(),
    [&](const sensor_msgs::msg::JointState::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(mtx);
      latest = msg;
      received = true;
      cv.notify_one();
    });

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_sec);
  {
    std::unique_lock<std::mutex> lock(mtx);
    while (!received && std::chrono::steady_clock::now() < deadline)
    {
      cv.wait_until(lock, deadline);
    }
  }

  if (!latest)
  {
    RCLCPP_ERROR(node->get_logger(), "Timeout waiting for /joint_states (%.1f s)", timeout_sec);
    return std::nullopt;
  }

  JointMap all_positions;
  for (size_t i = 0; i < latest->name.size(); ++i)
  {
    if (i < latest->position.size())
    {
      all_positions[latest->name[i]] = latest->position[i];
    }
  }

  JointMap start = filterToGroup(jmg, all_positions);
  const auto expected = jmg->getVariableCount();
  if (start.size() != expected)
  {
    RCLCPP_ERROR(
      node->get_logger(),
      "joint_states missing planning-group joints: got %zu / %u",
      start.size(), expected);
    for (const std::string & name : jmg->getVariableNames())
    {
      if (start.find(name) == start.end())
      {
        RCLCPP_ERROR(node->get_logger(), "  missing: %s", name.c_str());
      }
    }
    return std::nullopt;
  }

  for (const auto & [name, pos] : start)
  {
    RCLCPP_INFO(node->get_logger(), "  start %s = %.4f", name.c_str(), pos);
  }
  return start;
}

moveit_msgs::msg::CollisionObject makeDeepFrameCollisionObject()
{
  const double L = kFrameLength;
  const double W = kFrameWidth;
  const double H = kFrameHeight;
  const double tb = kWallThickness;

  moveit_msgs::msg::CollisionObject obj;
  obj.header.frame_id = kPlanningFrame;
  obj.id = kCollisionObjectId;

  auto add_wall = [&](double dx, double dy, double dz, double x, double y, double z) {
    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    primitive.dimensions = {dx, dy, dz};
    geometry_msgs::msg::Pose pose;
    pose.orientation.w = 1.0;
    pose.position.x = kFrameBaseX + x;
    pose.position.y = kFrameBaseY + y;
    pose.position.z = kFrameBaseZ + z;
    obj.primitives.push_back(primitive);
    obj.primitive_poses.push_back(pose);
  };

  add_wall(L, W, tb, 0.0, 0.0, tb / 2.0);
  add_wall(tb, W, H, L / 2.0 - tb / 2.0, 0.0, H / 2.0);
  add_wall(tb, W, H, -(L / 2.0 - tb / 2.0), 0.0, H / 2.0);
  add_wall(L - 2.0 * tb, tb, H, 0.0, W / 2.0 - tb / 2.0, H / 2.0);
  add_wall(L - 2.0 * tb, tb, H, 0.0, -(W / 2.0 - tb / 2.0), H / 2.0);

  obj.operation = obj.ADD;
  return obj;
}

void removeDeepFrame(moveit::planning_interface::PlanningSceneInterface & psi)
{
  moveit_msgs::msg::CollisionObject obj;
  obj.id = kCollisionObjectId;
  obj.operation = obj.REMOVE;
  psi.applyCollisionObject(obj);
}

}  // namespace

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto const node = std::make_shared<rclcpp::Node>(
    "hello_go1",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  auto const logger = rclcpp::get_logger("hello_go1");

  auto cfg_it = kConfigs.find(kPlanningGroup);
  if (cfg_it == kConfigs.end())
  {
    RCLCPP_ERROR(logger, "Unknown group '%s'. Available: body, dual_arm", kPlanningGroup);
    rclcpp::shutdown();
    return 1;
  }
  const JointMap & goal_joints = cfg_it->second;

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  auto spinner = std::thread([&executor]() { executor.spin(); });

  using moveit::planning_interface::MoveGroupInterface;
  using moveit::planning_interface::PlanningSceneInterface;
  MoveGroupInterface move_group(node, kPlanningGroup);
  PlanningSceneInterface planning_scene;

  auto const jmg = move_group.getRobotModel()->getJointModelGroup(kPlanningGroup);
  if (!jmg)
  {
    RCLCPP_ERROR(logger, "Planning group '%s' not found.", kPlanningGroup);
    rclcpp::shutdown();
    spinner.join();
    return 1;
  }

  move_group.setMaxVelocityScalingFactor(0.5);
  move_group.setMaxAccelerationScalingFactor(0.5);
  move_group.setPlanningTime(30.0);
  move_group.setPlannerId("RRTConnect");
  move_group.setNumPlanningAttempts(kParallelPlanningAttempts);

  const auto deep_frame = makeDeepFrameCollisionObject();
  planning_scene.applyCollisionObject(deep_frame);
  RCLCPP_INFO(
    logger, "Added deep frame at (%.1f, %.1f, %.1f), size %.1f×%.1f×%.1f m",
    kFrameBaseX, kFrameBaseY, kFrameBaseZ, kFrameLength, kFrameWidth, kFrameHeight);

  const auto start_opt = readStartFromJointStates(node, jmg);
  if (!start_opt)
  {
    removeDeepFrame(planning_scene);
    rclcpp::shutdown();
    spinner.join();
    return 1;
  }
  const JointMap & start = *start_opt;
  const JointMap goal = filterToGroup(jmg, goal_joints);

  moveit::core::RobotState start_state(*move_group.getCurrentState());
  start_state.setVariablePositions(start);
  move_group.setStartState(start_state);
  move_group.setJointValueTarget(goal);

  RCLCPP_INFO(
    logger, "Planning group: %s (%u joints)", kPlanningGroup, jmg->getVariableCount());
  RCLCPP_INFO(
    logger, "Planning start -> goal (joint space, %u parallel attempts)...",
    kParallelPlanningAttempts);

  MoveGroupInterface::Plan plan;
  const auto plan_start = std::chrono::steady_clock::now();
  const bool plan_ok = static_cast<bool>(move_group.plan(plan));
  const auto plan_end = std::chrono::steady_clock::now();
  const double plan_ms =
    std::chrono::duration<double, std::milli>(plan_end - plan_start).count();

  RCLCPP_INFO(logger, "Planning time: %.3f ms (%s)", plan_ms, plan_ok ? "success" : "failed");

  int exit_code = 1;
  if (!plan_ok)
  {
    RCLCPP_ERROR(logger, "Planning failed. See move_group terminal for details.");
  }
  else
  {
    const auto exec_start = std::chrono::steady_clock::now();
    const bool exec_ok = static_cast<bool>(move_group.execute(plan));
    const auto exec_end = std::chrono::steady_clock::now();
    const double exec_ms =
      std::chrono::duration<double, std::milli>(exec_end - exec_start).count();

    RCLCPP_INFO(logger, "Execution time: %.3f ms (%s)", exec_ms, exec_ok ? "success" : "failed");
    RCLCPP_INFO(logger, "Total time: %.3f ms", plan_ms + exec_ms);
    exit_code = exec_ok ? 0 : 1;
  }

  removeDeepFrame(planning_scene);
  RCLCPP_INFO(logger, "Removed deep frame from planning scene.");

  rclcpp::shutdown();
  spinner.join();
  return exit_code;
}
