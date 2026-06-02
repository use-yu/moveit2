// =============================================================================
// G01 MoveIt 演示（C++，对应 scripts/g01.py）
// =============================================================================
//
// 编译与运行：
//   colcon build --packages-select hello_moveit
//   source install/setup.bash
//   ros2 launch g01_moveit_config demo.launch.py    # 另开终端
//   ros2 run hello_moveit hello_g01
//
// 流程：
//   1. 向规划场景添加「深框」碰撞体（PlanningSceneInterface）
//   2. dual_arm 多点关节路径：waypoints.push_back(...) 后分段规划、拼接、一次执行
//   3. left_body 末端位姿：setPoseTarget + plan + execute（参考 hello_moveit.cpp）
//   4. 退出前移除深框
//
// =============================================================================

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Geometry>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

namespace
{

constexpr char kJointGroup[] = "dual_arm";
constexpr char kPoseGroup[] = "left_body";
constexpr char kEndEffectorLink[] = "L6";

constexpr char kPlannerId[] = "RRTConnect";
constexpr unsigned int kJointPlanningAttempts = 100;
constexpr unsigned int kPosePlanningAttempts = 40;
constexpr double kJointPlanningTimeSec = 20.0;
constexpr double kPosePlanningTimeSec = 20.0;
constexpr double kVelocityScale = 0.5;
constexpr double kAccelerationScale = 0.5;

constexpr double deg2rad(double deg)
{
  return deg * M_PI / 180.0;
}

// 预备位（与 g01.py / hello_go1.cpp dual_arm 一致）
const std::map<std::string, double> kDualArmGoalRaw{
  {"base_joint1", 1.25},
  {"base_joint2", 0.0},
  {"body_joint1", -0.25},
  {"body_joint2", 1.1},
  {"l_arm_joint1", deg2rad(-20.0)},
  {"l_arm_joint2", deg2rad(-102.0)},
  {"l_arm_joint3", deg2rad(-92.0)},
  {"l_arm_joint4", deg2rad(137.0)},
  {"l_arm_joint5", 0.0},
  {"l_arm_joint6", 0.0},
  {"r_arm_joint1", deg2rad(-20.0)},
  {"r_arm_joint2", deg2rad(-102.0)},
  {"r_arm_joint3", deg2rad(-92.0)},
  {"r_arm_joint4", deg2rad(137.0)},
  {"r_arm_joint5", 0.0},
  {"r_arm_joint6", 0.0},
};

// 第二路径点示例（可按需继续 push_back）
const std::map<std::string, double> kDualArmGoalRaw2{
  {"base_joint1", 1.25},
  {"base_joint2", 0.0},
  {"body_joint1", 0.0},
  {"body_joint2", deg2rad(73.0)},
  {"l_arm_joint1", deg2rad(95.0)},
  {"l_arm_joint2", deg2rad(11.0)},
  {"l_arm_joint3", deg2rad(9.0)},
  {"l_arm_joint4", deg2rad(70.0)},
  {"l_arm_joint5", deg2rad(113.0)},
  {"l_arm_joint6", deg2rad(-90.0)},
  
  {"r_arm_joint1", deg2rad(80.0)},
  {"r_arm_joint2", deg2rad(-102.0)},
  {"r_arm_joint3", deg2rad(-92.0)},
  {"r_arm_joint4", deg2rad(137.0)},
  {"r_arm_joint5", 0.0},
  {"r_arm_joint6", 0.0},
};

constexpr double kEeX = -0.75;
constexpr double kEeY = -0.35;
constexpr double kEeZ = 0.0;
constexpr double kEeRoll = -M_PI / 2.0;
constexpr double kEePitch = -M_PI / 2.0;
constexpr double kEeYaw = -M_PI;

constexpr char kSceneFrame[] = "world";
constexpr char kCollisionObjectId[] = "深框";
constexpr double kFrameLength = 0.8;
constexpr double kFrameWidth = 0.8;
constexpr double kFrameHeight = 0.7;
constexpr double kWallThickness = 0.02;
constexpr double kFrameBaseX = 2.0;
constexpr double kFrameBaseY = 0.0;
constexpr double kFrameBaseZ = 0.0;
// RViz 中深框半透明显示（与 g01.py DISPLAY_COLOR 一致；碰撞仍由 CollisionObject 定义）
constexpr float kFrameColorR = 0.2f;
constexpr float kFrameColorG = 0.6f;
constexpr float kFrameColorB = 1.0f;
constexpr float kFrameColorA = 0.5f;

using JointMap = std::map<std::string, double>;

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

JointMap jointStateMsgToMap(const sensor_msgs::msg::JointState & msg)
{
  JointMap out;
  for (size_t i = 0; i < msg.name.size(); ++i)
  {
    if (i < msg.position.size())
    {
      out[msg.name[i]] = msg.position[i];
    }
  }
  return out;
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

  const auto deadline =
    std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_sec);
  {
    std::unique_lock<std::mutex> lock(mtx);
    while (!received && std::chrono::steady_clock::now() < deadline)
    {
      cv.wait_until(lock, deadline);
    }
  }

  if (!latest)
  {
    RCLCPP_ERROR(node->get_logger(), "等待 /joint_states 超时 (%.1f s)", timeout_sec);
    return std::nullopt;
  }

  JointMap start = filterToGroup(jmg, jointStateMsgToMap(*latest));
  if (start.size() != jmg->getVariableCount())
  {
    RCLCPP_ERROR(
      node->get_logger(), "joint_states 缺少规划组关节: %zu / %u", start.size(),
      jmg->getVariableCount());
    return std::nullopt;
  }
  return start;
}

// 丢弃订阅建立后的第一帧，等待运动后的新 joint_states（对应 g01.py wait_new）
std::optional<JointMap> readFreshJointStates(
  const rclcpp::Node::SharedPtr & node,
  const moveit::core::JointModelGroup * jmg,
  double timeout_sec = 10.0)
{
  struct State
  {
    sensor_msgs::msg::JointState::SharedPtr baseline;
    sensor_msgs::msg::JointState::SharedPtr fresh;
    bool got_baseline{false};
    bool got_fresh{false};
  } state;
  std::mutex mtx;
  std::condition_variable cv;

  auto sub = node->create_subscription<sensor_msgs::msg::JointState>(
    "joint_states", rclcpp::SensorDataQoS(),
    [&](const sensor_msgs::msg::JointState::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(mtx);
      if (!state.got_baseline)
      {
        state.baseline = msg;
        state.got_baseline = true;
      }
      else
      {
        state.fresh = msg;
        state.got_fresh = true;
        cv.notify_one();
      }
    });

  const auto deadline =
    std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_sec);
  {
    std::unique_lock<std::mutex> lock(mtx);
    while (!state.got_fresh && std::chrono::steady_clock::now() < deadline)
    {
      cv.wait_until(lock, deadline);
    }
  }

  sensor_msgs::msg::JointState::SharedPtr use_msg = state.fresh;
  if (!use_msg)
  {
    RCLCPP_WARN(node->get_logger(), "%.1f s 内未收到新 joint_states，使用基线帧", timeout_sec);
    use_msg = state.baseline;
  }
  if (!use_msg)
  {
    return std::nullopt;
  }

  JointMap start = filterToGroup(jmg, jointStateMsgToMap(*use_msg));
  if (start.size() != jmg->getVariableCount())
  {
    return std::nullopt;
  }
  for (const auto & [name, pos] : start)
  {
    RCLCPP_INFO(node->get_logger(), "  pose start %s = %.4f", name.c_str(), pos);
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
  obj.header.frame_id = kSceneFrame;
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

std_msgs::msg::ColorRGBA makeDeepFrameDisplayColor()
{
  std_msgs::msg::ColorRGBA color;
  color.r = kFrameColorR;
  color.g = kFrameColorG;
  color.b = kFrameColorB;
  color.a = kFrameColorA;
  return color;
}

void removeDeepFrame(moveit::planning_interface::PlanningSceneInterface & psi)
{
  moveit_msgs::msg::CollisionObject obj;
  obj.id = kCollisionObjectId;
  obj.operation = obj.REMOVE;
  psi.applyCollisionObject(obj);
}

geometry_msgs::msg::PoseStamped makeEeGoalPoseStamped(
  const rclcpp::Node::SharedPtr & node, const std::string & frame_id)
{
  geometry_msgs::msg::PoseStamped stamped;
  stamped.header.frame_id = frame_id;
  stamped.header.stamp = node->now();
  stamped.pose.position.x = kEeX;
  stamped.pose.position.y = kEeY;
  stamped.pose.position.z = kEeZ;

  const Eigen::Quaterniond q =
    Eigen::AngleAxisd(kEeYaw, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(kEePitch, Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(kEeRoll, Eigen::Vector3d::UnitX());
  stamped.pose.orientation.x = q.x();
  stamped.pose.orientation.y = q.y();
  stamped.pose.orientation.z = q.z();
  stamped.pose.orientation.w = q.w();
  return stamped;
}

void configureMoveGroup(
  moveit::planning_interface::MoveGroupInterface & mg,
  const double planning_time_sec,
  const unsigned int num_attempts)
{
  mg.setMaxVelocityScalingFactor(kVelocityScale);
  mg.setMaxAccelerationScalingFactor(kAccelerationScale);
  mg.setPlanningTime(planning_time_sec);
  mg.setPlannerId(kPlannerId);
  mg.setNumPlanningAttempts(num_attempts);
}

bool jointMapsNearEqual(
  const moveit::core::RobotState & state, const moveit::core::JointModelGroup * jmg,
  const JointMap & goal, double eps = 1e-4)
{
  for (const std::string & name : jmg->getVariableNames())
  {
    auto it = goal.find(name);
    if (it == goal.end())
    {
      return false;
    }
    if (std::abs(state.getVariablePosition(name) - it->second) > eps)
    {
      return false;
    }
  }
  return true;
}

// 多点关节路径：逐段 plan，拼接为一条轨迹后一次 execute
bool planAndExecuteJointWaypoints(
  const rclcpp::Logger & logger,
  moveit::planning_interface::MoveGroupInterface & mg,
  const moveit::core::JointModelGroup * jmg,
  const JointMap & start_joints,
  const std::vector<JointMap> & waypoints,
  const char * label)
{
  if (waypoints.empty())
  {
    RCLCPP_ERROR(logger, "%s: 路径点列表为空", label);
    return false;
  }

  moveit::core::RobotState segment_start(*mg.getCurrentState());
  segment_start.setVariablePositions(start_joints);

  robot_trajectory::RobotTrajectory combined(mg.getRobotModel(), jmg);
  double total_plan_ms = 0.0;

  for (size_t i = 0; i < waypoints.size(); ++i)
  {
    if (jointMapsNearEqual(segment_start, jmg, waypoints[i]))
    {
      RCLCPP_WARN(
        logger, "%s segment %zu/%zu: 起点与目标相同，跳过规划", label, i + 1, waypoints.size());
      continue;
    }

    mg.setStartState(segment_start);
    if (!mg.setJointValueTarget(waypoints[i]))
    {
      RCLCPP_ERROR(logger, "%s waypoint %zu: setJointValueTarget 失败", label, i + 1);
      return false;
    }

    moveit::planning_interface::MoveGroupInterface::Plan segment_plan;
    const auto plan_start = std::chrono::steady_clock::now();
    const bool plan_ok = static_cast<bool>(mg.plan(segment_plan));
    const auto plan_end = std::chrono::steady_clock::now();
    const double plan_ms =
      std::chrono::duration<double, std::milli>(plan_end - plan_start).count();
    total_plan_ms += plan_ms;

    RCLCPP_INFO(
      logger, "%s segment %zu/%zu planning: %.3f ms (%s)", label, i + 1, waypoints.size(), plan_ms,
      plan_ok ? "success" : "failed");

    if (!plan_ok)
    {
      RCLCPP_ERROR(logger, "%s segment %zu/%zu planning failed.", label, i + 1, waypoints.size());
      return false;
    }

    robot_trajectory::RobotTrajectory segment_traj(mg.getRobotModel(), jmg);
    segment_traj.setRobotTrajectoryMsg(segment_start, segment_plan.trajectory_);

    if (combined.getWayPointCount() == 0)
    {
      combined = segment_traj;
    }
    else
    {
      const double bridge_dt =
        combined.getWayPointDurationFromPrevious(combined.getWayPointCount() - 1);
      combined.append(segment_traj, bridge_dt, 1);
    }

    segment_start = segment_traj.getLastWayPoint();
  }

  if (combined.getWayPointCount() == 0)
  {
    RCLCPP_ERROR(logger, "%s: 无有效轨迹（路径点均与起点相同？）", label);
    return false;
  }

  moveit::planning_interface::MoveGroupInterface::Plan merged;
  moveit::core::robotStateToRobotStateMsg(segment_start, merged.start_state_);
  combined.getRobotTrajectoryMsg(merged.trajectory_);
  merged.planning_time_ = total_plan_ms / 1000.0;

  RCLCPP_INFO(
    logger, "%s merged trajectory: %zu waypoints, planning total: %.3f ms", label,
    merged.trajectory_.joint_trajectory.points.size(), total_plan_ms);

  const auto exec_start = std::chrono::steady_clock::now();
  const bool exec_ok = static_cast<bool>(mg.execute(merged));
  const auto exec_end = std::chrono::steady_clock::now();
  const double exec_ms =
    std::chrono::duration<double, std::milli>(exec_end - exec_start).count();

  RCLCPP_INFO(
    logger, "%s execution time: %.3f ms (%s)", label, exec_ms, exec_ok ? "success" : "failed");
  RCLCPP_INFO(logger, "%s total time: %.3f ms", label, total_plan_ms + exec_ms);
  return exec_ok;
}

bool planAndExecute(
  const rclcpp::Logger & logger,
  moveit::planning_interface::MoveGroupInterface & mg,
  const char * label)
{
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  const auto plan_start = std::chrono::steady_clock::now();
  const bool plan_ok = static_cast<bool>(mg.plan(plan));
  const auto plan_end = std::chrono::steady_clock::now();
  const double plan_ms =
    std::chrono::duration<double, std::milli>(plan_end - plan_start).count();

  RCLCPP_INFO(
    logger, "%s planning time: %.3f ms (%s)", label, plan_ms, plan_ok ? "success" : "failed");

  if (!plan_ok)
  {
    RCLCPP_ERROR(logger, "%s planning failed. See move_group terminal.", label);
    return false;
  }

  const auto exec_start = std::chrono::steady_clock::now();
  const bool exec_ok = static_cast<bool>(mg.execute(plan));
  const auto exec_end = std::chrono::steady_clock::now();
  const double exec_ms =
    std::chrono::duration<double, std::milli>(exec_end - exec_start).count();

  RCLCPP_INFO(
    logger, "%s execution time: %.3f ms (%s)", label, exec_ms, exec_ok ? "success" : "failed");
  RCLCPP_INFO(logger, "%s total time: %.3f ms", label, plan_ms + exec_ms);
  return exec_ok;
}

}  // namespace

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto const node = std::make_shared<rclcpp::Node>(
    "hello_g01",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
  auto const logger = rclcpp::get_logger("hello_g01");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  auto spinner = std::thread([&executor]() { executor.spin(); });

  using moveit::planning_interface::MoveGroupInterface;
  using moveit::planning_interface::PlanningSceneInterface;
  PlanningSceneInterface planning_scene;

  int exit_code = 1;

  // ---------------------------------------------------------------------------
  // 1. 添加深框
  // ---------------------------------------------------------------------------
  planning_scene.applyCollisionObject(
    makeDeepFrameCollisionObject(), makeDeepFrameDisplayColor());
  RCLCPP_INFO(
    logger,
    "已添加半透明深框 @ %s (%.1f, %.1f, %.1f), %.1f×%.1f×%.1f m, alpha=%.2f",
    kSceneFrame, kFrameBaseX, kFrameBaseY, kFrameBaseZ, kFrameLength, kFrameWidth, kFrameHeight,
    kFrameColorA);

  // ---------------------------------------------------------------------------
  // 2. dual_arm 多点关节路径（waypoints.push_back 添加路径点）
  // ---------------------------------------------------------------------------
  {
    MoveGroupInterface joint_mg(node, kJointGroup);
    auto const jmg = joint_mg.getRobotModel()->getJointModelGroup(kJointGroup);
    if (!jmg)
    {
      RCLCPP_ERROR(logger, "规划组 '%s' 不存在", kJointGroup);
      removeDeepFrame(planning_scene);
      rclcpp::shutdown();
      spinner.join();
      return 1;
    }

    configureMoveGroup(joint_mg, kJointPlanningTimeSec, kJointPlanningAttempts);

    const auto start_opt = readStartFromJointStates(node, jmg);
    if (!start_opt)
    {
      removeDeepFrame(planning_scene);
      rclcpp::shutdown();
      spinner.join();
      return 1;
    }

    std::vector<JointMap> waypoints;
    waypoints.push_back(filterToGroup(jmg, kDualArmGoalRaw));
    waypoints.push_back(filterToGroup(jmg, kDualArmGoalRaw2));
    // waypoints.push_back(filterToGroup(jmg, kYourNextGoal));

    RCLCPP_INFO(
      logger, "[%s] 多点关节路径 (%zu waypoints, %u joints, %u attempts, %.1fs)", kJointGroup,
      waypoints.size(), jmg->getVariableCount(), kJointPlanningAttempts, kJointPlanningTimeSec);

    if (!planAndExecuteJointWaypoints(logger, joint_mg, jmg, *start_opt, waypoints, kJointGroup))
    {
      removeDeepFrame(planning_scene);
      rclcpp::shutdown();
      spinner.join();
      return 1;
    }
  }
  // ---------------------------------------------------------------------------
  // 3. left_body 末端位姿（hello_moveit：setPoseTarget）
  // ---------------------------------------------------------------------------
  {
    MoveGroupInterface pose_mg(node, kPoseGroup);
    auto const jmg = pose_mg.getRobotModel()->getJointModelGroup(kPoseGroup);
    if (!jmg)
    {
      RCLCPP_ERROR(logger, "规划组 '%s' 不存在", kPoseGroup);
      removeDeepFrame(planning_scene);
      rclcpp::shutdown();
      spinner.join();
      return 1;
    }

    configureMoveGroup(pose_mg, kPosePlanningTimeSec, kPosePlanningAttempts);

    const auto start_opt = readFreshJointStates(node, jmg);
    if (!start_opt)
    {
      removeDeepFrame(planning_scene);
      rclcpp::shutdown();
      spinner.join();
      return 1;
    }

    moveit::core::RobotState start_state(*pose_mg.getCurrentState());
    start_state.setVariablePositions(*start_opt);
    pose_mg.setStartState(start_state);

    pose_mg.setEndEffectorLink(kEndEffectorLink);
    const std::string planning_frame = pose_mg.getPlanningFrame();
    const geometry_msgs::msg::PoseStamped goal_pose =
      makeEeGoalPoseStamped(node, planning_frame);

    pose_mg.clearPoseTargets();
    pose_mg.setPoseTarget(goal_pose);

    RCLCPP_INFO(
      logger,
      "[%s] 位姿目标 %s @ %s: pos (%.3f, %.3f, %.3f), RPY (%.3f, %.3f, %.3f), %u attempts",
      kPoseGroup, kEndEffectorLink, planning_frame.c_str(), kEeX, kEeY, kEeZ, kEeRoll, kEePitch,
      kEeYaw, kPosePlanningAttempts);

    if (!planAndExecute(logger, pose_mg, kPoseGroup))
    {
      removeDeepFrame(planning_scene);
      rclcpp::shutdown();
      spinner.join();
      return 1;
    }
  }

  exit_code = 0;
  removeDeepFrame(planning_scene);
  RCLCPP_INFO(logger, "全部完成，深框已移除。");

  rclcpp::shutdown();
  spinner.join();
  return exit_code;
}
