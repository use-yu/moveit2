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

namespace
{
constexpr char kPlanningGroup[] = "dual_manipulator";
// OMPL 并行规划：多路 RRTConnect 同时采样，碰撞检测在多线程中执行（MoveIt 默认最多 4 线程）
// hybridize: true 时尝试A前半 + B中段 + C结尾组合成更短 / 更平滑 / 更自然
constexpr unsigned int kParallelPlanningAttempts = 20;

using JointMap = std::map<std::string, double>;

const JointMap kGoalJoints{
  {"base_joint1", 0.5},
  {"base_joint2", 0.2},
  {"base_joint3", 0.2},
  {"body_joint1", 0.2},
  {"body_joint2", 0.2},
  {"body_joint3", 0.2},
  {"body_joint4", 0.2},
  {"l_arm_Joint1", 0.9},
  {"l_arm_Joint2", 0.0},
  {"l_arm_Joint3", 0.4},
  {"l_arm_Joint4", 0.5},
  {"l_arm_Joint5", 0.5},
  {"l_arm_Joint6", 0.5},
  {"l_arm_Joint7", 0.5},
  {"r_arm_Joint1", 0.9},
  {"r_arm_Joint2", 0.0},
  {"r_arm_Joint3", 0.5},
  {"r_arm_Joint4", 0.5},
  {"r_arm_Joint5", 0.5},
  {"r_arm_Joint6", 0.5},
  {"r_arm_Joint7", 0.5},
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

}  // namespace

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto const node = std::make_shared<rclcpp::Node>(
    "hello_qj2m",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  auto const logger = rclcpp::get_logger("hello_qj2m");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  auto spinner = std::thread([&executor]() { executor.spin(); });

  using moveit::planning_interface::MoveGroupInterface;
  MoveGroupInterface move_group(node, kPlanningGroup);

  auto const jmg = move_group.getRobotModel()->getJointModelGroup(kPlanningGroup);
  if (!jmg)
  {
    RCLCPP_ERROR(logger, "Planning group '%s' not found.", kPlanningGroup);
    rclcpp::shutdown();
    spinner.join();
    return 1;
  }

  move_group.setMaxVelocityScalingFactor(0.1);
  move_group.setMaxAccelerationScalingFactor(0.1);
  move_group.setPlanningTime(30.0);
  move_group.setPlannerId("RRTConnect");
  move_group.setNumPlanningAttempts(kParallelPlanningAttempts);

  const auto start_opt = readStartFromJointStates(node, jmg);
  if (!start_opt)
  {
    rclcpp::shutdown();
    spinner.join();
    return 1;
  }
  const JointMap & start = *start_opt;
  const JointMap goal = filterToGroup(jmg, kGoalJoints);

  moveit::core::RobotState start_state(*move_group.getCurrentState());
  start_state.setVariablePositions(start);
  move_group.setStartState(start_state);
  move_group.setJointValueTarget(goal);

  RCLCPP_INFO(
    logger, "Planning group: %s (%u joints)", kPlanningGroup,
    jmg->getVariableCount());
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

  if (!plan_ok)
  {
    RCLCPP_ERROR(logger, "Planning failed. See move_group terminal for details.");
    rclcpp::shutdown();
    spinner.join();
    return 1;
  }

  const auto exec_start = std::chrono::steady_clock::now();
  const bool exec_ok = static_cast<bool>(move_group.execute(plan));
  const auto exec_end = std::chrono::steady_clock::now();
  const double exec_ms =
    std::chrono::duration<double, std::milli>(exec_end - exec_start).count();

  RCLCPP_INFO(logger, "Execution time: %.3f ms (%s)", exec_ms, exec_ok ? "success" : "failed");
  RCLCPP_INFO(logger, "Total time: %.3f ms", plan_ms + exec_ms);

  rclcpp::shutdown();
  spinner.join();
  return exec_ok ? 0 : 1;
}
