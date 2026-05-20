// colcon build --packages-select hello_moveit
// pkill -f ros2 || true
// pkill -f move_group || true
// htop -p $(pgrep -f move_group)

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
constexpr char kPlanningGroup[] = "left_arm";
// 从零开始多次独立规划，再取最短路径
constexpr unsigned int kParallelPlanningAttempts = 100;
// 给 joint limit 的最大速度乘一个比例系数
constexpr double kVelocityScale = 0.5;
constexpr double kAccelerationScale = 0.5;

using JointMap = std::map<std::string, double>;

// 仅左臂 7 关节目标 [rad]
const JointMap kGoalJoints{
  {"l_arm_Joint1", 0.5},
  {"l_arm_Joint2", -1},
  {"l_arm_Joint3", 1.5},
  {"l_arm_Joint4", 0.4},
  {"l_arm_Joint5", 1.3},
  {"l_arm_Joint6", 0.8},
  {"l_arm_Joint7", 0.9},
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

void warnGoalOutOfJointLimits(
  const rclcpp::Logger & logger,
  const moveit::core::RobotModelConstPtr & model,
  const JointMap & goal)
{
  for (const auto & [name, value] : goal)
  {
    const moveit::core::VariableBounds & bounds = model->getVariableBounds(name);
    if (!bounds.position_bounded_)
    {
      continue;
    }
    if (value < bounds.min_position_)
    {
      RCLCPP_WARN(
        logger,
        "Goal joint '%s': requested %.4f < min bound %.4f; move_group will clamp to min.",
        name.c_str(), value, bounds.min_position_);
    }
    else if (value > bounds.max_position_)
    {
      RCLCPP_WARN(
        logger,
        "Goal joint '%s': requested %.4f > max bound %.4f; move_group will clamp to max.",
        name.c_str(), value, bounds.max_position_);
    }
  }
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
    "hello_left_arm",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  auto const logger = rclcpp::get_logger("hello_left_arm");

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

  move_group.setMaxVelocityScalingFactor(kVelocityScale);
  move_group.setMaxAccelerationScalingFactor(kAccelerationScale);
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
  warnGoalOutOfJointLimits(logger, move_group.getRobotModel(), goal);
  if (!move_group.setJointValueTarget(goal))
  {
    RCLCPP_WARN(logger, "setJointValueTarget() returned false (goal out of bounds).");
  }

  RCLCPP_INFO(
    logger, "Planning group: %s (%u joints), planner: RRTConnect, attempts: %u",
    kPlanningGroup, jmg->getVariableCount(), kParallelPlanningAttempts);

  MoveGroupInterface::Plan plan;
  const auto plan_start = std::chrono::steady_clock::now();
  const bool plan_ok = static_cast<bool>(move_group.plan(plan));
  const auto plan_end = std::chrono::steady_clock::now();
  const double plan_wall_ms =
    std::chrono::duration<double, std::milli>(plan_end - plan_start).count();
  const double plan_server_ms = plan.planning_time_ * 1000.0;

  RCLCPP_INFO(
    logger,
    "plan() wall time (client, incl. ROS action): %.3f ms (%s)",
    plan_wall_ms, plan_ok ? "success" : "failed");
  RCLCPP_INFO(
    logger,
    "move_group planning_time: %.3f ms",
    plan_server_ms);
  if (plan_ok)
  {
    const size_t num_waypoints = plan.trajectory_.joint_trajectory.points.size();
    RCLCPP_INFO(logger, "planned trajectory: %zu waypoints", num_waypoints);
  }

  if (!plan_ok)
  {
    RCLCPP_ERROR(logger, "Planning failed. See move_group terminal for OMPL details.");
    rclcpp::shutdown();
    spinner.join();
    return 1;
  }

  const auto exec_start = std::chrono::steady_clock::now();
  const bool exec_ok = static_cast<bool>(move_group.execute(plan));
  const auto exec_end = std::chrono::steady_clock::now();
  // const double exec_ms =
  //   std::chrono::duration<double, std::milli>(exec_end - exec_start).count();

  // RCLCPP_INFO(logger, "Execution time: %.3f ms (%s)", exec_ms, exec_ok ? "success" : "failed");
  // RCLCPP_INFO(logger, "Total time: %.3f ms", plan_wall_ms + exec_ms);

  rclcpp::shutdown();
  spinner.join();
  return exec_ok ? 0 : 1;
}
