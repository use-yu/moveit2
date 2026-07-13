/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2013, Unbounded Robotics Inc.
 *  Copyright (c) 2012, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/* Author: Michael Ferguson, Ioan Sucan, E. Gil Jones */

#include <moveit_simple_controller_manager/follow_joint_trajectory_controller_handle.h>

#include <algorithm>
#include <chrono>
#include <thread>

using namespace std::placeholders;

namespace moveit_simple_controller_manager
{
namespace
{
constexpr char G01_HARDWARE_PLUGIN[] = "g01_topic_hardware/G01TopicSystem";
constexpr char MOTOR_COMMAND_TOPIC[] = "/g01/motor_commands";
constexpr char WAIST_CSP_PATH_TOPIC[] = "/waist_motor_control/csp_path";
constexpr char LIFT_CSP_PATH_TOPIC[] = "/lift_motor_control/csp_path";
constexpr char WAIST_JOINT[] = "body_joint2";
constexpr char LIFT_JOINT[] = "body_joint1";
constexpr double WAIST_POSITION_SCALE = 1.0;    // rad -> rad
constexpr double LIFT_POSITION_SCALE = 1000.0;  // m -> mm
constexpr auto CSP_PRELOAD_DELAY = std::chrono::milliseconds(100);
}  // namespace

FollowJointTrajectoryControllerHandle::FollowJointTrajectoryControllerHandle(const rclcpp::Node::SharedPtr& node,
                                                                             const std::string& name,
                                                                             const std::string& action_ns)
  : ActionBasedControllerHandle<control_msgs::action::FollowJointTrajectory>(
        node, name, action_ns, "moveit.simple_controller_manager.follow_joint_trajectory_controller_handle")
{
  // G01 的仿真和真机共用 moveit_controllers.yaml。只在 robot_description 明确使用
  // 真机硬件插件时启用 CSP 完整轨迹预下发，避免仿真启动时触发电机话题。
  std::string robot_description;
  if (!node->get_parameter("robot_description", robot_description) ||
      robot_description.find(G01_HARDWARE_PLUGIN) == std::string::npos)
  {
    return;
  }

  g01_csp_preload_enabled_ = true;
  motor_command_pub_ = node->create_publisher<std_msgs::msg::UInt8>(MOTOR_COMMAND_TOPIC, rclcpp::QoS(5));
  waist_csp_path_pub_ = node->create_publisher<std_msgs::msg::Float32MultiArray>(WAIST_CSP_PATH_TOPIC, rclcpp::QoS(5));
  lift_csp_path_pub_ = node->create_publisher<std_msgs::msg::Float32MultiArray>(LIFT_CSP_PATH_TOPIC, rclcpp::QoS(5));

  RCLCPP_INFO_STREAM(LOGGER, "G01 CSP trajectory preload enabled for " << name_);
}

bool FollowJointTrajectoryControllerHandle::appendJointPath(const trajectory_msgs::msg::JointTrajectory& trajectory,
                                                            const std::string& joint_name, const double position_scale,
                                                            std_msgs::msg::Float32MultiArray& path) const
{
  const auto joint_it = std::find(trajectory.joint_names.begin(), trajectory.joint_names.end(), joint_name);
  if (joint_it == trajectory.joint_names.end())
    return false;

  const auto joint_index = static_cast<std::size_t>(std::distance(trajectory.joint_names.begin(), joint_it));
  path.data.reserve(trajectory.points.size());
  for (const auto& point : trajectory.points)
  {
    if (joint_index >= point.positions.size())
    {
      RCLCPP_ERROR_STREAM(LOGGER, "Cannot preload " << joint_name << " CSP path: trajectory point has "
                                                    << point.positions.size() << " positions, but joint index is "
                                                    << joint_index);
      path.data.clear();
      return false;
    }
    path.data.push_back(static_cast<float>(point.positions[joint_index] * position_scale));
  }

  std_msgs::msg::MultiArrayDimension dimension;
  dimension.label = joint_name;
  dimension.size = static_cast<uint32_t>(path.data.size());
  dimension.stride = dimension.size;
  path.layout.dim.push_back(std::move(dimension));
  return true;
}

void FollowJointTrajectoryControllerHandle::publishG01CspPaths(const trajectory_msgs::msg::JointTrajectory& trajectory)
{
  if (!g01_csp_preload_enabled_ || trajectory.points.empty())
    return;

  std_msgs::msg::Float32MultiArray waist_path;
  std_msgs::msg::Float32MultiArray lift_path;
  const bool has_waist = appendJointPath(trajectory, WAIST_JOINT, WAIST_POSITION_SCALE, waist_path);
  const bool has_lift = appendJointPath(trajectory, LIFT_JOINT, LIFT_POSITION_SCALE, lift_path);
  if (!has_waist && !has_lift)
    return;

  // 发布顺序是协议的一部分：先给电机控制节点发 1，再发送完整 CSP 轨迹，
  // 最后才把 FollowJointTrajectory goal 交给控制器。真机硬件插件只有在 goal
  // 进入 EXECUTING 后才会开始发布 /g01/joint_commands。
  std_msgs::msg::UInt8 motor_command;
  motor_command.data = 1;
  motor_command_pub_->publish(motor_command);

  // 先让电机控制节点处理停止/切换命令，再发送新的 CSP 完整轨迹。
  std::this_thread::sleep_for(CSP_PRELOAD_DELAY);

  if (has_waist)
    waist_csp_path_pub_->publish(std::move(waist_path));
  if (has_lift)
    lift_csp_path_pub_->publish(std::move(lift_path));

  // 给电机控制节点留出 0.1 秒接收和处理完整 CSP 数组，然后才发送 action goal。
  std::this_thread::sleep_for(CSP_PRELOAD_DELAY);

  RCLCPP_INFO_STREAM(LOGGER, "Preloaded G01 CSP trajectory with two "
                                 << CSP_PRELOAD_DELAY.count() << " ms delays before " << name_
                                 << " execution: " << (has_waist ? "waist " : "") << (has_lift ? "lift " : "") << "("
                                 << trajectory.points.size() << " points)");
}

bool FollowJointTrajectoryControllerHandle::sendTrajectory(const moveit_msgs::msg::RobotTrajectory& trajectory)
{
  RCLCPP_DEBUG_STREAM(LOGGER, "new trajectory to " << name_);

  if (!controller_action_client_)
    return false;

  if (!isConnected())
  {
    RCLCPP_ERROR_STREAM(LOGGER, "Action client not connected to action server: " << getActionName());
    return false;
  }

  if (done_)
    RCLCPP_INFO_STREAM(LOGGER, "sending trajectory to " << name_);
  else
    RCLCPP_INFO_STREAM(LOGGER, "sending continuation for the currently executed trajectory to " << name_);

  control_msgs::action::FollowJointTrajectory::Goal goal = goal_template_;
  goal.trajectory = trajectory.joint_trajectory;
  goal.multi_dof_trajectory = trajectory.multi_dof_joint_trajectory;

  publishG01CspPaths(goal.trajectory);

  rclcpp_action::Client<control_msgs::action::FollowJointTrajectory>::SendGoalOptions send_goal_options;
  // Active callback
  send_goal_options.goal_response_callback =
      [this](
          const rclcpp_action::Client<control_msgs::action::FollowJointTrajectory>::GoalHandle::SharedPtr& goal_handle) {
        RCLCPP_INFO_STREAM(LOGGER, name_ << " started execution");
        if (!goal_handle)
          RCLCPP_WARN(LOGGER, "Goal request rejected");
        else
          RCLCPP_INFO(LOGGER, "Goal request accepted!");
      };

  done_ = false;
  last_exec_ = moveit_controller_manager::ExecutionStatus::RUNNING;

  // Send goal
  auto current_goal_future = controller_action_client_->async_send_goal(goal, send_goal_options);
  current_goal_ = current_goal_future.get();
  if (!current_goal_)
  {
    RCLCPP_ERROR(LOGGER, "Goal was rejected by server");
    return false;
  }
  return true;
}

// TODO(JafarAbdi): Revise parameter lookup
// void FollowJointTrajectoryControllerHandle::configure(XmlRpc::XmlRpcValue& config)
//{
//  if (config.hasMember("path_tolerance"))
//    configure(config["path_tolerance"], "path_tolerance", goal_template_.path_tolerance);
//  if (config.hasMember("goal_tolerance"))
//    configure(config["goal_tolerance"], "goal_tolerance", goal_template_.goal_tolerance);
//  if (config.hasMember("goal_time_tolerance"))
//    goal_template_.goal_time_tolerance = ros::Duration(parseDouble(config["goal_time_tolerance"]));
//}

namespace
{
enum ToleranceVariables
{
  POSITION,
  VELOCITY,
  ACCELERATION
};
template <ToleranceVariables>
double& variable(control_msgs::msg::JointTolerance& msg);

template <>
inline double& variable<POSITION>(control_msgs::msg::JointTolerance& msg)
{
  return msg.position;
}
template <>
inline double& variable<VELOCITY>(control_msgs::msg::JointTolerance& msg)
{
  return msg.velocity;
}
template <>
inline double& variable<ACCELERATION>(control_msgs::msg::JointTolerance& msg)
{
  return msg.acceleration;
}

static std::map<ToleranceVariables, std::string> VAR_NAME = { { POSITION, "position" },
                                                              { VELOCITY, "velocity" },
                                                              { ACCELERATION, "acceleration" } };
static std::map<ToleranceVariables, decltype(&variable<POSITION>)> VAR_ACCESS = { { POSITION, &variable<POSITION> },
                                                                                  { VELOCITY, &variable<VELOCITY> },
                                                                                  { ACCELERATION,
                                                                                    &variable<ACCELERATION> } };

const char* errorCodeToMessage(int error_code)
{
  switch (error_code)
  {
    case control_msgs::action::FollowJointTrajectory::Result::SUCCESSFUL:
      return "SUCCESSFUL";
    case control_msgs::action::FollowJointTrajectory::Result::INVALID_GOAL:
      return "INVALID_GOAL";
    case control_msgs::action::FollowJointTrajectory::Result::INVALID_JOINTS:
      return "INVALID_JOINTS";
    case control_msgs::action::FollowJointTrajectory::Result::OLD_HEADER_TIMESTAMP:
      return "OLD_HEADER_TIMESTAMP";
    case control_msgs::action::FollowJointTrajectory::Result::PATH_TOLERANCE_VIOLATED:
      return "PATH_TOLERANCE_VIOLATED";
    case control_msgs::action::FollowJointTrajectory::Result::GOAL_TOLERANCE_VIOLATED:
      return "GOAL_TOLERANCE_VIOLATED";
    default:
      return "unknown error";
  }
}
}  // namespace

// TODO(JafarAbdi): Revise parameter lookup
// void FollowJointTrajectoryControllerHandle::configure(XmlRpc::XmlRpcValue& config, const std::string& config_name,
//                                                      std::vector<control_msgs::JointTolerance>& tolerances)
//{
//  if (isStruct(config))  // config should be either a struct of position, velocity, acceleration
//  {
//    for (ToleranceVariables var : { POSITION, VELOCITY, ACCELERATION })
//    {
//      if (!config.hasMember(VAR_NAME[var]))
//        continue;
//      XmlRpc::XmlRpcValue values = config[VAR_NAME[var]];
//      if (isArray(values, joints_.size()))
//      {
//        size_t i = 0;
//        for (const auto& joint_name : joints_)
//          VAR_ACCESS[var](getTolerance(tolerances, joint_name)) = parseDouble(values[i++]);
//      }
//      else
//      {  // use common value for all joints
//        double value = parseDouble(values);
//        for (const auto& joint_name : joints_)
//          VAR_ACCESS[var](getTolerance(tolerances, joint_name)) = value;
//      }
//    }
//  }
//  else if (isArray(config))  // or an array of JointTolerance msgs
//  {
//    for (int i = 0; i < config.size(); ++i)  // NOLINT(modernize-loop-convert)
//    {
//      control_msgs::JointTolerance& tol = getTolerance(tolerances, config[i]["name"]);
//      for (ToleranceVariables var : { POSITION, VELOCITY, ACCELERATION })
//      {
//        if (!config[i].hasMember(VAR_NAME[var]))
//          continue;
//        VAR_ACCESS[var](tol) = parseDouble(config[i][VAR_NAME[var]]);
//      }
//    }
//  }
//  else
//    ROS_WARN_STREAM_NAMED(LOGNAME, "Invalid " << config_name);
//}

control_msgs::msg::JointTolerance&
FollowJointTrajectoryControllerHandle::getTolerance(std::vector<control_msgs::msg::JointTolerance>& tolerances,
                                                    const std::string& name)
{
  auto it = std::lower_bound(tolerances.begin(), tolerances.end(), name,
                             [](const control_msgs::msg::JointTolerance& lhs, const std::string& rhs) {
                               return lhs.name < rhs;
                             });
  if (it == tolerances.cend() || it->name != name)
  {  // insert new entry if not yet available
    it = tolerances.insert(it, control_msgs::msg::JointTolerance());
    it->name = name;
  }
  return *it;
}

void FollowJointTrajectoryControllerHandle::controllerDoneCallback(
    const rclcpp_action::ClientGoalHandle<control_msgs::action::FollowJointTrajectory>::WrappedResult& wrapped_result)
{
  // Output custom error message for FollowJointTrajectoryResult if necessary
  if (!wrapped_result.result)
    RCLCPP_WARN_STREAM(LOGGER, "Controller '" << name_ << "' done, no result returned");
  else if (wrapped_result.result->error_code == control_msgs::action::FollowJointTrajectory::Result::SUCCESSFUL)
    RCLCPP_INFO_STREAM(LOGGER, "Controller '" << name_ << "' successfully finished");
  else
    RCLCPP_WARN_STREAM(LOGGER, "Controller '" << name_ << "' failed with error "
                                              << errorCodeToMessage(wrapped_result.result->error_code) << ": "
                                              << wrapped_result.result->error_string);
  finishControllerExecution(wrapped_result.code);
}

}  // end namespace moveit_simple_controller_manager
