#include "g01_topic_hardware/g01_topic_system.hpp"

#include <cmath>
#include <limits>
#include <sstream>
#include <utility>

#include <action_msgs/msg/goal_status_array.hpp>
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <pluginlib/class_list_macros.hpp>

namespace g01_topic_hardware
{

namespace
{
std::vector<std::string> split_csv(const std::string & value)
{
  std::vector<std::string> parts;
  std::stringstream ss(value);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      parts.push_back(item);
    }
  }
  return parts;
}
}  // namespace

hardware_interface::CallbackReturn G01TopicSystem::on_init(const hardware_interface::HardwareInfo & info)
{
  if (hardware_interface::SystemInterface::on_init(info) !=
      hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }

  joint_names_.clear();
  joint_index_.clear();
  joint_names_.reserve(info_.joints.size());
  for (size_t i = 0; i < info_.joints.size(); ++i) {
    joint_names_.push_back(info_.joints[i].name);
    joint_index_.emplace(info_.joints[i].name, i);
  }

  const double nan = std::numeric_limits<double>::quiet_NaN();
  pos_state_.assign(joint_names_.size(), nan);
  vel_state_.assign(joint_names_.size(), 0.0);
  pos_cmd_.assign(joint_names_.size(), nan);

  for (size_t i = 0; i < info_.joints.size(); ++i) {
    for (const auto & si : info_.joints[i].state_interfaces) {
      if (si.name == hardware_interface::HW_IF_POSITION && !si.initial_value.empty()) {
        try {
          pos_state_[i] = std::stod(si.initial_value);
          pos_cmd_[i] = pos_state_[i];
        } catch (const std::exception &) {
          RCLCPP_WARN(
            rclcpp::get_logger("g01_topic_hardware"),
            "Invalid initial_value for joint %s", info_.joints[i].name.c_str());
        }
        break;
      }
    }
  }

  auto it_state = info_.hardware_parameters.find("state_topic");
  if (it_state != info_.hardware_parameters.end()) {
    state_topic_ = it_state->second;
  }
  auto it_cmd = info_.hardware_parameters.find("command_topic");
  if (it_cmd != info_.hardware_parameters.end()) {
    command_topic_ = it_cmd->second;
  }

  if (state_topic_.empty() || command_topic_.empty()) {
    RCLCPP_ERROR(
      rclcpp::get_logger("g01_topic_hardware"),
      "Missing state_topic/command_topic. See real_hardware_topics.yaml.");
    return hardware_interface::CallbackReturn::ERROR;
  }

  auto it_timeout = info_.hardware_parameters.find("state_timeout_sec");
  if (it_timeout != info_.hardware_parameters.end()) {
    state_timeout_sec_ = std::stod(it_timeout->second);
  }

  node_ = std::make_shared<rclcpp::Node>("g01_topic_hardware");
  executor_.add_node(node_);

  command_pub_ = node_->create_publisher<sensor_msgs::msg::JointState>(command_topic_, rclcpp::QoS(1));
  state_sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
    state_topic_, rclcpp::QoS(10),
    [this](sensor_msgs::msg::JointState::SharedPtr msg) { this->on_joint_state(*msg); });

  setup_trajectory_action_watchers();

  return hardware_interface::CallbackReturn::SUCCESS;
}

void G01TopicSystem::setup_trajectory_action_watchers()
{
  std::string controllers_csv = "left_arm_controller,right_arm_controller,body_controller";
  auto it_ctrl = info_.hardware_parameters.find("trajectory_controllers");
  if (it_ctrl != info_.hardware_parameters.end()) {
    controllers_csv = it_ctrl->second;
  }

  controller_executing_.clear();
  action_status_subs_.clear();

  for (const auto & controller : split_csv(controllers_csv)) {
    auto executing_flag = std::make_shared<std::atomic<bool>>(false);
    controller_executing_.push_back(executing_flag);

    const std::string status_topic =
      "/" + controller + "/follow_joint_trajectory/_action/status";

    auto sub = node_->create_subscription<action_msgs::msg::GoalStatusArray>(
      status_topic, rclcpp::QoS(10),
      [executing_flag](const action_msgs::msg::GoalStatusArray::SharedPtr msg) {
        bool active = false;
        for (const auto & status : msg->status_list) {
          if (status.status == action_msgs::msg::GoalStatus::STATUS_EXECUTING) {
            active = true;
            break;
          }
        }
        executing_flag->store(active);
      });
    action_status_subs_.push_back(sub);

    RCLCPP_INFO(
      rclcpp::get_logger("g01_topic_hardware"),
      "Trajectory gate: %s", status_topic.c_str());
  }
}

bool G01TopicSystem::is_trajectory_executing() const
{
  for (const auto & flag : controller_executing_) {
    if (flag && flag->load()) {
      return true;
    }
  }
  return false;
}

bool G01TopicSystem::has_valid_state() const
{
  if (!state_received_.load()) {
    return false;
  }
  const int64_t last_ns = last_state_rx_ns_.load();
  if (last_ns <= 0) {
    return false;
  }
  const int64_t now_ns = node_->get_clock()->now().nanoseconds();
  const int64_t timeout_ns = static_cast<int64_t>(state_timeout_sec_ * 1e9);
  return (now_ns - last_ns) >= 0 && (now_ns - last_ns) < timeout_ns;
}

std::vector<hardware_interface::StateInterface> G01TopicSystem::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;
  state_interfaces.reserve(info_.joints.size() * 2);

  for (size_t i = 0; i < info_.joints.size(); ++i) {
    state_interfaces.emplace_back(
      hardware_interface::StateInterface(info_.joints[i].name, hardware_interface::HW_IF_POSITION,
        &pos_state_[i]));
    state_interfaces.emplace_back(
      hardware_interface::StateInterface(info_.joints[i].name, hardware_interface::HW_IF_VELOCITY,
        &vel_state_[i]));
  }

  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface> G01TopicSystem::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;
  command_interfaces.reserve(info_.joints.size());

  for (size_t i = 0; i < info_.joints.size(); ++i) {
    command_interfaces.emplace_back(
      hardware_interface::CommandInterface(info_.joints[i].name, hardware_interface::HW_IF_POSITION,
        &pos_cmd_[i]));
  }
  return command_interfaces;
}

hardware_interface::CallbackReturn G01TopicSystem::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  running_.store(true);
  spin_thread_ = std::thread([this]() {
    rclcpp::WallRate r(500.0);
    while (rclcpp::ok() && running_.load()) {
      executor_.spin_some();
      r.sleep();
    }
  });

  {
    std::scoped_lock lk(state_mutex_);
    pos_cmd_ = pos_state_;
  }

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn G01TopicSystem::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  state_received_.store(false);
  last_state_rx_ns_.store(0);
  for (auto & flag : controller_executing_) {
    if (flag) {
      flag->store(false);
    }
  }
  running_.store(false);
  if (spin_thread_.joinable()) {
    spin_thread_.join();
  }
  executor_.remove_node(node_);
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::return_type G01TopicSystem::read(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  if (!has_valid_state()) {
    RCLCPP_WARN_THROTTLE(
      rclcpp::get_logger("g01_topic_hardware"), *node_->get_clock(), 2000,
      "No valid %s yet (using URDF initial values until full 16-joint state arrives).",
      state_topic_.c_str());
  }

  std::scoped_lock lk(state_mutex_);
  pos_cmd_ = pos_state_;
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type G01TopicSystem::write(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  if (!has_valid_state() || !command_pub_) {
    return hardware_interface::return_type::OK;
  }

  // 仅 MoveIt 点 Execute、FollowJointTrajectory 进入 EXECUTING 后才发 command_topic
  if (!is_trajectory_executing()) {
    return hardware_interface::return_type::OK;
  }

  sensor_msgs::msg::JointState msg;
  msg.header.stamp = node_->get_clock()->now();
  msg.name = joint_names_;
  msg.position.resize(joint_names_.size());
  {
    std::scoped_lock lk(state_mutex_);
    msg.position = pos_cmd_;
  }

  command_pub_->publish(std::move(msg));
  return hardware_interface::return_type::OK;
}

void G01TopicSystem::on_joint_state(const sensor_msgs::msg::JointState & msg)
{
  std::scoped_lock lk(state_mutex_);
  last_state_stamp_ = msg.header.stamp;

  size_t matched = 0;
  for (size_t i = 0; i < msg.name.size(); ++i) {
    auto it = joint_index_.find(msg.name[i]);
    if (it == joint_index_.end()) {
      continue;
    }
    const size_t idx = it->second;
    if (i < msg.position.size()) {
      pos_state_[idx] = msg.position[i];
      ++matched;
    }
    if (i < msg.velocity.size()) {
      vel_state_[idx] = msg.velocity[i];
    }
  }

  if (matched == joint_names_.size()) {
    state_received_.store(true);
    last_state_rx_ns_.store(node_->now().nanoseconds());
  }
}

}  // namespace g01_topic_hardware

PLUGINLIB_EXPORT_CLASS(g01_topic_hardware::G01TopicSystem, hardware_interface::SystemInterface)
