#include "g01_topic_hardware/g01_topic_system.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <utility>

#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <pluginlib/class_list_macros.hpp>

namespace g01_topic_hardware
{

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

  pos_state_.assign(joint_names_.size(), 0.0);
  vel_state_.assign(joint_names_.size(), 0.0);
  pos_cmd_.assign(joint_names_.size(), 0.0);
  last_published_pos_cmd_.assign(joint_names_.size(), std::numeric_limits<double>::quiet_NaN());

  // Read topics from <param> in ros2_control xacro
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
      "Missing required hardware params: state_topic/command_topic. "
      "Check g01_moveit_config/config/real_hardware_topics.yaml and xacro mappings.");
    return hardware_interface::CallbackReturn::ERROR;
  }

  node_ = std::make_shared<rclcpp::Node>("g01_topic_hardware");
  executor_.add_node(node_);

  command_pub_ = node_->create_publisher<sensor_msgs::msg::JointState>(command_topic_, rclcpp::QoS(1));
  state_sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
    state_topic_, rclcpp::QoS(10),
    [this](sensor_msgs::msg::JointState::SharedPtr msg) { this->on_joint_state(*msg); });

  return hardware_interface::CallbackReturn::SUCCESS;
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

  // Start from last known state to avoid a jump
  {
    std::scoped_lock lk(state_mutex_);
    pos_cmd_ = pos_state_;
  }

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn G01TopicSystem::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
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
  // The real robot requires "full joint command" every cycle.
  // Controllers may only claim a subset of joints (e.g., left arm), so we pre-fill all commands
  // with the latest measured state here. Claimed joints will then overwrite their own command
  // values during controller update(), and write() will publish a full vector.
  {
    std::scoped_lock lk(state_mutex_);
    pos_cmd_ = pos_state_;
  }
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type G01TopicSystem::write(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  if (!command_pub_) {
    return hardware_interface::return_type::ERROR;
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

  const auto & names = msg.name;
  for (size_t i = 0; i < names.size(); ++i) {
    auto it = joint_index_.find(names[i]);
    if (it == joint_index_.end()) {
      continue;
    }
    const size_t idx = it->second;
    if (i < msg.position.size()) {
      pos_state_[idx] = msg.position[i];
    }
    if (i < msg.velocity.size()) {
      vel_state_[idx] = msg.velocity[i];
    }
  }
}

}  // namespace g01_topic_hardware

PLUGINLIB_EXPORT_CLASS(g01_topic_hardware::G01TopicSystem, hardware_interface::SystemInterface)

