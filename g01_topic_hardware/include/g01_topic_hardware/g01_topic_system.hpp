#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <thread>

#include <hardware_interface/system_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/state.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

namespace g01_topic_hardware
{

class G01TopicSystem final : public hardware_interface::SystemInterface
{
public:
  RCLCPP_SHARED_PTR_DEFINITIONS(G01TopicSystem)

  hardware_interface::CallbackReturn on_init(const hardware_interface::HardwareInfo & info) override;
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;
  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  void on_joint_state(const sensor_msgs::msg::JointState & msg);
  void setup_trajectory_action_watchers();
  bool is_trajectory_executing() const;
  bool has_valid_state() const;

  std::shared_ptr<rclcpp::Node> node_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  std::thread spin_thread_;
  std::atomic<bool> running_{false};
  std::atomic<bool> state_received_{false};

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr state_sub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr command_pub_;

  std::string state_topic_;
  std::string command_topic_;

  std::vector<std::string> joint_names_;
  std::unordered_map<std::string, size_t> joint_index_;

  std::mutex state_mutex_;
  std::vector<double> pos_state_;
  std::vector<double> vel_state_;
  std::vector<double> pos_cmd_;

  rclcpp::Time last_state_stamp_;
  std::atomic<int64_t> last_state_rx_ns_{0};
  double state_timeout_sec_{0.5};

  std::vector<std::shared_ptr<std::atomic<bool>>> controller_executing_;
  std::vector<rclcpp::SubscriptionBase::SharedPtr> action_status_subs_;
};

}  // namespace g01_topic_hardware
