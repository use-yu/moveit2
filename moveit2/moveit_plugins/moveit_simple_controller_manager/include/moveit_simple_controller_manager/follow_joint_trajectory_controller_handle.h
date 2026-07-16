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

#pragma once

#include <moveit_simple_controller_manager/action_based_controller_handle.h>
#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <control_msgs/msg/joint_tolerance.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <functional>

namespace moveit_simple_controller_manager
{
/*
 * This is generally used for arms, but could also be used for multi-dof hands,
 * or anything using a control_mgs/FollowJointTrajectoryAction.
 */
class FollowJointTrajectoryControllerHandle
  : public ActionBasedControllerHandle<control_msgs::action::FollowJointTrajectory>
{
public:
  FollowJointTrajectoryControllerHandle(const rclcpp::Node::SharedPtr& node, const std::string& name,
                                        const std::string& action_ns);

  bool prepareTrajectory(moveit_msgs::msg::RobotTrajectory& trajectory) override;
  bool coordinatesTrajectoryExecution() const override;
  bool prepareTrajectoryExecution(const std::vector<moveit_msgs::msg::RobotTrajectory>& trajectories) override;
  rclcpp::Duration getTrajectoryStartDelay() const override;
  bool publishTrajectoryExecutionSchedule(const rclcpp::Time& start_time,
                                          const std::vector<std::string>& joint_names) override;
  bool sendTrajectory(const moveit_msgs::msg::RobotTrajectory& trajectory) override;

  // TODO(JafarAbdi): Revise parameter lookup
  // void configure(XmlRpc::XmlRpcValue& config) override;

protected:
  static control_msgs::msg::JointTolerance& getTolerance(std::vector<control_msgs::msg::JointTolerance>& tolerances,
                                                         const std::string& name);

  void controllerDoneCallback(
      const rclcpp_action::ClientGoalHandle<control_msgs::action::FollowJointTrajectory>::WrappedResult& wrapped_result)
      override;

  control_msgs::action::FollowJointTrajectory::Goal goal_template_;

private:
  bool synchronizeG01BodyTrajectory(trajectory_msgs::msg::JointTrajectory& trajectory) const;
  bool publishG01CspPaths(const std::vector<moveit_msgs::msg::RobotTrajectory>& trajectories);
  bool appendJointPath(const trajectory_msgs::msg::JointTrajectory& trajectory, const std::string& joint_name,
                       double position_scale, std_msgs::msg::Float32MultiArray& path) const;

  bool g01_body_trajectory_sync_enabled_{ false };
  bool g01_csp_preload_enabled_{ false };
  rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr motor_command_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr waist_csp_path_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr lift_csp_path_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr execution_schedule_pub_;
  bool trajectory_prepared_{ false };
};

}  // end namespace moveit_simple_controller_manager
