#include <chrono>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <thread>  // <---- add this to the set of includes at the top
#include <moveit/planning_scene_interface/planning_scene_interface.h>

int main(int argc, char * argv[])
{
  // Initialize ROS and create the Node
  rclcpp::init(argc, argv);
  auto const node = std::make_shared<rclcpp::Node>(
    "hello_moveit",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)
  );

  // Create a ROS logger
  auto const logger = rclcpp::get_logger("hello_moveit");

  // Spin up a SingleThreadedExecutor for MoveItVisualTools to interact with ROS
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  auto spinner = std::thread([&executor]() { executor.spin(); });

  // Create the MoveIt MoveGroup Interface
  using moveit::planning_interface::MoveGroupInterface;
  auto move_group_interface = MoveGroupInterface(node, "panda_arm");

  // Construct and initialize MoveItVisualTools
  auto moveit_visual_tools = moveit_visual_tools::MoveItVisualTools{
    node, "panda_link0", rviz_visual_tools::RVIZ_MARKER_TOPIC,
    move_group_interface.getRobotModel()};
  moveit_visual_tools.deleteAllMarkers();
  moveit_visual_tools.loadRemoteControl();
    
  // Create closures for visualization
  auto const draw_title = [&moveit_visual_tools](auto text) {
    auto const text_pose = [] {
      auto msg = Eigen::Isometry3d::Identity();
      msg.translation().z() = 1.0;  // Place text 1m above the base link
      return msg;
    }();
    moveit_visual_tools.publishText(text_pose, text, rviz_visual_tools::WHITE,
                                    rviz_visual_tools::XLARGE);
  };
  auto const prompt = [&moveit_visual_tools](auto text) {
    moveit_visual_tools.prompt(text);
  };
  auto const draw_trajectory_tool_path =
      [&moveit_visual_tools,
      jmg = move_group_interface.getRobotModel()->getJointModelGroup(
          "panda_arm")](auto const trajectory) {
        moveit_visual_tools.publishTrajectoryLine(trajectory, jmg);
      };

  // Set a target Pose with updated values !!!
  auto const target_pose = [] {
    geometry_msgs::msg::Pose msg;
    msg.orientation.y = 0.8;
    msg.orientation.w = 0.6;
    msg.position.x = 0.1;
    msg.position.y = 0.4;
    msg.position.z = 0.4;
    return msg;
  }();
  move_group_interface.setPoseTarget(target_pose);

  // Create deep frame (深框) collision object: outer 0.7 x 0.7 x 0.6 m, wall tb = 0.01 m
  auto const collision_object = [frame_id =
    move_group_interface.getPlanningFrame()] {
  constexpr double L = 0.7;
  constexpr double W = 0.7;
  constexpr double H = 0.6;
  constexpr double tb = 0.01;
  // Bottom-center of the frame in planning frame
  constexpr double base_x = 0.5;
  constexpr double base_y = 0.0;
  constexpr double base_z = 0.0;

  moveit_msgs::msg::CollisionObject collision_object;
  collision_object.header.frame_id = frame_id;
  collision_object.id = "深框";

  auto add_wall = [&](double dx, double dy, double dz, double x, double y, double z) {
    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    primitive.dimensions = { dx, dy, dz };
    geometry_msgs::msg::Pose pose;
    pose.orientation.w = 1.0;
    pose.position.x = base_x + x;
    pose.position.y = base_y + y;
    pose.position.z = base_z + z;
    collision_object.primitives.push_back(primitive);
    collision_object.primitive_poses.push_back(pose);
  };

  // Bottom + four walls (open top)
  add_wall(L, W, tb, 0.0, 0.0, tb / 2.0);
  add_wall(tb, W, H, L / 2.0 - tb / 2.0, 0.0, H / 2.0);
  add_wall(tb, W, H, -(L / 2.0 - tb / 2.0), 0.0, H / 2.0);
  add_wall(L - 2.0 * tb, tb, H, 0.0, W / 2.0 - tb / 2.0, H / 2.0);
  add_wall(L - 2.0 * tb, tb, H, 0.0, -(W / 2.0 - tb / 2.0), H / 2.0);

  collision_object.operation = collision_object.ADD;
  return collision_object;
  }();

  // Add the collision object to the scene
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
  // planning_scene_interface.applyCollisionObject(collision_object);

  // Create a plan to that target pose
  prompt("Press 'Next' in the RvizVisualToolsGui window to plan");
  draw_title("Planning");
  moveit_visual_tools.trigger();
  MoveGroupInterface::Plan plan;
  const auto plan_start = std::chrono::steady_clock::now();
  const bool success = static_cast<bool>(move_group_interface.plan(plan));
  const auto plan_end = std::chrono::steady_clock::now();
  const double plan_ms =
    std::chrono::duration<double, std::milli>(plan_end - plan_start).count();
  RCLCPP_INFO(logger, "Planning time: %.3f ms (%s)", plan_ms, success ? "success" : "failed");

  // Execute the plan
  if (success) {
    draw_trajectory_tool_path(plan.trajectory_);
    moveit_visual_tools.trigger();
    prompt("Press 'Next' in the RvizVisualToolsGui window to execute");
    draw_title("Executing");
    moveit_visual_tools.trigger();
    const auto exec_start = std::chrono::steady_clock::now();
    const bool exec_ok = static_cast<bool>(move_group_interface.execute(plan));
    const auto exec_end = std::chrono::steady_clock::now();
    const double exec_ms =
      std::chrono::duration<double, std::milli>(exec_end - exec_start).count();
    RCLCPP_INFO(logger, "Execution time: %.3f ms (%s)", exec_ms, exec_ok ? "success" : "failed");
    RCLCPP_INFO(logger, "Total time: %.3f ms", plan_ms + exec_ms);
  } else {
    draw_title("Planning Failed!");
    moveit_visual_tools.trigger();
    RCLCPP_ERROR(logger, "Planning failed!");
  }

  // Shutdown ROS
  rclcpp::shutdown();  // <--- This will cause the spin function in the thread to return
  spinner.join();  // <--- Join the thread before exiting
  return 0;
}