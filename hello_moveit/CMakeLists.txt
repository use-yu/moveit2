cmake_minimum_required(VERSION 3.8)
project(hello_moveit)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_visual_tools REQUIRED)

add_executable(hello_moveit src/hello_moveit.cpp)
add_executable(hello_qj2m src/hello_qj2m.cpp)
add_executable(hello_left_arm src/hello_left_arm.cpp)
add_executable(hello_left_arm_ompl_chomp src/hello_left_arm_ompl_chomp.cpp)
target_include_directories(hello_moveit PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(hello_moveit PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(
  hello_moveit
  "moveit_ros_planning_interface"
  "moveit_visual_tools"
  "rclcpp"
)

target_include_directories(hello_qj2m PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(hello_qj2m PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(
  hello_qj2m
  "moveit_ros_planning_interface"
  "rclcpp"
  "sensor_msgs"
)

target_include_directories(hello_left_arm PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(hello_left_arm PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(
  hello_left_arm
  "moveit_ros_planning_interface"
  "rclcpp"
  "sensor_msgs"
)

target_include_directories(hello_left_arm_ompl_chomp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(hello_left_arm_ompl_chomp PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(
  hello_left_arm_ompl_chomp
  "moveit_ros_planning_interface"
  "rclcpp"
  "sensor_msgs"
)

install(TARGETS hello_moveit hello_qj2m hello_left_arm hello_left_arm_ompl_chomp
  DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS
  scripts/hello_deep_frame.py
  scripts/hello_deep_frame_body.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
