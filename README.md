# moveit2
运行代码：

g01实际部署：

启动tcp-ros通信：
    python3 servoj_30ms.py
启动真机话题桥接：
    python3 comm.py
启动腰部控制通信：
    python3 waist.py
启动moveit：
    ros2 launch g01_moveit_config demo.launch.py use_real_hardware:=true
    
ros2 service call /g01/right/tool_commands dobot_msgs_v4/srv/SetToolPower "{status: 0}"

官方 ros2 SDK：
    ros2 launch cr_robot_ros2 dobot_bringup_ros2.launch.py

仿真：
ros2 launch g01_moveit_config demo.launch.py
抓取轨迹：
    python3 g01.py

官方例子
ros2 launch moveit2_tutorials demo.launch.py

qj2m：
ros2 launch qj2m_moveit_config demo.launch.py
ros2 run hello_moveit hello_qj2m

MoveIt 设置助手：
ros2 launch moveit_setup_assistant setup_assistant.launch.py
1. moveit_resources文件夹下复制一个包
2. 修改 package.xml 和 CMakeLists.txt 的包名字，修改urdf的meshes路径
3. colcon build --packages-select moveit_resources && source install/setup.bash

注意生成的joint_limits.yaml文件整数要改成小数
注意生成控制器时要拆细一点：底盘+腰+双臂
注意要添加虚拟关节，不然会报错
注意meshes文件太大会卡，可以用open3D库缩小

#
对于双臂无法给双臂的位姿规划，需要先求逆解再规划
适用于单臂关节空间或笛卡尔空间规划，双臂只能自己先求逆解关节空间规划

对位姿目标，MoveIt在关节空间里现算一个（或少量）同时满足：位姿约束 +无碰撞 的关节状态，作为这次规划里可用的构型
容差
move_group_interface.cpp
    goal_joint_tolerance_ = 1e-4;
    goal_position_tolerance_ = 1e-4;     // 0.1 mm
    goal_orientation_tolerance_ = 1e-3;  // ~0.1 deg

#
TOTG 时间参数化：给路径加上时间

#
chomp规划时间太长，超10s

#
MTC 先在 ComputeIK 那里枚举 N 个 IK 解作为分支；对每个分支尝试 MoveRelative approach。approach 失败的 IK 解被剪掉，剩下的可行 IK 解再回填给上游 Connect（OMPL）作为目标关节配置。所以"approach 不可行的 IK 直接淘汰"这一句话就是 MTC 内置的行为

#
实际测试同一起点和终点
不加hybridize和simplifySolution实际70个点  73.910 ms-

加上simplifySolution实际26或27个点 383.810 ms
加上hybridize实际26的频率增多  100次20核 910.490 ms


model_based_planning_context.cpp
hybridize: true 时尝试A前半 + B中段 + C结尾组合成更短 / 更平滑 / 更自然


planning_context_manager.cpp 
max_planning_threads_()可以设置同时跑几个 planner，默认是 4。

起点和终点超出关节限制，超界目标会被静默改到限位


wheel_robot.srdf 可以修改忽略的碰撞link
去掉前190个，去掉后不检测的碰撞后还有30个

规划成功后，服务端 OMPL 会自动做simplifySolution。
1. Shortcut（最重要）
尝试跳过中间节点
例如：

A -> B -> C

检查：

A -> C

是否：

无碰撞
满足约束

如果可以：

删除 B

这是最核心的操作。

2. Path Pruning

删除：

冗余点
非必要 waypoint
小抖动

让路径更干净。

3. Smoothing

有些 planner 会进一步：

插值
局部优化
曲线平滑