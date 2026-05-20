# moveit2
planning_context_manager.cpp 
max_planning_threads_()可以设置同时跑几个 planner，默认是 4。

超出关节限制，超界目标会被静默改到限位


wheel_robot.srdf 可以修改忽略的碰撞link

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