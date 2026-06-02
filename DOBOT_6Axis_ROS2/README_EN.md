## ROS2 Service 列表（含说明）
ros2 launch cr_robot_ros2 dobot_bringup_ros2.launch.py

以下服务由 Dobot bringup 节点提供。

切换tcp
ros2 service call /dobot_right/dobot_bringup_ros2/srv/RequestControl dobot_msgs_v4/srv/RequestControl "{}"

上电
ros2 service call /dobot_right/dobot_bringup_ros2/srv/PowerOn dobot_msgs_v4/srv/PowerOn "{}"

使能
ros2 service call /dobot_right/dobot_bringup_ros2/srv/EnableRobot dobot_msgs_v4/srv/EnableRobot "{}"

错误清除
ros2 service call /dobot_left/dobot_bringup_ros2/srv/ClearError dobot_msgs_v4/srv/ClearError "{}"

MovJ


ServoJ 角度
ros2 service call /dobot_right/dobot_bringup_ros2/srv/ServoJ dobot_msgs_v4/srv/ServoJ "{a: 0.0, b: 0.0, c: 0.0, d: 0.0, e: 0.0, f: 0.0, param_value: ['t=0.03']}"

### 命名说明

- `/dobot_left/...`：单台机器人实例的命名空间（对应 `param.json` 中的 `robot_node_name`）。
- `/dobot_left/dobot_bringup_ros2/srv/...`：将 Dobot TCP 指令封装后的机器人控制/查询服务。
- `/dobot_left/get_parameters` 等：节点自身的 ROS2 标准参数服务。

### 功能分组速览

- 上电与安全：`PowerOn`, `EnableRobot`, `DisableRobot`, `EmergencyStop`, `ClearError`, `ResetRobot`。
- 运动控制：`MovJ`, `MovL`, `Arc`, `Circle`, `MoveJog`, `ServoJ`, `ServoP`, `Stop`, `Pause`。
- 运动参数：`AccJ`, `AccL`, `VelJ`, `VelL`, `CP`, `SpeedFactor`。
- 位姿与运动学：`GetPose`, `GetAngle`, `InverseKin`, `PositiveKin`, `CalcTool`, `CalcUser`。
- IO 与工具 IO：`DI`, `DO`, `AI`, `AO`, `ToolDI`, `ToolDO`, `ToolAI`。
- 力控相关：`FC*`, `ForceDrive*`, `EnableFTSensor`, `SixForceHome`。
- Modbus/寄存器：`Modbus*`, `SetCoils/GetCoils`, `SetHoldRegs/GetHoldRegs`。
- 工具/用户坐标与负载：`Tool`, `User`, `SetTool`, `SetUser`, `SetPayload`, `SetToolPower`, `SetToolMode`。
- 安全配置：`SetCollisionLevel`, `SetSafeSkin`, `SetSafeWallEnable`, `SetWorkZoneEnable`。

### 提示

可用下面命令查看某个 service 的精确请求/响应字段：

```bash
ros2 service type /dobot_left/dobot_bringup_ros2/srv/MovJ
ros2 interface show <service_type_from_above>
```

/dobot_left/describe_parameters
/dobot_left/dobot_bringup_ros2/srv/AI
/dobot_left/dobot_bringup_ros2/srv/AO
/dobot_left/dobot_bringup_ros2/srv/AOInstant
/dobot_left/dobot_bringup_ros2/srv/AccJ
/dobot_left/dobot_bringup_ros2/srv/AccL
/dobot_left/dobot_bringup_ros2/srv/Arc
/dobot_left/dobot_bringup_ros2/srv/BrakeControl
/dobot_left/dobot_bringup_ros2/srv/CP
/dobot_left/dobot_bringup_ros2/srv/CalcTool
/dobot_left/dobot_bringup_ros2/srv/CalcUser
/dobot_left/dobot_bringup_ros2/srv/CheckOddMovC
/dobot_left/dobot_bringup_ros2/srv/CheckOddMovJ
/dobot_left/dobot_bringup_ros2/srv/CheckOddMovL
/dobot_left/dobot_bringup_ros2/srv/Circle
/dobot_left/dobot_bringup_ros2/srv/ClearError
/dobot_left/dobot_bringup_ros2/srv/DI
/dobot_left/dobot_bringup_ros2/srv/DIGroup
/dobot_left/dobot_bringup_ros2/srv/DIGroupDEC
/dobot_left/dobot_bringup_ros2/srv/DO
/dobot_left/dobot_bringup_ros2/srv/DOGroupDEC
/dobot_left/dobot_bringup_ros2/srv/DOInstant
/dobot_left/dobot_bringup_ros2/srv/DisableRobot
/dobot_left/dobot_bringup_ros2/srv/DoGroup
/dobot_left/dobot_bringup_ros2/srv/DragSensivity
/dobot_left/dobot_bringup_ros2/srv/EmergencyStop
/dobot_left/dobot_bringup_ros2/srv/EnableFTSensor
/dobot_left/dobot_bringup_ros2/srv/EnableRobot
/dobot_left/dobot_bringup_ros2/srv/EnableSafeSkin
/dobot_left/dobot_bringup_ros2/srv/EndRTOffset
/dobot_left/dobot_bringup_ros2/srv/FCCollisionSwitch
/dobot_left/dobot_bringup_ros2/srv/FCForceMode
/dobot_left/dobot_bringup_ros2/srv/FCOff
/dobot_left/dobot_bringup_ros2/srv/FCSetDamping
/dobot_left/dobot_bringup_ros2/srv/FCSetDeviation
/dobot_left/dobot_bringup_ros2/srv/FCSetForce
/dobot_left/dobot_bringup_ros2/srv/FCSetForceLimit
/dobot_left/dobot_bringup_ros2/srv/FCSetForceSpeedLimit
/dobot_left/dobot_bringup_ros2/srv/FCSetMass
/dobot_left/dobot_bringup_ros2/srv/FCSetStiffness
/dobot_left/dobot_bringup_ros2/srv/ForceDriveMode
/dobot_left/dobot_bringup_ros2/srv/ForceDriveSpeed
/dobot_left/dobot_bringup_ros2/srv/GetAO
/dobot_left/dobot_bringup_ros2/srv/GetAngle
/dobot_left/dobot_bringup_ros2/srv/GetCoils
/dobot_left/dobot_bringup_ros2/srv/GetCurrentCommandId
/dobot_left/dobot_bringup_ros2/srv/GetDO
/dobot_left/dobot_bringup_ros2/srv/GetDOGroup
/dobot_left/dobot_bringup_ros2/srv/GetDOGroupDEC
/dobot_left/dobot_bringup_ros2/srv/GetError
/dobot_left/dobot_bringup_ros2/srv/GetErrorID
/dobot_left/dobot_bringup_ros2/srv/GetForce
/dobot_left/dobot_bringup_ros2/srv/GetHoldRegs
/dobot_left/dobot_bringup_ros2/srv/GetInBits
/dobot_left/dobot_bringup_ros2/srv/GetInRegs
/dobot_left/dobot_bringup_ros2/srv/GetInputBool
/dobot_left/dobot_bringup_ros2/srv/GetInputFloat
/dobot_left/dobot_bringup_ros2/srv/GetInputInt
/dobot_left/dobot_bringup_ros2/srv/GetOutputBool
/dobot_left/dobot_bringup_ros2/srv/GetOutputFloat
/dobot_left/dobot_bringup_ros2/srv/GetOutputInt
/dobot_left/dobot_bringup_ros2/srv/GetPose
/dobot_left/dobot_bringup_ros2/srv/GetStartPose
/dobot_left/dobot_bringup_ros2/srv/GetToolDO
/dobot_left/dobot_bringup_ros2/srv/InverseKin
/dobot_left/dobot_bringup_ros2/srv/ModbusClose
/dobot_left/dobot_bringup_ros2/srv/ModbusCreate
/dobot_left/dobot_bringup_ros2/srv/ModbusRTUCreate
/dobot_left/dobot_bringup_ros2/srv/MovJ
/dobot_left/dobot_bringup_ros2/srv/MovJIO
/dobot_left/dobot_bringup_ros2/srv/MovL
/dobot_left/dobot_bringup_ros2/srv/MovLIO
/dobot_left/dobot_bringup_ros2/srv/MoveJog
/dobot_left/dobot_bringup_ros2/srv/Pause
/dobot_left/dobot_bringup_ros2/srv/PositiveKin
/dobot_left/dobot_bringup_ros2/srv/PowerOn
/dobot_left/dobot_bringup_ros2/srv/RelJointMovJ
/dobot_left/dobot_bringup_ros2/srv/RelMovJUser
/dobot_left/dobot_bringup_ros2/srv/RelMovLTool
/dobot_left/dobot_bringup_ros2/srv/RelMovLUser
/dobot_left/dobot_bringup_ros2/srv/RequestControl
/dobot_left/dobot_bringup_ros2/srv/ResetRobot
/dobot_left/dobot_bringup_ros2/srv/RobotMode
/dobot_left/dobot_bringup_ros2/srv/RunScript
/dobot_left/dobot_bringup_ros2/srv/RunTo
/dobot_left/dobot_bringup_ros2/srv/ServoJ
/dobot_left/dobot_bringup_ros2/srv/ServoP
/dobot_left/dobot_bringup_ros2/srv/SetBackDistance
/dobot_left/dobot_bringup_ros2/srv/SetCoils
/dobot_left/dobot_bringup_ros2/srv/SetCollisionLevel
/dobot_left/dobot_bringup_ros2/srv/SetFCCollision
/dobot_left/dobot_bringup_ros2/srv/SetHoldRegs
/dobot_left/dobot_bringup_ros2/srv/SetOutputBool
/dobot_left/dobot_bringup_ros2/srv/SetOutputFloat
/dobot_left/dobot_bringup_ros2/srv/SetOutputInt
/dobot_left/dobot_bringup_ros2/srv/SetPayload
/dobot_left/dobot_bringup_ros2/srv/SetPostCollisionMode
/dobot_left/dobot_bringup_ros2/srv/SetSafeSkin
/dobot_left/dobot_bringup_ros2/srv/SetSafeWallEnable
/dobot_left/dobot_bringup_ros2/srv/SetTool
/dobot_left/dobot_bringup_ros2/srv/SetTool485
/dobot_left/dobot_bringup_ros2/srv/SetToolMode
/dobot_left/dobot_bringup_ros2/srv/SetToolPower
/dobot_left/dobot_bringup_ros2/srv/SetUser
/dobot_left/dobot_bringup_ros2/srv/SetWorkZoneEnable
/dobot_left/dobot_bringup_ros2/srv/SixForceHome
/dobot_left/dobot_bringup_ros2/srv/SpeedFactor
/dobot_left/dobot_bringup_ros2/srv/StartDrag
/dobot_left/dobot_bringup_ros2/srv/StartPatht
/dobot_left/dobot_bringup_ros2/srv/StartRTOffset
/dobot_left/dobot_bringup_ros2/srv/Stop
/dobot_left/dobot_bringup_ros2/srv/StopDrag
/dobot_left/dobot_bringup_ros2/srv/StopMoveJog
/dobot_left/dobot_bringup_ros2/srv/Tool
/dobot_left/dobot_bringup_ros2/srv/ToolAI
/dobot_left/dobot_bringup_ros2/srv/ToolDI
/dobot_left/dobot_bringup_ros2/srv/ToolDO
/dobot_left/dobot_bringup_ros2/srv/ToolDOInstant
/dobot_left/dobot_bringup_ros2/srv/User
/dobot_left/dobot_bringup_ros2/srv/VelJ
/dobot_left/dobot_bringup_ros2/srv/VelL
/dobot_left/get_parameter_types
/dobot_left/get_parameters
/dobot_left/list_parameters
/dobot_left/set_parameters
/dobot_left/set_parameters_atomically
