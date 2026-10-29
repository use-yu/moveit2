import unittest
from types import SimpleNamespace
from unittest.mock import patch

from test_input4_admittance_place import g, logger


class Input4ForceRecoveryTests(unittest.TestCase):
    def test_force_limit_logs_without_stopping_or_canceling_motion(self):
        errors = []
        node = SimpleNamespace(
            use_force_sensors=True, _ft_sensor_z={'left': None},
            _ft_sensor_count={'left': 0}, _force_range_enabled=True,
            _force_range_fault=None, _force_task_active=True,
            _force_range_warning_active=set(), _force_range_samples={},
            _active_approach_guard=None,
            get_logger=lambda: SimpleNamespace(error=errors.append),
            _send_servoj_control=lambda *args: self.fail('unexpected stop command'),
            _motor_command_pub=SimpleNamespace(publish=lambda *args: self.fail('unexpected motor command')),
            _force_action_goals=[SimpleNamespace(cancel_goal_async=lambda: self.fail('unexpected cancellation'))],
        )
        overloaded = SimpleNamespace(data=[0., 0., -185.6, 0., 0., 0.])
        with patch('builtins.print', side_effect=lambda message: errors.append(message)):
            g.G01Demo._on_ft_sensor(node, overloaded, 'left')
            g.G01Demo._on_ft_sensor(node, overloaded, 'left')
            g.G01Demo._on_ft_sensor(node, SimpleNamespace(data=[0.]*6), 'left')
            g.G01Demo._on_ft_sensor(node, overloaded, 'left')
        self.assertEqual(len(errors), 2)
        self.assertIsNone(node._force_range_fault)
        self.assertTrue(all(message.startswith(g.RED) and message.endswith(g.RESET) for message in errors))
        self.assertEqual(node._ft_sensor_count['left'], 4)

    def test_approach_force_drop_only_prints_without_stopping(self):
        errors = []
        guard = {
            'monitoring': True, 'sides': ('left',), 'baseline_force_z_by_side': {'left': 0.},
            'progress_fraction': 0., 'total_distance': .1,
            'near_force_drop_threshold': 60., 'stage_label': '直线抓取',
            'baseline_label': 'q_pre', 'contact': False,
        }
        node = SimpleNamespace(
            use_force_sensors=True, _ft_sensor_z={'left': None},
            _ft_sensor_count={'left': 0}, _force_range_enabled=False,
            _active_approach_guard=guard,
            _send_servoj_control=lambda *args: self.fail('unexpected stop command'),
            _motor_command_pub=SimpleNamespace(publish=lambda *args: self.fail('unexpected motor command')),
        )
        with patch('builtins.print', side_effect=lambda message: errors.append(message)):
            for force_z in (-70., -70., 0., -220.):
                g.G01Demo._on_ft_sensor(node, SimpleNamespace(data=[0., 0., force_z]), 'left')
        self.assertEqual(len(errors), 2)
        self.assertFalse(guard['contact'])
        self.assertTrue(all(message.startswith(g.RED) and message.endswith(g.RESET) for message in errors))

    def test_force_and_torque_limits(self):
        for axis in range(6):
            limit = 180. if axis < 3 else 10.
            for sign in (-1., 1.):
                values = [0.] * 6
                values[axis] = sign*(limit-.001)
                self.assertIsNone(g.ft_range_violation(values))
                values[axis] = sign*limit
                self.assertIsNotNone(g.ft_range_violation(values))

    def test_retreat_ignores_force_and_torque_limits_but_rejects_invalid_data(self):
        self.assertIsNone(g.ft_range_violation([0., 0., 0., 11., 0., 0.], ignore_limits=True))
        self.assertIsNone(g.ft_range_violation([180., 0., 0., 11., 0., 0.], ignore_limits=True))
        self.assertIsNotNone(g.ft_range_violation([0., 0., 0., float('nan'), 0., 0.], ignore_limits=True))
        self.assertIsNotNone(g.ft_range_violation([0.] * 3, ignore_limits=True))

    def test_retreat_restores_force_and_torque_guard_on_success_and_failure(self):
        for result in (True, False):
            events = []
            node = SimpleNamespace(get_logger=logger)
            node._recover_input4_force_stop = lambda: events.append(node._input4_retreat_ignore_limits) or True
            node._return_to_pre_after_approach_contact = lambda *args: events.append(node._input4_retreat_ignore_limits) or result
            self.assertEqual(g.G01Demo._continue_input4_retreat_after_force_stop(
                node, 'right_arm', 'r_tool', object(), [], 'base', .1), result)
            self.assertEqual(events, [True, True])
            self.assertFalse(node._input4_retreat_ignore_limits)

    def test_resume_requires_confirmed_stop_but_does_not_wait_for_force_samples(self):
        for case in ('safe', 'overload', 'stale', 'invalid', 'stop_failed', 'resume_failed'):
            with self.subTest(case=case):
                clock = SimpleNamespace(now=1.)
                commands = []
                node = SimpleNamespace(
                    _force_task_active=True, _force_range_fault='overload',
                    _force_range_samples=(
                        {'left': (0., [0., 0., float('nan'), 0., 0., 0.])}
                        if case == 'invalid' else {}
                    ), get_logger=logger,
                    stop_upper_grasp_motion=lambda: case != 'stop_failed',
                    _send_servoj_control=lambda side, command: commands.append((side, command)) or 1,
                    _wait_for_servoj_control_state=lambda *args: case != 'resume_failed')

                def spin_once(timeout_sec):
                    clock.now += timeout_sec
                    if case != 'stale':
                        node._force_range_samples = {
                            side: (clock.now, [0., 0., 181. if case == 'overload' else 10., 0., 0., 0.])
                            for side in ('left', 'right')}

                node._executor = SimpleNamespace(spin_once=spin_once)
                with patch.object(g.time, 'monotonic', lambda: clock.now), \
                     patch.object(g.rclpy, 'ok', return_value=True):
                    result = g.G01Demo._recover_input4_force_stop(node)
                self.assertEqual(result, case not in ('invalid', 'stop_failed', 'resume_failed'))
                self.assertTrue(node._force_task_active)
                self.assertFalse(node._input4_retreat_ignore_limits)
                self.assertEqual(clock.now, 1.)
                if case in ('invalid', 'stop_failed'):
                    self.assertEqual(commands, [])
                    self.assertEqual(node._force_range_fault, 'overload')
                else:
                    self.assertEqual(commands, [('left', 'resume'), ('right', 'resume')])

    def test_upper_q_pre_motion_alarm_returns_recovery_reason(self):
        names = list(g.JOINT_TARGETS['right_arm'])
        trajectory = g.RobotTrajectory()
        trajectory.joint_trajectory.joint_names = names
        trajectory.joint_trajectory.points.append(
            g.JointTrajectoryPoint(positions=[0.0] * len(names))
        )
        q_pre = {name: 0.0 for name in names}
        node = SimpleNamespace(
            get_logger=logger,
            last_pick_failure_reason=None,
            _upper_grasp_active=True,
            _wait_for_driver_signal=lambda: True,
            _select_available_place_target=lambda *args: (True, 'sw1'),
            set_tool_do=lambda *args: True,
            _get_joints=lambda requested, **kwargs: {name: 0.0 for name in requested},
            _make_grasp_transfer_obstacles=lambda *args: [],
            _execute_traj=lambda *args: (_ for _ in ()).throw(
                g.UpperGraspMotionFailure('arm alarm')
            ),
        )
        result = g.G01Demo.pick_and_return(
            node,
            target_pose=g.make_pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            speed_scale=0.4,
            group='right_arm',
            link='r_tool',
            plan_frame=g.SCENE_FRAME,
            joint_names=names,
            first_return_mode=2,
            prevalidated_pair=(q_pre, q_pre, trajectory, trajectory),
        )
        self.assertFalse(result)
        self.assertEqual(node.last_pick_failure_reason, 'q_pre')

    def test_upper_empty_arm_alarm_recovery_clears_and_returns_to_q1(self):
        names = g.joint_names_for_group('dual_arm_body')
        joint_state = {name: 0.0 for name in names}
        cleared = []

        def move_to_q1(group, speed, joint_names, waypoints, **kwargs):
            self.assertTrue(node._upper_grasp_recovering)
            self.assertEqual(group, 'dual_arm_body')
            joint_state.update(dict(zip(joint_names, waypoints[0])))
            return True

        node = SimpleNamespace(
            get_logger=logger,
            _tool_load_state={'left': False, 'right': False},
            _upper_grasp_recovering=False,
            stop_upper_grasp_motion=lambda: True,
            clear_arm_alarm=lambda side: cleared.append(side) or True,
            _get_joints=lambda requested, **kwargs: {
                name: joint_state[name] for name in requested
            },
            _send_servoj_control=lambda *args: 1,
            _wait_for_servoj_control_state=lambda *args: True,
            plan_execute_joint_waypoints=move_to_q1,
        )
        self.assertTrue(g.G01Demo.reset_empty_arms_for_upper_grasp(node, clear_alarms=True))
        self.assertEqual(cleared, ['left', 'right'])
        self.assertFalse(node._upper_grasp_recovering)


if __name__ == '__main__':
    unittest.main()
