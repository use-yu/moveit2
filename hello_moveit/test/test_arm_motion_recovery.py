"""运行中状态异常停止、回车恢复、阶段续跑和螺旋时钟回归，不连接实机。"""
from concurrent.futures import Future
from types import MethodType, SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
from test_input4_admittance_place import g, logger, trajectory


def bind(node, *names):
    for name in names:
        setattr(node, name, MethodType(getattr(g.G01Demo, name), node))
    return node


class RuntimeRecoveryTests(unittest.TestCase):
    def make_monitor(self):
        events = []
        node = NS(sim_mode=False, _arm_motion_names=('r_arm_joint1',),
                  _arm_motion_fault=None, _arm_motion_recovering=False,
                  _arm_robot_status={'left': (1, 10., 1), 'right': (1, 10., 1)},
                  get_logger=logger,
                  _motor_command_pub=NS(publish=lambda msg: events.append(('motor', msg.data))),
                  _send_servoj_control=lambda side, cmd: events.append((side, cmd)),
                  _force_action_goals=[NS(cancel_goal_async=lambda: events.append('cancel'))])
        bind(node, '_poll_arm_motion_status', '_trip_arm_motion', '_raise_arm_motion_fault')
        return node, events

    def test_either_arm_fault_or_stale_feedback_stops_once(self):
        for side in ('left', 'right'):
            for code, stamp in ((0, 10.), (2, 10.), (-1, 10.), (1, 9.)):
                node, events = self.make_monitor()
                node._arm_robot_status[side] = (code, stamp, 2)
                with patch.object(g.time, 'monotonic', return_value=10.):
                    node._poll_arm_motion_status()
                    node._poll_arm_motion_status()
                    with self.assertRaises(g.ArmMotionStatusFault):
                        node._raise_arm_motion_fault()
                self.assertEqual(events, [('motor', 1), ('left', 'stop'), ('right', 'stop'), 'cancel'])

    def test_idle_simulation_and_recovery_do_not_retrip(self):
        for field, value in (('sim_mode', True), ('_arm_motion_names', ()),
                             ('_arm_motion_recovering', True)):
            node, events = self.make_monitor()
            node._arm_robot_status['left'] = (2, 10., 2)
            setattr(node, field, value)
            with patch.object(g.time, 'monotonic', return_value=10.):
                node._poll_arm_motion_status()
            self.assertEqual(events, [])

    def test_pending_goal_accepted_after_fault_is_cancelled(self):
        accepted, completed = Future(), Future()
        events = []
        node = NS(_abort_on_force_range=lambda: None, _check_arm_motion_ready=lambda names: True,
                  _raise_arm_motion_fault=lambda: None, _arm_motion_pending_goals=[],
                  _force_action_goals=[], _force_range_fault=None, _arm_motion_fault=None)
        goal = g.ExecuteTrajectory.Goal()
        goal.trajectory.joint_trajectory.joint_names = ['r_arm_joint1']
        g.G01Demo._send_force_guarded_goal(node, NS(send_goal_async=lambda goal: accepted), goal)
        self.assertEqual(node._arm_motion_pending_goals, [accepted])
        node._arm_motion_fault = 'alarm'
        accepted.set_result(NS(accepted=True, get_result_async=lambda: completed,
                               cancel_goal_async=lambda: events.append('cancel')))
        self.assertEqual(events, ['cancel'])
        self.assertEqual(node._arm_motion_pending_goals, [])
        completed.set_result(None)
        self.assertEqual(node._force_action_goals, [])

    def test_recovery_requires_enter_then_clear_then_enabled_then_resume(self):
        events = []
        ready = iter((False, True))
        node = NS(_arm_motion_fault='alarm', _arm_motion_pending_goals=[], get_logger=logger,
                  _motor_command_pub=NS(publish=lambda msg: events.append('motor')),
                  stop_upper_grasp_motion=lambda: events.append('stopped') or True,
                  clear_arm_alarms_for_recovery=lambda *args: events.append('clear') or True,
                  _check_arm_motion_ready=lambda names: events.append('status') or next(ready),
                  _send_servoj_control=lambda side, cmd: events.append((side, cmd)) or 1,
                  _wait_for_servoj_control_state=lambda *args: True)
        with patch.object(g.rclpy, 'ok', return_value=True), \
             patch('builtins.input', side_effect=lambda prompt: events.append('enter') or ''):
            g.G01Demo._recover_arm_motion_interactively(node)
        self.assertEqual(events, ['motor', 'stopped', 'motor', 'enter', 'clear', 'status',
                                  'motor', 'stopped', 'motor', 'enter', 'clear', 'status',
                                  ('left', 'resume'), ('right', 'resume')])
        self.assertIsNone(node._arm_motion_fault)
        self.assertFalse(node._arm_motion_recovering)

    def test_no_resume_when_old_goal_cannot_be_confirmed_or_input_is_closed(self):
        events = []
        pending = Future()
        node = NS(_arm_motion_fault='alarm', _arm_motion_pending_goals=[pending],
                  _motor_command_pub=NS(publish=lambda msg: None), get_logger=logger,
                  _spin_until=lambda *args: False, stop_upper_grasp_motion=lambda: True,
                  clear_arm_alarms_for_recovery=lambda *args: self.fail('must not clear'),
                  _send_servoj_control=lambda *args: self.fail('must not resume'))
        with patch.object(g.rclpy, 'ok', return_value=True), \
             patch('builtins.input', side_effect=['', EOFError()]):
            with self.assertRaises(KeyboardInterrupt):
                g.G01Demo._recover_arm_motion_interactively(node)
        self.assertEqual(node._arm_motion_fault, 'alarm')

    def test_trajectory_replans_original_target_after_each_fault(self):
        original, replacement = object(), object()
        seen, replans = [], []
        def execute(traj, **kwargs):
            seen.append(traj)
            if len(seen) < 3:
                raise g.ArmMotionStatusFault('alarm')
            return True
        node = NS(_execute_traj_impl=execute, _recover_arm_motion_interactively=lambda: None,
                  _replan_interrupted_trajectory=lambda old: replans.append(old) or replacement,
                  _raise_arm_motion_fault=lambda: None,
                  _upper_grasp_active=False, _upper_place_active=False)
        self.assertTrue(g.G01Demo._execute_traj(node, original))
        self.assertEqual(seen, [original, replacement, replacement])
        self.assertEqual(replans, [original, original])

    def test_replan_uses_measured_full_state_and_original_endpoint(self):
        traj = trajectory(g.joint_names_for_group('right_arm'), end=[.2]*6)
        current = dict.fromkeys(g.JOINT_TARGETS['robot'], .1)
        seen = []
        node = NS(get_logger=logger, _get_joints=lambda *args, **kwargs: current,
                  plan_joint_motion=lambda *args, **kwargs: seen.append((args, kwargs)) or (True, 0., 'new'))
        self.assertEqual(g.G01Demo._replan_interrupted_trajectory(node, traj), 'new')
        args, kwargs = seen[0]
        self.assertEqual(args[0], 'right_arm')
        self.assertEqual(list(args[1].values()), [.2]*6)
        self.assertIs(kwargs['start_joints'], current)
        self.assertFalse(kwargs['execute'])

    def test_move_group_refreshes_start_preserving_constraints_and_obstacles(self):
        seen = []
        current, constraints, obstacles = object(), object(), object()
        def move(*args):
            seen.append(args)
            if len(seen) == 1:
                raise g.ArmMotionStatusFault('alarm')
            return True, 0., None, 1
        node = NS(_move_once_impl=move, _recover_arm_motion_interactively=lambda: None,
                  _get_joints=lambda *args, **kwargs: current, _raise_arm_motion_fault=lambda: None)
        result = g.G01Demo._move_once(node, 'right_arm', constraints, {}, False, .4,
                                     planning_obstacles=obstacles)
        self.assertTrue(result[0])
        self.assertIs(seen[1][1], constraints)
        self.assertIs(seen[1][2], current)
        self.assertIs(seen[1][-1], obstacles)

    def test_input4_approach_recovers_then_retreats_without_reapproach(self):
        calls = []
        def fail(*args, **kwargs):
            calls.append('approach')
            raise g.ArmMotionStatusFault('alarm')
        node = NS(_execute_approach_with_force_guard_impl=fail, get_logger=logger,
                  _recover_arm_motion_interactively=lambda: calls.append('recover'),
                  _replan_interrupted_trajectory=lambda *args: self.fail('must not reapproach'))
        result = g.G01Demo._execute_approach_with_force_guard(
            node, object(), absolute_force_delta_threshold=70.)
        self.assertEqual(result, 'contact')
        self.assertEqual(calls, ['approach', 'recover'])

    def test_admittance_loop_resumes_same_controller_including_spiral(self):
        clock = [10.]
        names = g.joint_names_for_group('right_arm')
        events = []
        model = NS(joint_names=names, evaluate=lambda q: (np.eye(4), np.eye(6)))
        controller = g.Input4PlacementControl(
            model, np.zeros(6), np.zeros(4), 10., xy_spiral=g.INPUT7_XY_SPIRAL)
        controller.contact_time = 10.
        controller.finished = False
        def step(q, wrench, now):
            events.append('step')
            controller.previous_time = now
            if events.count('step') == 2:
                controller.finished = True
                controller.finish_reason = 'duration'
                return None
            return np.zeros(6), np.zeros(6)
        controller.step = step
        polls = [0]
        def poll():
            polls[0] += 1
            if polls[0] == 2:
                node._arm_motion_fault = 'alarm'
                raise g.ArmMotionStatusFault('alarm')
        def recover():
            events.append('recover')
            clock[0] += 100.
            node._arm_motion_fault = None
        node = NS(get_logger=logger, use_force_sensors=True,
                  _input4_wrench={}, _input4_wrench_time={}, _input4_joint_time={},
                  _input4_description='test', _force_action_goals=[],
                  _input4_models={'right': model}, _joints=dict.fromkeys(names, 0.),
                  _force_range_fault=None, _arm_motion_fault=None,
                  _abort_on_force_range=lambda: None,
                  _check_arm_motion_ready=lambda names: True,
                  _raise_arm_motion_fault=poll, _recover_arm_motion_interactively=recover,
                  _get_joints=lambda *args, **kwargs: dict.fromkeys(names, .1),
                  get_clock=lambda: NS(now=lambda: NS(to_msg=lambda: g.RobotTrajectory().joint_trajectory.header.stamp)),
                  _input4_command_pub=NS(publish=lambda msg: events.append('publish')))
        def spin(**kwargs):
            clock[0] += g.INPUT4_ADMITTANCE['period']
            node._input4_wrench['right'] = np.zeros(4)
            node._input4_wrench_time['right'] = clock[0]
            node._input4_joint_time['right'] = clock[0]
        node._executor = NS(spin_once=spin)
        with patch.object(g.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(g.rclpy, 'ok', return_value=True), \
             patch.object(g, 'Input4PlacementControl', return_value=controller) as factory:
            self.assertTrue(g.G01Demo._execute_input4_admittance(node, 'right_arm'))
        self.assertEqual(factory.call_count, 1)  # 恢复不重建控制器或丢失螺旋
        self.assertEqual(events[:5], ['step', 'publish', 'recover', 'step', 'publish'])
        self.assertEqual(controller.xy_spiral, g.INPUT7_XY_SPIRAL)
        self.assertGreater(controller.contact_time, 110.)
        np.testing.assert_array_equal(controller.command, [.1]*6)

    def test_admittance_pause_preserves_phase_spiral_baseline_and_height_goal(self):
        model = NS(evaluate=lambda q: (np.eye(4), np.eye(6)))
        for contact_time in (None, 2.):
            control = g.Input4PlacementControl(model, np.zeros(6), np.array([1., 2., 3., 4.]),
                                               0., target_z=-.2, xy_spiral=g.INPUT7_XY_SPIRAL)
            control.contact_time = contact_time
            old_force = control.xy_target_force(4.)
            control.xyz.velocity[:] = 1.
            control.rz.velocity[:] = 1.
            control.resume_after_pause(np.full(6, .1), 4., 104.)
            np.testing.assert_allclose(control.xy_target_force(104.), old_force)
            self.assertEqual(control.contact_time, None if contact_time is None else 102.)
            self.assertEqual(control.started, 100.)
            self.assertEqual(control.previous_time, 104.)
            self.assertEqual(control.target_z, -.2)
            np.testing.assert_array_equal(control.baseline, [1., 2., 3., 4.])
            np.testing.assert_array_equal(control.command, [.1]*6)
            np.testing.assert_array_equal(control.xyz.velocity, [0.]*3)


if __name__ == '__main__':
    unittest.main()
