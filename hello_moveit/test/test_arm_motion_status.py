"""运动前状态检查，不启动ROS节点或连接硬件。"""
from concurrent.futures import Future
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from test_input4_admittance_place import g, logger


class ArmMotionStatusTests(unittest.TestCase):
    def check(self, names, incoming, old=None, sim=False):
        clock = [10.]
        errors = []
        node = NS(sim_mode=sim, _arm_robot_status=dict(old or {}),
                  _abort_on_force_range=lambda: None,
                  get_logger=lambda: NS(error=errors.append))
        def spin(**kwargs):
            clock[0] += .1
            if incoming:
                for side, code in incoming.pop(0).items():
                    g.G01Demo._on_arm_robot_status(node, NS(data=code), side)
        node._executor = NS(spin_once=spin)
        with patch.object(g.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(g.rclpy, 'ok', return_value=True):
            result = g.G01Demo._check_arm_motion_ready(node, names)
        return result, errors

    def test_ready_requires_new_enabled_state_for_each_participating_arm(self):
        self.assertTrue(self.check(['l_arm_joint1'], [{'left': 1, 'right': 2}])[0])
        self.assertTrue(self.check(['r_arm_joint1'], [{'right': 1}])[0])
        self.assertTrue(self.check(['l_arm_joint1', 'r_arm_joint1'],
                                   [{'left': 1}, {'right': 1}])[0])
        for code in (0, 2, -1, 7):
            with self.subTest(code=code):
                self.assertFalse(self.check(['l_arm_joint1', 'r_arm_joint1'],
                                            [{'left': 1, 'right': code}])[0])

    def test_missing_stale_or_cached_state_cannot_authorize_motion(self):
        for old in ({}, {'left': (1, 9.9, 1)}, {'left': (1, 0., 1)}):
            result, errors = self.check(['l_arm_joint1'], [], old=old)
            self.assertFalse(result)
            self.assertTrue(errors)
        self.assertFalse(self.check(['l_arm_joint1', 'r_arm_joint1'], [{'left': 1}])[0])
        # 左臂新帧在等右臂期间过期，不能把两帧当作同时在线。
        self.assertFalse(self.check(['l_arm_joint1', 'r_arm_joint1'],
                                   [{'left': 1}]+[{}]*6+[{'right': 1}])[0])

    def test_simulation_and_body_only_do_not_require_arm_feedback(self):
        self.assertTrue(self.check(['l_arm_joint1'], [], sim=True)[0])
        self.assertTrue(self.check(['body_joint1', 'body_joint2'], [])[0])

    def test_direct_admittance_is_blocked_before_any_command(self):
        for group in ('left_arm', 'right_arm', 'dual_arm'):
            checked = []
            node = NS(
                use_force_sensors=True, get_logger=logger,
                _input4_wrench={}, _input4_wrench_time={}, _input4_joint_time={},
                _input4_description='test', _force_action_goals=[],
                _abort_on_force_range=lambda: None,
                _check_arm_motion_ready=lambda names: checked.append(names) or False,
                _input4_command_pub=NS(publish=lambda msg: self.fail('must not send any motion')),
            )
            def spin(**kwargs):
                for side in ('left', 'right'):
                    node._input4_wrench[side] = [0.]*4
                    node._input4_wrench_time[side] = 10.
                    node._input4_joint_time[side] = 10.
            node._executor = NS(spin_once=spin)
            with patch.object(g.time, 'monotonic', return_value=10.), \
                 patch.object(g.rclpy, 'ok', return_value=True):
                self.assertFalse(g.G01Demo._execute_input4_admittance(node, group))
            self.assertEqual(checked, [g.joint_names_for_group(group)])
            self.assertIsNone(node._input4_active_side)

    def test_goal_send_blocks_execution_but_allows_plan_only(self):
        for kind in ('trajectory', 'move', 'plan_only'):
            for ready in (True, False):
                events = []
                future = Future()
                node = NS(_abort_on_force_range=lambda: None, _force_action_goals=[],
                          _force_range_fault=None,
                          _check_arm_motion_ready=lambda names: events.append(('check', list(names))) or ready)
                client = NS(send_goal_async=lambda goal: events.append('send') or future)
                if kind == 'trajectory':
                    goal = g.ExecuteTrajectory.Goal()
                    goal.trajectory.joint_trajectory.joint_names = ['r_arm_joint1']
                else:
                    goal = g.MoveGroup.Goal()
                    goal.request.group_name = 'right_arm'
                    goal.planning_options.plan_only = kind == 'plan_only'
                result = g.G01Demo._send_force_guarded_goal(node, client, goal)
                allowed = ready or kind == 'plan_only'
                self.assertEqual('send' in events, allowed)
                self.assertIs(result, future if allowed else None)
                if kind == 'plan_only':
                    self.assertEqual(events, ['send'])


if __name__ == '__main__':
    unittest.main()
