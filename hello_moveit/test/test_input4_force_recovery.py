import unittest
from types import SimpleNamespace
from unittest.mock import patch

from test_input4_admittance_place import g, logger


class Input4ForceRecoveryTests(unittest.TestCase):
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

    def test_resume_requires_confirmed_stop_and_new_safe_samples(self):
        for case in ('safe', 'overload', 'stale', 'stop_failed', 'resume_failed'):
            with self.subTest(case=case):
                clock = SimpleNamespace(now=1.)
                commands = []
                node = SimpleNamespace(
                    _force_task_active=True, _force_range_fault='overload',
                    _force_range_samples={}, get_logger=logger,
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
                self.assertEqual(result, case == 'safe')
                self.assertTrue(node._force_task_active)
                if case in ('overload', 'stale', 'stop_failed'):
                    self.assertEqual(commands, [])
                    self.assertEqual(node._force_range_fault, 'overload')


if __name__ == '__main__':
    unittest.main()
