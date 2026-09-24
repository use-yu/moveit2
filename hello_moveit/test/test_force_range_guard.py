"""Force range boundary, abort, and pending-action cancellation regression tests."""
from concurrent.futures import Future
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location(
    'force_guard_subject', Path(__file__).resolve().parents[1] / 'scripts/grasp_test.py',
)
g = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = g
spec.loader.exec_module(g)


class ForceRangeTests(unittest.TestCase):
    def test_all_six_axes_positive_and_negative_boundaries(self):
        for axis in range(6):
            limit = 160. if axis < 3 else 6.4
            for sign in (-1., 1.):
                values = [0.] * 6
                values[axis] = sign * (limit - .001)
                self.assertIsNone(g.ft_range_violation(values))
                for magnitude in (limit, limit + 1.):
                    values[axis] = sign * magnitude
                    self.assertIsNotNone(g.ft_range_violation(values))

    def test_missing_and_nonfinite_values_are_unsafe(self):
        self.assertIsNotNone(g.ft_range_violation([0.] * 3))
        for value in (float('nan'), float('inf'), float('-inf')):
            self.assertIsNotNone(g.ft_range_violation([0., 0., 0., value, 0., 0.]))

    def make_node(self):
        self.events = []
        node = SimpleNamespace(
            _force_range_fault=None, _force_range_enabled=True,
            _force_task_active=True, _force_action_goals=[],
            _force_stop_requests={}, _ft_sensor_z={'left': None, 'right': None},
            get_logger=lambda: SimpleNamespace(error=lambda text: self.events.append('log')),
            _motor_command_pub=SimpleNamespace(publish=lambda msg: self.events.append(('motor', msg.data))),
            _send_servoj_control=lambda side, cmd: self.events.append((side, cmd)) or 1,
        )
        node._abort_on_force_range = lambda: g.G01Demo._abort_on_force_range(node)
        return node

    def test_trip_stops_both_arms_cancels_and_bypasses_recovery(self):
        node = self.make_node()
        node._force_action_goals.append(SimpleNamespace(cancel_goal_async=lambda: self.events.append('cancel')))
        with self.assertRaises(g.ForceRangeAbort):
            g.G01Demo._on_ft_sensor(node, SimpleNamespace(data=[0., -160., 0., 0., 0., 0.]), 'right')
        self.assertIn(('motor', 1), self.events)
        self.assertIn(('left', 'stop'), self.events)
        self.assertIn(('right', 'stop'), self.events)
        self.assertIn('cancel', self.events)
        self.assertFalse(issubclass(g.ForceRangeAbort, Exception))
        with self.assertRaises(g.ForceRangeAbort):
            g.G01Demo._send_force_guarded_goal(node, None, None)

    def test_late_goal_acceptance_is_cancelled_after_trip(self):
        node = self.make_node()
        accepted = Future()
        completed = Future()
        handle = SimpleNamespace(accepted=True, get_result_async=lambda: completed,
                                 cancel_goal_async=lambda: self.events.append('late cancel'))
        client = SimpleNamespace(send_goal_async=lambda goal: accepted)
        g.G01Demo._send_force_guarded_goal(node, client, object())
        node._force_range_fault = 'overload'
        accepted.set_result(handle)
        self.assertIn('late cancel', self.events)
        completed.set_result(None)
        self.assertFalse(node._force_action_goals)


if __name__ == '__main__':
    unittest.main()
