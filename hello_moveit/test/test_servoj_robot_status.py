"""状态反馈测试：只加载 AST 定义，不执行驱动顶层的实机连接。"""
import ast
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[2] / 'servoj_30ms.py'
tree = ast.parse(SOURCE.read_text())
namespace = {
    'np': np, 'Node': object, 'threading': threading,
    'time': SimpleNamespace(monotonic=lambda: 10.),
    'Int32': lambda **kwargs: SimpleNamespace(**kwargs),
    'stop': True,
}
selected = []
for item in tree.body:
    if isinstance(item, ast.ClassDef) and item.name in ('fankuis', 'ArmRosBridge'):
        selected.append(item)
    elif isinstance(item, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id in
        ('MyType', 'ROBOT_MODE_NAMES', 'ROBOT_STATUS_TIMEOUT_SEC')
        for target in item.targets
    ):
        selected.append(item)
exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), 'exec'), namespace)
Bridge = namespace['ArmRosBridge']


class RobotStatusTests(unittest.TestCase):
    def setUp(self):
        self.messages = {'left': [], 'right': []}
        self.node = SimpleNamespace(
            _robot_status_lock=threading.Lock(),
            _robot_status_samples={'left': None, 'right': None},
            _robot_status_pubs={side: SimpleNamespace(publish=messages.append)
                                for side, messages in self.messages.items()},
        )

    def status(self, side):
        return self.messages[side][-1].data

    def test_valid_feedback_extracts_mode_and_error_without_changing_existing_fields(self):
        frame = np.zeros(1, dtype=namespace['MyType'])
        frame['robot_mode'] = 7
        frame['error_status'] = 1
        frame['enable_status'] = 1
        frame['q_actual'] = [1., 2., 3., 4., 5., 6.]
        reader = SimpleNamespace(_read_latest_frame=lambda: frame.tobytes())
        result = namespace['fankuis'].feed(reader)
        self.assertEqual(result[7:], [7, 1, 1])
        np.testing.assert_array_equal(result[1], [1., 2., 3., 4., 5., 6.])

    def test_startup_and_each_arm_timeout_are_independent(self):
        Bridge._publish_robot_status(self.node)
        self.assertEqual([self.status(s) for s in ('left', 'right')], [0, 0])
        Bridge.update_robot_status(self.node, 'left', 5, 0, 1)
        Bridge.update_robot_status(self.node, 'right', 7, 0, 1)
        self.node._robot_status_samples['right'] = (9.49, 7, 0, 1)
        Bridge._publish_robot_status(self.node)
        self.assertEqual([self.status(s) for s in ('left', 'right')], [1, 0])
        Bridge.update_robot_status(self.node, 'right', 9, 1, 1)
        Bridge._publish_robot_status(self.node)
        self.assertEqual(self.status('right'), 2)

    def test_alarm_has_priority_over_enable_status(self):
        for mode in range(1, 12):
            for enabled in (0, 1):
                for error in (0, 1):
                    with self.subTest(mode=mode, enabled=enabled, error=error):
                        Bridge.update_robot_status(self.node, 'left', mode, error, enabled)
                        Bridge._publish_robot_status(self.node)
                        expected = 2 if error or mode in (9, 11) else enabled
                        self.assertEqual(self.status('left'), expected)

    def test_feedback_timeout_and_driver_stop(self):
        Bridge.update_robot_status(self.node, 'left', 5, 0, 1)
        with patch.object(namespace['time'], 'monotonic', return_value=10.5):
            Bridge._publish_robot_status(self.node)
            self.assertEqual(self.status('left'), 1)
        with patch.object(namespace['time'], 'monotonic', return_value=10.501):
            Bridge._publish_robot_status(self.node)
            self.assertEqual(self.status('left'), 0)
        with patch.dict(namespace, stop=False):
            Bridge._publish_robot_status(self.node)
            self.assertEqual(self.status('left'), 0)


if __name__ == '__main__':
    unittest.main()
