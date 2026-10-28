"""抓取脚本DO1控制及只读力话题自检；不连接机器人。"""
from types import SimpleNamespace as NS
import unittest

from test_input4_admittance_place import g, logger
from test_input7_admittance_place import nested_function


class ToolDOControlTests(unittest.TestCase):
    def test_each_arm_sends_do1_on_and_off_and_retains_load_state(self):
        for side in ('left', 'right'):
            for status in (0, 1):
                with self.subTest(side=side, status=status):
                    sent = {'left': [], 'right': []}
                    def client(name):
                        return NS(wait_for_service=lambda **kw: True,
                            call_async=lambda req: sent[name].append(req) or NS(result=lambda: NS(res=0)))
                    node = NS(sim_mode=False, _tool_load_state={'left': True, 'right': True},
                        get_logger=logger, _left_tool_cli=client('left'), _right_tool_cli=client('right'),
                        _spin_until=lambda *a: True, publish_error=lambda *a: self.fail(str(a)))
                    self.assertTrue(g.G01Demo.set_tool_do(node, side, status))
                    self.assertEqual(len(sent[side]), 1)
                    self.assertFalse(sent['right' if side == 'left' else 'left'])
                    request = sent[side][0]
                    self.assertIsInstance(request, g.ToolDOInstant.Request)
                    self.assertEqual((request.index, request.status), (1, status))
                    self.assertEqual(node._tool_load_state[side], None if status else False)
        self.assertEqual(g.LEFT_TOOL_COMMAND_SERVICE, '/g01/left/tool_do_instant')
        self.assertEqual(g.RIGHT_TOOL_COMMAND_SERVICE, '/g01/right/tool_do_instant')

    def test_do_failures_report_error_and_never_confirm_output(self):
        for failure in ('unavailable', 'timeout', 'rejected', 'exception'):
            with self.subTest(failure=failure):
                errors = []
                def call(request):
                    if failure == 'exception':
                        raise RuntimeError('transport failed')
                    return NS(result=lambda: NS(res=-2 if failure == 'rejected' else 0))
                node = NS(sim_mode=False, _tool_load_state={'left': True}, get_logger=logger,
                    _left_tool_cli=NS(wait_for_service=lambda **kw: failure != 'unavailable', call_async=call),
                    _spin_until=lambda *a: failure != 'timeout', publish_error=lambda *a: errors.append(a))
                self.assertFalse(g.G01Demo.set_tool_do(node, 'left', 0))
                self.assertIsNone(node._tool_load_state['left'])
                self.assertEqual(errors[0][0], 7)

    def test_simulation_and_invalid_arguments_do_not_send_commands(self):
        node = NS(sim_mode=True, _tool_load_state={'left': True}, get_logger=logger)
        self.assertTrue(g.G01Demo.set_tool_do(node, 'left', 0))
        self.assertFalse(node._tool_load_state['left'])
        self.assertFalse(g.G01Demo.set_tool_do(node, 'left', 2))
        self.assertFalse(g.G01Demo.set_tool_do(node, 'unknown', 1))

    def test_startup_selfcheck_only_reads_topics_and_joints(self):
        for enabled, missing in ((True, False), (True, True), (False, False)):
            with self.subTest(enabled=enabled, missing=missing):
                reads, joint_reads = [], []
                def read(side):
                    reads.append(side)
                    return None if missing and side == 'right' else 0.
                node = NS(use_force_sensors=enabled, get_logger=logger,
                    wait_for_ft_sensor_z=read, publish_error=lambda *a: None,
                    _get_joints=lambda *a, **kw: joint_reads.append(kw) or {'joint': .1},
                    set_tool_do=lambda *a: self.fail('selfcheck must not switch suction'))
                self.assertEqual(g.G01Demo.verify_startup_hardware(node), not missing)
                self.assertEqual(reads, ['left', 'right'] if enabled else [])
                self.assertEqual(len(joint_reads), 0 if missing else 1)

    def test_batch_sensor_check_retries_topic_reads_without_switching_outputs(self):
        for case in ('success', 'retry', 'missing', 'disabled'):
            with self.subTest(case=case):
                samples = {'left': [1., 1.], 'right': [2., 2.]}
                if case == 'retry':
                    samples['right'][0] = None
                if case == 'missing':
                    samples['right'] = [None, None]
                reads, errors = [], []
                def read(side):
                    reads.append(side)
                    return samples[side].pop(0)
                node = NS(use_force_sensors=case != 'disabled', wait_for_ft_sensor_z=read,
                    publish_error=lambda *a: errors.append(a),
                    set_tool_do=lambda *a: self.fail('sensor check must not switch suction'))
                check = nested_function('prepare_force_sensors_for_grasp_batch', node=node, log=logger())
                self.assertEqual(check(), case != 'missing')
                self.assertEqual(len(reads), 0 if case == 'disabled' else 2 if case == 'success' else 4)
                self.assertEqual(bool(errors), case == 'missing')


if __name__ == '__main__':
    unittest.main()
