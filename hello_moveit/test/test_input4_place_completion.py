"""输入 4 只有在新 SW 信号确认四位有料后才能完成。"""
import unittest
from types import SimpleNamespace

from test_input4_admittance_place import g, logger


class Input4PlaceCompletionTests(unittest.TestCase):
    def make_node(self, frames):
        state = {'filled': set(), 'reads': 0}

        def wait_for_signal(*, require_new):
            self.assertTrue(require_new)
            state['reads'] += 1
            if not frames:
                return False
            state['filled'] = frames.pop(0)
            return True

        node = SimpleNamespace(
            get_logger=logger,
            _wait_for_driver_signal=wait_for_signal,
            _place_slot_has_material=lambda name: name in state['filled'],
        )
        return node, state

    def test_full_requires_two_new_full_frames(self):
        all_slots = set(g.PLACE_SLOT_MASKS)
        node, state = self.make_node([set(all_slots), set(all_slots)])
        self.assertEqual(
            g.G01Demo.confirm_all_place_slots_full(node),
            (True, all_slots),
        )
        self.assertEqual(state['reads'], 2)

    def test_late_empty_slot_prevents_completion(self):
        all_slots = set(g.PLACE_SLOT_MASKS)
        remaining = all_slots - {'sw4'}
        node, state = self.make_node([set(all_slots), remaining])
        self.assertEqual(
            g.G01Demo.confirm_all_place_slots_full(node),
            (False, remaining),
        )
        self.assertEqual(state['reads'], 2)

    def test_missing_new_signal_cannot_confirm_full(self):
        node, _ = self.make_node([])
        self.assertIsNone(g.G01Demo.confirm_all_place_slots_full(node))


if __name__ == '__main__':
    unittest.main()
