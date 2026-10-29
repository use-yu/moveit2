"""输入 4 的物体姿态上限应在 IK 和路径规划前生效。"""
import math
import unittest
from types import SimpleNamespace

from test_input4_admittance_place import g, logger


class GraspTiltLimitTests(unittest.TestCase):
    def test_above_50_degrees_is_cached_and_skipped_before_planning(self):
        point = {
            "sj": (0.7, 0.1, 0.5, 0.0, 0.0, 0.0),
            "right_body": (0.7, 0.1, 0.6, math.pi - math.radians(51), 0.0, 0.0),
        }
        node = SimpleNamespace(get_logger=logger)
        failed_points = []

        self.assertIsNone(g.validate_reachable_grasp(node, [point], ungraspable_points=failed_points))
        self.assertEqual(failed_points, [(0.7, 0.1, 0.5)])
        self.assertIsNone(g.validate_reachable_grasp(node, [point], ungraspable_points=failed_points))
        self.assertEqual(len(failed_points), 1)

    def test_exactly_50_degrees_is_within_limit(self):
        point = {"right_body": (0.0, 0.0, 0.0, math.pi - math.radians(50), 0.0, 0.0)}
        tilt_rad, _ = g.grasp_point_tilt_from_base_minus_z(point)
        self.assertLessEqual(tilt_rad, math.radians(g.GRASP_MAX_GRASP_TILT_DEG) + 1e-12)


if __name__ == "__main__":
    unittest.main()
