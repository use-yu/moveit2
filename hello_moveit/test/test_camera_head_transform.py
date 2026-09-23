"""Run with sourced ROS/workspace: python3 -m unittest discover -s hello_moveit/test."""
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


g = load_module('grasp_head_test_subject', ROOT / 'hello_moveit/scripts/grasp_test.py')
c = load_module('comm_head_test_subject', ROOT / 'comm.py')


class CameraTransformTest(unittest.TestCase):
    def assertMatrixAlmostEqual(self, actual, expected):
        for row_a, row_b in zip(actual, expected):
            for a, b in zip(row_a, row_b):
                self.assertAlmostEqual(a, b, places=9)

    def test_calibration_angle_recovers_both_original_extrinsics(self):
        reference = g.make_pose(.12, -.23, .34, .1, -.2, math.pi / 4)
        for original in (g.VISION_RIGHT_TRANSFORM_MM, g.VISION_LEFT_TRANSFORM_MM):
            mount = g.camera_mount_transform(original, reference)
            actual = g.camera_transform_at_head_position(mount, reference)
            self.assertMatrixAlmostEqual(actual, original)

    def test_rotation_moves_camera_about_head_pivot(self):
        mount = g.xyz_rpy_to_matrix([.1, 0, .2, 0, 0, 0])
        head = g.make_pose(1., 2., 3., 0., 0., math.pi / 2)
        actual = g.camera_transform_at_head_position(mount, head)
        expected = g.xyz_rpy_to_matrix([1000., 2100., 3200., 0, 0, math.pi / 2])
        self.assertMatrixAlmostEqual(actual, expected)

    def test_same_world_object_at_different_head_and_body_positions(self):
        mount = g.xyz_rpy_to_matrix([.08, -.03, .12, .1, -.2, .3])
        world_object = g.xyz_rpy_to_matrix([1., .2, .4, .3, -.4, .2])
        for head_angle, waist_angle, lift in ((math.pi / 4, 0., 0.), (.2854, .7, .1)):
            world_arm = g.xyz_rpy_to_matrix([.1, .2, lift, 0, waist_angle, 0])
            arm_head = g.make_pose(.05, .1, .3, 0, 0, head_angle)
            # Generate the camera observation from independently composed physical poses.
            world_camera = g.matmul4(world_arm, g.matmul4(g.pose_to_matrix(arm_head), mount))
            camera_object = g.matmul4(g.invert_transform4(world_camera), world_object)
            arm_camera_mm = g.camera_transform_at_head_position(mount, arm_head)
            arm_camera = [row[:] for row in arm_camera_mm]
            for i in range(3):
                arm_camera[i][3] *= .001
            recovered = g.matmul4(world_arm, g.matmul4(arm_camera, camera_object))
            self.assertMatrixAlmostEqual(recovered, world_object)

    def test_fixed_head_model_is_rejected(self):
        errors = []
        node = SimpleNamespace(
            _camera_mount_transforms={},
            _get_link_pose_fk=lambda *a, **kw: g.make_pose(0., 0., 0.),
            get_logger=lambda: SimpleNamespace(error=errors.append),
        )
        result = g.G01Demo.camera_transforms_for_joints(node, {'head_joint': .2854})
        self.assertIsNone(result)
        self.assertTrue(errors)


class HeadFeedbackTest(unittest.TestCase):
    def test_head_is_required_and_stale_or_invalid_feedback_cannot_refresh_it(self):
        published = []
        now = SimpleNamespace(nanoseconds=1_000_000_000, to_msg=lambda: c.JointState().header.stamp)
        node = SimpleNamespace(
            _pos=dict.fromkeys(c.JOINT_ORDER, 0.),
            _left_rx=True, _right_rx=True, _lift_rx=True, _waist_rx=True, _head_rx=False,
            _left_last_ns=now.nanoseconds, _right_last_ns=now.nanoseconds,
            _lift_last_ns=now.nanoseconds, _waist_last_ns=now.nanoseconds, _head_last_ns=0,
            _part_timeout_sec=.5, get_clock=lambda: SimpleNamespace(now=lambda: now),
            _state_pub=SimpleNamespace(publish=published.append),
        )
        node._touch_group = lambda group: c.G01Comm._touch_group(node, group)
        c.G01Comm._publish_joint_states(node)
        self.assertFalse(published)
        c.G01Comm._on_head_state(node, SimpleNamespace(position=.2854))
        c.G01Comm._publish_joint_states(node)
        self.assertEqual(published[-1].position[published[-1].name.index('head_joint')], .2854)
        self.assertEqual(len(published[-1].name), 17)
        now.nanoseconds += 600_000_000
        for part in ('left', 'right', 'lift', 'waist'):
            setattr(node, f'_{part}_last_ns', now.nanoseconds)
        c.G01Comm._on_head_state(node, SimpleNamespace(position=float('nan')))
        c.G01Comm._publish_joint_states(node)
        self.assertEqual(len(published), 1)
        self.assertEqual(len(c.LEFT_JOINTS), 6)
        self.assertEqual(len(c.RIGHT_JOINTS), 6)


if __name__ == '__main__':
    unittest.main()
