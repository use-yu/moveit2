"""输入4导纳放置与实际起点直退回归测试；不启动节点、不连接真机。"""
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import xacro

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/grasp_actor.py'
spec = importlib.util.spec_from_file_location('input4_actor_subject', SCRIPT)
g = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = g
spec.loader.exec_module(g)


def logger():
    return NS(info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None)


def trajectory(names, start=None, end=None):
    result = g.RobotTrajectory()
    result.joint_trajectory.joint_names = list(names)
    for positions in (start or [0.]*len(names), end or [0.]*len(names)):
        result.joint_trajectory.points.append(g.JointTrajectoryPoint(positions=list(positions)))
    return result


class DynamicsAndControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = SCRIPT.parents[2]
        cls.xml = xacro.process_file(str(root / 'moveit_resources/g01_description/urdf/G01-URDF888.urdf')).toxml()

    def test_both_arms_jacobian_matches_position_finite_difference(self):
        q = np.array([.2, -.7, .8, .6, .9, -.4])
        for side in ('left', 'right'):
            model = g.Input4ArmKinematics(self.xml, side)
            _, J = model.evaluate(q)
            for index in range(6):
                delta = np.zeros(6); delta[index] = 1e-6
                plus, _ = model.evaluate(q+delta)
                minus, _ = model.evaluate(q-delta)
                np.testing.assert_allclose(J[:3, index], (plus[:3, 3]-minus[:3, 3])/2e-6, atol=1e-8)
            self.assertEqual(model.joint_names, [f'{side[0]}_arm_joint{i}' for i in range(1, 7)])

    def test_two_stage_force_xyz_rz_and_completion(self):
        for side in ('left', 'right'):
            model = g.Input4ArmKinematics(self.xml, side)
            q = np.array([.2, -.7, .8, .6, .9, -.4])
            controller = g.Input4PlacementControl(model, q, [1., 2., 3., 1.], 1.)
            now = 1.
            for _ in range(10):
                now += .03
                q, _ = controller.step(q, [11., -8., 3., 1.5], now)
                np.testing.assert_array_equal(controller.xyz.x[:2], np.zeros(2))
                np.testing.assert_array_equal(controller.rz.x, np.zeros(1))
            np.testing.assert_allclose(controller.xyz.velocity[2], 100/5000, atol=1e-9)
            now += .03
            q, _ = controller.step(q, [11., -8., -28., 1.5], now)
            self.assertIsNotNone(controller.contact_time)
            self.assertGreater(controller.xyz.x[0], 0)
            self.assertLess(controller.xyz.x[1], 0)
            self.assertGreater(controller.rz.x[0], 0)
            for _ in range(10):
                now += .03
                q, _ = controller.step(q, [1., 2., -57., 1.], now)
            self.assertLess(abs(controller.xyz.velocity[2]), 1e-10)
            now = controller.contact_time+controller.config['contact_duration']
            self.assertIsNone(controller.step(q, [1., 2., -57., 1.], now))
            self.assertTrue(controller.finished)
            self.assertEqual(controller.finish_reason, 'duration')

    def test_measured_tool_crossing_place_height_stops_before_contact(self):
        model = g.Input4ArmKinematics(self.xml, 'right')
        start = np.array([.2, -.7, .8, .6, .9, -.4])
        end = start + np.array([.05, -.04, .03, 0., 0., 0.])
        base = np.eye(4)
        base[:3, :3] = g.input4_axis_rotation([0., 1., 0.], .6)
        base[:3, 3] = [1., 2., 3.]
        tool_offset = np.array([.01, .02, .15])
        def height(q):
            transform, _ = model.evaluate(q)
            return (base @ transform @ np.append(tool_offset, 1.))[2]
        goal = (height(start)+height(end))/2
        self.assertGreater(abs(height(start)-height(end)), 1e-4)
        for q0, q1 in ((start, end), (end, start)):
            controller = g.Input4PlacementControl(model, q0, np.zeros(4), 1.,
                target_z=goal, base_transform=base, tool_offset=tool_offset)
            self.assertIsNotNone(controller.step(q0, np.zeros(4), 1.03))
            self.assertFalse(controller.finished)
            self.assertIsNone(controller.step(q1, np.zeros(4), 1.06))
            self.assertTrue(controller.finished)
            self.assertEqual(controller.finish_reason, 'height')
            self.assertIsNone(controller.contact_time)
            np.testing.assert_allclose(controller.tool_height(model.evaluate(q1)[0]), height(q1))

    def test_initial_tool_already_at_place_height_stops_without_movement(self):
        model = g.Input4ArmKinematics(self.xml, 'right')
        q = np.array([.2, -.7, .8, .6, .9, -.4])
        transform, _ = model.evaluate(q)
        controller = g.Input4PlacementControl(model, q, np.zeros(4), 1.,
                                              target_z=float(transform[2, 3]))
        self.assertIsNone(controller.step(q, np.zeros(4), 1.03))
        self.assertEqual(controller.finish_reason, 'height')

    def test_control_gap_and_no_contact_timeout_fail(self):
        model = g.Input4ArmKinematics(self.xml, 'right')
        q = np.zeros(6)
        c = g.Input4PlacementControl(model, q, np.zeros(4), 1.)
        with self.assertRaisesRegex(ValueError, '周期'):
            c.step(q, np.zeros(4), 1.3)
        config = copy.deepcopy(g.INPUT4_ADMITTANCE)
        config['approach_timeout'] = .05
        c = g.Input4PlacementControl(model, q, np.zeros(4), 1., config)
        q, _ = c.step(q, np.zeros(4), 1.03)
        with self.assertRaisesRegex(ValueError, '接近阶段超时'):
            c.step(q, np.zeros(4), 1.06)

    def test_actual_force_filter_uses_selected_arm(self):
        node = NS(_input4_active_side='right', _input4_wrench={'left': None, 'right': None},
                  _input4_wrench_time={'left': 0., 'right': 0.})
        msg = NS(data=[1., 2., 3., 0., 0., .1])
        g.G01Demo._on_input4_wrench(node, msg, 'left')
        self.assertIsNone(node._input4_wrench['left'])
        with patch.object(g.time, 'monotonic', return_value=1.):
            g.G01Demo._on_input4_wrench(node, msg, 'right')
        np.testing.assert_array_equal(node._input4_wrench['right'], [1., 2., 3., .1])
        with patch.object(g.time, 'monotonic', return_value=1.01):
            g.G01Demo._on_input4_wrench(node, NS(data=[11., 12., 13., 0., 0., 1.1]), 'right')
        alpha = 1-np.exp(-2*np.pi*5*.01)
        np.testing.assert_allclose(node._input4_wrench['right'], [1+10*alpha, 2+10*alpha, 3+10*alpha, .1+alpha])
        g.G01Demo._on_input4_wrench(node, NS(data=[np.nan]*6), 'right')
        self.assertIsNone(node._input4_wrench['right'])

    def test_direct_motion_loop_completes_and_failure_stops(self):
        for stale in (False, True):
            clock = NS(value=1.)
            events, messages = [], []
            config = copy.deepcopy(g.INPUT4_ADMITTANCE)
            config.update(contact_duration=.09, feedback_timeout=.04 if stale else .2)
            node = NS(use_force_sensors=True, _input4_active_side=None,
                      _input4_wrench={'left': None, 'right': None},
                      _input4_wrench_time={'left': 0., 'right': 0.},
                      _input4_joint_time={'left': 0., 'right': 0.},
                      _input4_description=self.xml, _input4_models={},
                      _joints={f'r_arm_joint{i}': value for i, value in enumerate([.2,-.7,.8,.6,.9,-.4], 1)},
                      _force_action_goals=[], _force_range_fault=None,
                      _abort_on_force_range=lambda: None, get_logger=logger,
                      get_clock=lambda: NS(now=lambda: NS(to_msg=lambda: g.JointState().header.stamp)),
                      _send_servoj_control=lambda side, cmd: events.append(cmd) or 1,
                      _wait_for_servoj_control_state=lambda *a: True)
            def publish(msg):
                messages.append(msg)
                node._joints.update(zip(msg.name, msg.position))
            node._input4_command_pub = NS(publish=publish)
            spins = NS(count=0)
            def spin_once(timeout_sec):
                clock.value += max(.001, timeout_sec)
                spins.count += 1
                if stale and spins.count > 1:
                    return
                node._input4_joint_time['right'] = clock.value
                force = 0. if spins.count == 1 else -60.
                g.G01Demo._on_input4_wrench(node, NS(data=[0.,0.,force,0.,0.,0.]), 'right')
            node._executor = NS(spin_once=spin_once)
            with patch.object(g, 'INPUT4_ADMITTANCE', config), patch.object(g.time, 'monotonic', lambda: clock.value), patch.object(g.rclpy, 'ok', return_value=True):
                result = g.G01Demo._execute_input4_admittance(node, 'right_arm')
            self.assertEqual(result, not stale)
            self.assertIsNone(node._input4_active_side)
            np.testing.assert_array_equal(messages[-1].velocity, np.zeros(6))
            self.assertEqual(events, ['stop'] if stale else [])


class PlaceOrchestrationTests(unittest.TestCase):
    def make_node(self, *, failure=False, return_plan=True):
        events = []
        arms = g.joint_names_for_group('right_arm')
        body = g.joint_names_for_group('right_body')
        all_names = g.joint_names_for_group('dual_arm_body')
        state = {name: 0. for name in all_names}
        to_yubei, to_q1 = trajectory(body), trajectory(arms)
        dynamic_return = trajectory(arms, [.2]*6, [.1]*6)
        reset = trajectory(arms, [.1]*6, [.3]*6)
        saved_pose = g.Pose()
        place_pose = g.Pose()
        place_pose.position.z = .123
        def fk(link, *, joints, plan_frame, **kw):
            if plan_frame == g.SCENE_FRAME and link == 'r_tool':
                np.testing.assert_allclose([joints[name] for name in arms], [.25]*6)
                return place_pose
            if plan_frame in (g.SCENE_FRAME, 'R6'):
                pose = g.Pose()
                pose.orientation.w = 1.
                return pose
            return saved_pose
        plan = NS(arm_group='right_arm', slot_name='sw1', to_yubei=to_yubei,
                  descent=g.RobotTrajectory(), to_q1=to_q1)
        node = NS(get_logger=logger, _place_slot_is_empty=lambda name: True,
                  _has_any_empty_place_slot=lambda: True, _upper_grasp_active=False,
                  _upper_place_active=False, sim_mode=False, last_pick_failure_reason=None,
                  wait_for_operator=lambda *a: None, _wait_for_driver_signal=lambda **kw: True,
                  _get_joints=lambda names, **kw: {name: state[name] for name in names},
                  _get_link_pose_fk=fk,
                  _execute_yubei_with_recovery=lambda *a, **kw: events.append('to_yubei') or True,
                  set_tool_power=lambda side, status: events.append(('tool', status)) or True,
                  validate_trajectory_self_collision=lambda *a, **kw: events.append('check_return') or True)
        def admittance(group, *, height_stop):
            self.assertAlmostEqual(height_stop['target_z'], place_pose.position.z + .002)
            events.append('admittance')
            state.update({name: .2 for name in arms})
            return not failure
        def cartesian(*a, **kw):
            events.append('plan_return')
            self.assertIs(a[2], saved_pose)
            np.testing.assert_allclose([kw['start_joints'][name] for name in arms], [.2]*6)
            return dynamic_return if return_plan else None
        def joint_plan(group, target, **kw):
            events.append('replan_q1')
            self.assertEqual(group, 'right_arm')
            self.assertEqual(list(target), list(g.EXCHANGE_Q1[-6:]))
            np.testing.assert_allclose([kw['start_joints'][name] for name in arms], [.1]*6)
            return True, None, reset
        def execute(traj):
            events.append('return' if traj is dynamic_return else 'q1')
            state.update(zip(traj.joint_trajectory.joint_names, traj.joint_trajectory.points[-1].positions))
            return True
        node._execute_input4_admittance = admittance
        node.plan_cartesian_line = cartesian
        node.plan_joint_motion = joint_plan
        node._execute_traj = execute
        config = {'yubei': {'sw1': [0.]*len(body)}, 'fang': {'sw1': [.25]*6}}
        return node, config, plan, events

    def run_place(self, node, config, plan, return_to_q1=True):
        return g.G01Demo._place_and_return(node, .4, config, group='right_arm', link='r_tool',
            first_return_mode=0, selected_slot_name='sw1', preplanned=plan, return_to_q1=return_to_q1)

    def test_release_then_replan_measured_return_then_same_q1(self):
        node, config, plan, events = self.make_node()
        self.assertTrue(self.run_place(node, config, plan))
        self.assertEqual(events, ['to_yubei', 'admittance', ('tool',0), 'plan_return',
                                  'check_return', 'return', 'replan_q1', 'q1'])

    def test_pipeline_skips_q1_after_dynamic_return(self):
        node, config, plan, events = self.make_node()
        self.assertTrue(self.run_place(node, config, plan, False))
        self.assertEqual(events[-1], 'return')
        self.assertNotIn('q1', events)

    def test_admittance_failure_keeps_tool_power_and_skips_return(self):
        node, config, plan, events = self.make_node(failure=True)
        self.assertFalse(self.run_place(node, config, plan))
        self.assertEqual(events, ['to_yubei', 'admittance'])

    def test_failed_return_plan_does_not_execute_reset(self):
        node, config, plan, events = self.make_node(return_plan=False)
        self.assertFalse(self.run_place(node, config, plan))
        self.assertNotIn('return', events)
        self.assertNotIn('q1', events)

    def test_preplanning_no_longer_calls_fixed_fang_cartesian(self):
        names = g.joint_names_for_group('right_arm')
        body = g.joint_names_for_group('right_body')
        calls = []
        def joint_plan(group, *a, **kw):
            calls.append(group)
            return True, None, trajectory(body if group == 'right_body' else names)
        node = NS(get_logger=logger, plan_joint_motion=joint_plan,
                  _get_link_pose_fk=lambda *a, **kw: g.Pose(),
                  plan_cartesian_line=lambda *a, **kw: self.fail('fixed fang descent must not be planned'))
        with patch.object(g, '_load_trajectory_cache', return_value=None), patch.object(g, '_save_trajectory_cache', return_value=False):
            plan = g.G01Demo._plan_place_trajectories(node, arm_group='right_arm', arm_plan_frame='r_base_link',
                body_group='right_body', arm_joint_names=names, body_joint_names=body, slot_name='sw1',
                yubei_name='yubei.sw1', place_name='fang.sw1', start_joints={name: 0. for name in body},
                yubei_joints=[0.]*len(body), fang_joints=[0.]*6, link='r_tool', speed=.4)
        self.assertIsNotNone(plan)
        self.assertFalse(plan.descent.joint_trajectory.points)
        self.assertEqual(calls, ['right_body','right_arm'])


if __name__ == '__main__':
    unittest.main()
