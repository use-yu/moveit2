"""输入7双臂导纳及实测起点退回测试；不启动节点、不连接真机。"""
import ast
import copy
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import xacro

from test_input4_admittance_place import g, SCRIPT, logger, trajectory


def nested_function(name, **bindings):
    tree = ast.parse(SCRIPT.read_text())
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(vars(g), **bindings)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SCRIPT), 'exec'), namespace)
    return namespace[name]


class DualAdmittanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.xml = xacro.process_file(str(SCRIPT.parents[2] /
            'moveit_resources/g01_description/urdf/G01-URDF888.urdf')).toxml()

    def test_dual_loop_holds_finished_arm_and_stops_both_on_failure(self):
        for case in ('success', 'stale', 'force_limit'):
            with self.subTest(case=case):
                clock = NS(value=1., spins=0)
                messages, stops, motor = [], [], []
                config = copy.deepcopy(g.INPUT4_ADMITTANCE)
                config.update(contact_duration=.12, feedback_timeout=.04 if case == 'stale' else .2)
                initial = np.array([.2, -.7, .8, .6, .9, -.4])
                joints = {f'{prefix}_arm_joint{i}': value for prefix in ('l', 'r')
                          for i, value in enumerate(initial, 1)}
                left = g.Input4ArmKinematics(self.xml, 'left')
                height_stops = {'left': {'target_z': float(left.evaluate(initial)[0][2, 3])}}
                node = NS(use_force_sensors=True, _input4_active_side=None,
                    _input4_wrench={'left': None, 'right': None},
                    _input4_wrench_time={'left': 0., 'right': 0.},
                    _input4_joint_time={'left': 0., 'right': 0.},
                    _input4_description=self.xml, _input4_models={}, _joints=joints,
                    _force_action_goals=[], _force_range_fault=None,
                    _ft_sensor_z={'left': 0., 'right': 0.},
                    _motor_command_pub=NS(publish=lambda msg: motor.append(msg.data)),
                    _abort_on_force_range=lambda: None, get_logger=logger,
                    get_clock=lambda: NS(now=lambda: NS(to_msg=lambda: g.JointState().header.stamp)),
                    _send_servoj_control=lambda side, cmd: stops.append((side, cmd)) or len(stops),
                    _wait_for_servoj_control_state=lambda *a: True)
                def publish(msg):
                    messages.append(msg)
                    node._joints.update(zip(msg.name, msg.position))
                node._input4_command_pub = NS(publish=publish)
                def spin_once(timeout_sec):
                    clock.value += max(.001, timeout_sec)
                    clock.spins += 1
                    for side in ('left', 'right'):
                        if case == 'stale' and side == 'right' and clock.spins > 1:
                            continue
                        node._input4_joint_time[side] = clock.value
                        fz = -60. if side == 'right' and clock.spins > 1 else 0.
                        g.G01Demo._on_input4_wrench(node, NS(data=[0., 0., fz, 0., 0., 0.]), side)
                    if case == 'force_limit' and clock.spins > 5:
                        node._ft_sensor_z['right'] = -250.
                node._executor = NS(spin_once=spin_once)
                with patch.object(g, 'INPUT4_ADMITTANCE', config), \
                     patch.object(g.time, 'monotonic', lambda: clock.value), \
                     patch.object(g.rclpy, 'ok', return_value=True):
                    result = g.G01Demo._execute_input4_admittance(node, 'dual_arm',
                        height_stop=height_stops, force_baselines={'left': 0., 'right': 0.})
                self.assertEqual(result, case == 'success')
                self.assertIsNone(node._input4_active_side)
                self.assertTrue(messages)
                for msg in messages:
                    self.assertEqual(len(msg.name), 12)
                    np.testing.assert_allclose(msg.position[:6], initial)
                    np.testing.assert_array_equal(msg.velocity[:6], np.zeros(6))
                np.testing.assert_array_equal(messages[-1].velocity, np.zeros(12))
                if case == 'success':
                    self.assertGreater(len(messages), 3)
                    self.assertTrue(any(np.linalg.norm(msg.velocity[6:]) > 0 for msg in messages))
                    self.assertEqual(stops, [])
                else:
                    self.assertEqual(stops, [('left', 'stop'), ('right', 'stop')])
                self.assertEqual(motor, [1] if case == 'force_limit' else [])

    def test_height_targets_use_validated_descent_endpoint_and_actual_bases(self):
        names = g.joint_names_for_group('dual_arm')
        actual = {name: .1 for name in g.joint_names_for_group('dual_arm_body')}
        descent = trajectory(names, [.2]*12, [.3]*12)
        calls = []
        def fk(link, *, joints, plan_frame):
            pose = g.Pose()
            pose.orientation.w = 1.
            if link.endswith('_tool') and plan_frame == g.SCENE_FRAME:
                np.testing.assert_allclose([joints[n] for n in names], [.3]*12)
                self.assertEqual(joints['body_joint1'], actual['body_joint1'])
                pose.position.z = .5 if link == 'l_tool' else .6
            else:
                self.assertIs(joints, actual)
                if link.endswith('_base_link'):
                    pose.position.z = .8
                else:
                    pose.position.z = .15
            return pose
        node = NS(get_logger=logger, _get_joints=lambda *a, **kw: actual, _get_link_pose_fk=fk,
                  _execute_input4_admittance=lambda group, **kw: calls.append((group, kw)) or True)
        self.assertTrue(g.G01Demo._execute_dual_place_admittance(node, descent,
            force_baselines={'left': 2., 'right': 3.}))
        group, arguments = calls[0]
        self.assertEqual(group, 'dual_arm')
        self.assertEqual(arguments['force_baselines'], {'left': 2., 'right': 3.})
        for side, height in (('left', .502), ('right', .602)):
            stop = arguments['height_stop'][side]
            self.assertAlmostEqual(stop['target_z'], height)
            self.assertAlmostEqual(stop['base_transform'][2, 3], .8)
            np.testing.assert_allclose(stop['tool_offset'], [0., 0., .15])
        self.assertFalse(g.G01Demo._execute_dual_place_admittance(node, g.RobotTrajectory(),
            force_baselines={'left': 2., 'right': 3.}))
        self.assertEqual(len(calls), 1)

    def test_retreat_plans_both_arms_from_same_actual_state_and_checks_merged_collision(self):
        actual = {n: .2 for n in g.joint_names_for_group('dual_arm_body')}
        poses = [g.Pose(), g.Pose()]
        paths, checks = [], []
        def plan(group, link, pose, **kw):
            self.assertIs(kw['start_joints'], actual)
            self.assertEqual(kw['plan_frame'], g.SCENE_FRAME)
            self.assertFalse(kw['avoid_collisions'])
            self.assertIs(pose, poses[len(paths)])
            path = trajectory(kw['joint_names'], [.2]*6, [.1]*6)
            path.joint_trajectory.points[-1].time_from_start.sec = 1
            paths.append(path)
            return path
        node = NS(get_logger=logger, _get_joints=lambda *a, **kw: actual, plan_cartesian_line=plan,
                  validate_trajectory_self_collision=lambda *a, **kw: checks.append((a, kw)) or True)
        result = g.G01Demo._plan_dual_place_retreat(node, *poses, 'dual_arm_body')
        self.assertIsNotNone(result)
        self.assertEqual(len(paths), 2)
        self.assertEqual(len(result.joint_trajectory.joint_names), 12)
        self.assertIs(checks[0][0][0], result)
        self.assertIs(checks[0][1]['base_state'], actual)
        self.assertEqual(checks[0][1]['group'], 'dual_arm_body')

    def test_place_execution_uses_admittance_and_new_retreat_keeps_next_stage_callback(self):
        events = []
        names = g.joint_names_for_group('dual_arm_body')
        actual = {name: .1 for name in names}
        to_place, descent, retreat = (trajectory(names) for _ in range(3))
        actual_poses = {'l_tool': g.Pose(), 'r_tool': g.Pose()}
        def execute(path):
            self.assertIn(path, (to_place, retreat))
            events.append('to_place' if path is to_place else 'retreat')
            return True
        def admittance(path, **kw):
            self.assertIs(path, descent)
            self.assertEqual(kw['force_baselines'], {'left': 1., 'right': 2.})
            events.append('admittance')
            return True
        def plan_return(left, right, group):
            self.assertIs(left, actual_poses['l_tool'])
            self.assertIs(right, actual_poses['r_tool'])
            self.assertEqual(group, 'dual_arm_body')
            events.append('plan_return')
            return retreat
        node = NS(use_force_sensors=True, _execute_traj=execute, wait_for_operator=lambda *a: None,
            wait_for_ft_sensor_z=lambda side: 1. if side == 'left' else 2.,
            _get_joints=lambda *a, **kw: actual,
            _get_link_pose_fk=lambda link, **kw: actual_poses[link],
            _execute_dual_place_admittance=admittance, _plan_dual_place_retreat=plan_return,
            _reverse_trajectory=lambda *a: self.fail('must replan from actual state'),
            _upper_place_active=False, _upper_grasp_active=False)
        function = nested_function('execute_cached_place_plan', node=node, log=logger(),
            current_full=actual, dual_body_joint_names=names, left_point=1, right_point=3,
            on_place_motion_start=lambda state: events.append('next_stage'),
            release_unload_pair=lambda *a: events.append('release') or True)
        with patch.object(g.time, 'sleep'):
            self.assertTrue(function('dual_arm_body', 'test', to_place, descent, g.Pose(), g.Pose()))
        self.assertEqual(events, ['next_stage', 'to_place', 'admittance', 'release', 'plan_return', 'retreat'])

    def test_failed_descent_feasibility_still_rejects_candidate_before_ompl(self):
        planner = NS(plan_joint_motion=lambda *a, **kw: self.fail('unreachable descent must reject candidate'))
        function = nested_function('validate_place_candidate', current_full={}, table_pregrasp_only=False,
            merge_place_descents=lambda *a: None, planner=planner)
        self.assertIsNone(function('dual_arm', {}, {}, 'unreachable', g.RobotTrajectory(), g.RobotTrajectory()))

    def test_noforces_dual_place_executes_original_descent_and_reverse_without_force_reads(self):
        events = []
        names = g.joint_names_for_group('dual_arm_body')
        to_place, descent, retreat = (trajectory(names) for _ in range(3))
        def execute(path):
            events.append('to_place' if path is to_place else 'descent' if path is descent else 'retreat')
            return True
        def reverse(path):
            self.assertIs(path, descent)
            events.append('reverse')
            return retreat
        def forbidden(*a, **kw):
            self.fail('no-force placement must not read sensors, run admittance or replan retreat')
        node = NS(use_force_sensors=False, _execute_traj=execute, _reverse_trajectory=reverse,
            wait_for_operator=lambda *a: None, wait_for_ft_sensor_z=forbidden,
            _get_joints=forbidden, _get_link_pose_fk=forbidden,
            _execute_dual_place_admittance=forbidden, _plan_dual_place_retreat=forbidden)
        function = nested_function('execute_cached_place_plan', node=node, log=logger(),
            current_full={}, dual_body_joint_names=names, left_point=1, right_point=3,
            on_place_motion_start=lambda state: events.append('next_stage'),
            release_unload_pair=lambda *a: events.append('release') or True)
        with patch.object(g.time, 'sleep'):
            self.assertTrue(function('dual_arm_body', 'test', to_place, descent, g.Pose(), g.Pose()))
        self.assertEqual(events, ['next_stage', 'to_place', 'descent', 'release', 'reverse', 'retreat'])


if __name__ == '__main__':
    unittest.main()
