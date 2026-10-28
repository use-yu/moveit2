"""抓取后搬运/交换的请求内碰撞场景测试；不连接机器人。"""
import copy
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from test_input4_admittance_place import g, logger, trajectory


def obstacles():
    pose = g.Pose()
    pose.orientation.w = 1.
    pose.position.z = .4
    return [g.make_deep_frame_cutoff(.6, pose), g.make_grasp_objects_collision([pose])]


class TransferCollisionTests(unittest.TestCase):
    def test_snapshot_uses_full_q_pre_state_and_original_candidate_pose(self):
        state = {'body_joint1': .1, 'body_joint2': .2, 'r_arm_joint1': .3}
        pose = g.Pose()
        pose.position.x, pose.position.z, pose.orientation.w = .8, .4, 1.
        calls = []
        def surface(link, **kw):
            self.assertEqual(kw['joints'], state)
            calls.append(link)
            return .7, 'R6'
        node = NS(get_logger=logger, _frame_pose=pose,
            _pose_in_frame=lambda *a, **kw: pose, _link_collision_min_z=surface)
        result = g.G01Demo._make_grasp_transfer_obstacles(node, pose, 'r_base_link', 'r_tool', state, [pose])
        self.assertEqual(calls, ['r_tool'])
        self.assertEqual([o.id for o in result], [g.FRAME_CUTOFF_ID, g.GRASP_OBJECT_COLLISION_ID])
        plate = result[0]
        self.assertAlmostEqual(plate.primitive_poses[0].position.z+g.FRAME_CUTOFF_THICKNESS/2,
                               .7-g.FRAME_CUTOFF_BELOW_COLLISION)
        pose.position.z = 10.
        self.assertAlmostEqual(result[1].primitive_poses[0].position.z, .4)
        node._link_collision_min_z=lambda *a, **kw: None
        self.assertIsNone(g.G01Demo._make_grasp_transfer_obstacles(node, pose, 'r_base_link', 'r_tool', state, [pose]))

    def test_movegroup_request_contains_obstacles_and_preserves_collision_checks(self):
        goals = []
        path = trajectory(['r_arm_joint1'])
        def send(client, goal):
            goals.append(goal)
            result = NS(status=g.GoalStatus.STATUS_SUCCEEDED, result=NS(
                error_code=NS(val=g.MoveItErrorCodes.SUCCESS), planned_trajectory=path))
            return NS(result=lambda: NS(accepted=True, get_result_async=lambda: NS(result=lambda: result)))
        node = NS(get_logger=logger, _move_cli=object(), _send_force_guarded_goal=send,
                  _spin_until=lambda *a: True)
        models = obstacles()
        self.assertTrue(g.G01Demo._move_once(node, 'right_arm', [], {'r_arm_joint1': 0.}, True, .2,
            planning_obstacles=models)[0])
        diff = goals[0].planning_options.planning_scene_diff
        self.assertTrue(diff.is_diff)
        self.assertTrue(diff.robot_state.is_diff)
        self.assertEqual(list(diff.world.collision_objects), models)
        self.assertFalse(diff.allowed_collision_matrix.default_entry_names)
        models[0].primitive_poses[0].position.z = 999.
        self.assertNotEqual(diff.world.collision_objects[0].primitive_poses[0].position.z, 999.)
        self.assertFalse(g.G01Demo._move_once(node, 'right_arm', [], {}, True, .2,
            avoid_collisions=False, planning_obstacles=models)[0])
        self.assertEqual(len(goals), 1)

    def test_obstacles_reach_movegroup_through_joint_planning_and_waypoints(self):
        seen = []
        models = obstacles()
        node = NS(get_logger=logger, _upper_grasp_active=False, _upper_place_active=False,
            _move_cli=NS(wait_for_server=lambda **kw: True),
            _get_joints=lambda *a, **kw: {'r_arm_joint1': 0.})
        def once(*args):
            seen.append(args[-1])
            return True, 1., trajectory(['r_arm_joint1']), g.MoveItErrorCodes.SUCCESS
        node._move_once=once
        node.move=lambda *a, **kw: g.G01Demo.move(node, *a, **kw)
        node.plan_joint_motion=lambda *a, **kw: g.G01Demo.plan_joint_motion(node, *a, **kw)
        self.assertTrue(g.G01Demo.plan_execute_joint_waypoints_impl(node, 'right_arm', .2,
            ['r_arm_joint1'], [[.1]], start_joints={'r_arm_joint1': 0.}, planning_obstacles=models))
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0], models)

    def test_exchange_to_q2_uses_obstacles_and_records_signature(self):
        models, calls = obstacles(), []
        names = g.joint_names_for_group('dual_arm_body')
        pose = g.Pose(); pose.orientation.w = 1.
        def joint_plan(group, q, **kw):
            calls.append(kw)
            return True, 1., trajectory(kw['joint_names'], [0.]*len(q), list(q))
        node = NS(get_logger=logger, plan_joint_motion=joint_plan,
            _get_link_pose_fk=lambda *a, **kw: pose,
            plan_cartesian_line=lambda *a, **kw: trajectory(kw['joint_names']))
        result = g.G01Demo._plan_exchange_trajectories(node, g.EXCHANGE_Q2,
            source_link='r_tool', pick_group='right_body', dual_speed=.2, cartesian_speed=.2,
            z_down_distance=None, start_joints={n: 0. for n in names}, transfer_obstacles=models)
        self.assertIsNotNone(result)
        self.assertIs(calls[0]['planning_obstacles'], models)
        self.assertFalse(calls[0]['execute'])
        self.assertEqual(result.transfer_signature, g.transfer_obstacle_signature(models))

    def test_background_exchange_and_direct_place_receive_same_snapshot(self):
        for mode in (1, 2):
            with self.subTest(mode=mode):
                models, calls = obstacles(), []
                names = g.joint_names_for_group('dual_arm_body')
                state = {n: 0. for n in names}
                def exchange(*a, **kw):
                    self.assertIs(kw['transfer_obstacles'], models)
                    calls.append('exchange')
                    return NS(to_q2=trajectory(names, [0.]*len(names), [.2]*len(names)),
                        exchange_joint_names=names, receiver_side='left')
                def place(**kw):
                    self.assertIs(kw['transfer_obstacles'], models)
                    self.assertEqual(kw['start_joints'], {n: .2 for n in names} if mode==1 else state)
                    calls.append('place')
                    return NS()
                node = NS(_plan_exchange_trajectories=exchange, _plan_place_trajectories=place)
                config={key: [0.]*8 for key in ('yubei','yubei_j')}
                config.update({key: [0.]*6 for key in ('fang','fang_j')})
                self.assertIsNotNone(g.G01Demo._plan_post_grasp_trajectories(node,
                    recognition_state=state, pick_group='right_arm', pick_link='r_tool',
                    first_return_mode=mode, selected_slot_name=None, place_joint_config=config,
                    speed=.2, transfer_obstacles=models))
                self.assertEqual(calls, ['exchange', 'place'] if mode==1 else ['place'])

    def test_place_cache_reuses_only_matching_obstacle_geometry(self):
        models, planned, saved = obstacles(), [], {}
        arms = g.joint_names_for_group('right_arm')
        body = g.joint_names_for_group('right_body')
        def joint_plan(group, target, **kw):
            planned.append((group, kw))
            return True, 1., trajectory(kw['joint_names'])
        node = NS(use_force_sensors=True, get_logger=logger, plan_joint_motion=joint_plan,
            _get_link_pose_fk=lambda *a, **kw: g.Pose())
        def save(key, metadata, messages):
            saved[key] = copy.deepcopy((metadata, messages)); return True
        kwargs=dict(arm_group='right_arm', arm_plan_frame=g.SCENE_FRAME, body_group='right_body',
            arm_joint_names=arms, body_joint_names=body, slot_name='sw1', yubei_name='yubei.sw1',
            place_name='fang.sw1', start_joints={n: 0. for n in body}, yubei_joints=[0.]*len(body),
            fang_joints=[0.]*6, link='r_tool', speed=.2, transfer_obstacles=models)
        with patch.object(g, '_load_trajectory_cache', side_effect=lambda key, types: saved.get(key)), \
             patch.object(g, '_save_trajectory_cache', side_effect=save):
            first = g.G01Demo._plan_place_trajectories(node, **kwargs)
            self.assertIs(planned[0][1]['planning_obstacles'], models)
            self.assertNotIn('planning_obstacles', planned[1][1])  # 空载Q1复位不沿用该障碍物。
            planned.clear()
            second = g.G01Demo._plan_place_trajectories(node, **kwargs)
            self.assertFalse(planned)
            self.assertEqual(second.transfer_signature, first.transfer_signature)
            models[1].primitive_poses[0].position.x += .1
            third = g.G01Demo._plan_place_trajectories(node, **kwargs)
            self.assertTrue(planned)
            self.assertNotEqual(third.transfer_signature, first.transfer_signature)

    def test_yubei_recovery_replanning_keeps_same_obstacles(self):
        models, calls, execution = obstacles(), [], [False, True]
        body = g.joint_names_for_group('right_body')
        def plan(group, target, **kw):
            calls.append(kw)
            return True, 1., trajectory(body)
        node = NS(get_logger=logger, last_pick_failure_reason=None,
            _execute_traj=lambda path: execution.pop(0), _abort_on_force_range=lambda: None,
            _upper_grasp_active=False, _upper_grasp_recovering=False, _force_action_goals=[],
            _send_servoj_control=lambda *a: 1, _wait_for_servoj_control_state=lambda *a: True,
            clear_arm_alarm=lambda *a: True, _get_joints=lambda names, **kw: {n: 0. for n in names},
            plan_joint_motion=plan, publish_error=lambda *a: self.fail(str(a)))
        with patch.object(g.rclpy, 'ok', return_value=True):
            self.assertTrue(g.G01Demo._execute_yubei_with_recovery(node, trajectory(body),
                body_group='right_body', body_joint_names=body, yubei_joints=[0.]*len(body),
                yubei_name='yubei.sw1', side='right', speed=.2, transfer_obstacles=models))
        self.assertIs(calls[0]['planning_obstacles'], models)
        self.assertTrue(calls[0]['avoid_collisions'])
        self.assertFalse(execution)


if __name__ == '__main__':
    unittest.main()
