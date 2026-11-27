"""实际勾选计数、并发保存与滚动计划进度口径回归。"""
import ast
import logging
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from core.plan_progress import summarize_plan_progress, plan_progress_data
from core.storage import DataStorage
from core.message import Message, MessageType
from agents.progress_service import ProgressService
from agents.planner_service import PlannerService
from storage.plans import PlansStore


def plan_fixture(flags=None):
    return {'subject': 'Java', 'goal': '完成项目', 'duration_days': 61,
            'start_date': '2026-11-25T00:00:00', 'created_at': 'plan-1',
            'weekly_plan': [{'week': i, 'tasks': [f'任务{j}' for j in range(7)] if i == 1 else []}
                            for i in range(1, 10)],
            'task_completion': {'1': flags or [False] * 7}, 'progress': 0}


class TestTaskSummary(unittest.TestCase):
    def test_five_of_seven_has_explicit_week_scope(self):
        summary = summarize_plan_progress(plan_fixture([True] * 5 + [False] * 2), date(2026, 11, 25))
        self.assertEqual((summary['current_week_completed'], summary['current_week_total']), (5, 7))
        self.assertEqual(summary['current_week_percentage'], 71)
        self.assertEqual((summary['detailed_weeks'], summary['total_weeks']), (1, 9))
        self.assertTrue(summary['has_unplanned_weeks'])
        self.assertNotIn('overall_percentage', summary)

    def test_saved_four_flags_and_surplus_flags(self):
        plan = plan_fixture([True, True, False, False, False, True, True, True])
        plan['task_completion']['9'] = [True] * 50
        summary = summarize_plan_progress(plan, date(2026, 11, 25))
        self.assertEqual(summary['completed_tasks'], 4)
        self.assertEqual(summary['total_tasks'], 7)
        self.assertEqual(summary['current_week_percentage'], 57)

    def test_unplanned_current_week_has_no_percentage(self):
        summary = summarize_plan_progress(plan_fixture([True] * 7), date(2026, 12, 2))
        self.assertEqual(summary['current_week'], 2)
        self.assertIsNone(summary['current_week_percentage'])
        self.assertEqual(summary['current_week_total'], 0)

    def test_new_detailed_week_keeps_completed_flags(self):
        plan = plan_fixture([True] * 5 + [False] * 2)
        old = deepcopy(plan['task_completion'])
        plan['weekly_plan'][1]['tasks'] = ['新任务'] * 7
        summary = summarize_plan_progress(plan, date(2026, 11, 25))
        self.assertEqual(summary['current_week_percentage'], 71)
        self.assertEqual(summary['detailed_task_percentage'], 35)
        self.assertEqual(plan['task_completion'], old)


class TestTaskPersistence(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = PlansStore(self.temp.name)
        self.store.save('u', 'Java', plan_fixture())

    def test_concurrent_instances_keep_all_five_changes(self):
        for _ in range(4):
            self.store.save('u', 'Java', plan_fixture())
            barrier = threading.Barrier(5)
            def toggle(index):
                store = PlansStore(self.temp.name)
                barrier.wait()
                store.set_task_completed('u', 'Java', 1, index, True, 'plan-1')
            with ThreadPoolExecutor(5) as pool:
                list(pool.map(toggle, range(5)))
            saved = self.store.get('u', 'Java')
            self.assertEqual(saved['task_completion']['1'], [True] * 5 + [False] * 2)
            self.assertEqual(saved['progress'], 71)

    def test_idempotent_set_and_uncheck(self):
        for _ in range(2):
            self.store.set_task_completed('u', 'Java', 1, 0, True)
        self.assertEqual(self.store.get('u', 'Java')['progress_summary']['completed_tasks'], 1)
        self.store.set_task_completed('u', 'Java', 1, 0, False)
        self.assertEqual(self.store.get('u', 'Java')['progress'], 0)

    def test_stale_plan_rejected_and_task_index_validated(self):
        before = self.store.get('u', 'Java')
        for index, expected in [(0, 'old-plan'), (7, 'plan-1'), (-1, 'plan-1')]:
            with self.assertRaises(ValueError):
                self.store.set_task_completed('u', 'Java', 1, index, True, expected)
        self.assertEqual(self.store.get('u', 'Java'), before)

    def test_week_generation_cannot_overwrite_recent_check(self):
        before = self.store.get('u', 'Java')
        generated = deepcopy(before)
        generated['weekly_plan'][1]['tasks'] = ['下一周任务']
        self.store.set_task_completed('u', 'Java', 1, 0, True)
        self.assertFalse(self.store.save_if_unchanged('u', 'Java', before, generated))
        self.assertTrue(self.store.get('u', 'Java')['task_completion']['1'][0])


class TestProgressSynchronization(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.storage = DataStorage(self.temp.name)
        self.storage.save_plan('u', 'Java', plan_fixture([True] * 5 + [False] * 2))
        self.service = ProgressService.__new__(ProgressService)
        self.service.storage = self.storage

    def test_sync_verifies_actual_count_and_preserves_time_notes(self):
        self.storage.save_progress('u', 'Java', {'user_id': 'u', 'subject': 'Java', 'study_time': 80,
                                                'notes': [{'note': '学习笔记'}]})
        payload = plan_progress_data(self.storage.get_plan('u', 'Java'))
        result = self.service.update_progress('u', 'Java', payload)
        self.assertEqual(result['progress_percentage'], 71)
        self.assertTrue(result['percentage_verifiable'])
        self.assertEqual(result['study_time'], 80)
        self.assertEqual(result['notes'], [{'note': '学习笔记'}])

    def test_stale_legacy_progress_refreshed_on_read_without_writing(self):
        record = {'user_id': 'u', 'subject': 'Java', 'total_tasks': 7, 'completed_tasks': 4,
                  'progress_percentage': 0, 'percentage_verifiable': False,
                  'tasks': [{'week': '1', 'index': 0}], 'study_time': 80}
        self.storage.save_progress('u', 'Java', record)
        result = self.storage.get_all_progress('u')['u_Java']
        self.assertEqual((result['completed_tasks'], result['total_tasks']), (5, 7))
        self.assertEqual(result['progress_percentage'], 71)
        self.assertEqual(self.storage.progress_store.get('u', 'Java')['completed_tasks'], 4)

    def test_manual_counts_remain_independent(self):
        self.service.update_progress('u', 'Java', {'completed_tasks': 2, 'total_tasks': 10,
            'calculation_basis': {'verifiable': True, 'total_tasks_source': 'manual'}})
        result = self.storage.get_progress('u', 'Java')
        self.assertEqual(result['progress_percentage'], 20)
        self.assertEqual(result['progress_scope'], 'manual')
        self.assertNotIn('plan_progress_summary', result)

    def test_stale_sync_payload_uses_latest_plan(self):
        old_payload = plan_progress_data(plan_fixture())
        result = self.service.update_progress('u', 'Java', old_payload)
        self.assertEqual(result['completed_tasks'], 5)

    def test_time_only_update_retains_plan_task_scope(self):
        self.service.update_progress('u', 'Java', plan_progress_data(self.storage.get_plan('u', 'Java')))
        self.service.update_progress('u', 'Java', {'study_time': 20, 'note': '今天复习'})
        result = self.storage.get_progress('u', 'Java')
        self.assertEqual(result['progress_scope'], 'detailed_tasks')
        self.assertEqual(result['completed_tasks'], 5)
        self.assertEqual(result['study_time'], 20)

    def test_concurrent_study_time_updates_are_preserved(self):
        barrier = threading.Barrier(5)
        def update(_):
            service = ProgressService.__new__(ProgressService)
            service.storage = DataStorage(self.temp.name)
            barrier.wait()
            service.update_progress('u', 'Java', {'study_time': 10})
        with ThreadPoolExecutor(5) as pool:
            list(pool.map(update, range(5)))
        self.assertEqual(self.storage.get_progress('u', 'Java')['study_time'], 50)

    def test_read_plan_derives_week_summary_without_mutation(self):
        planner = PlannerService.__new__(PlannerService)
        planner.storage = self.storage
        before = self.storage.get_plan('u', 'Java')
        result = planner._attach_dynamic_fields(before, 'u', 'Java')
        self.assertEqual(result['progress_summary']['completed_tasks'], 5)
        self.assertEqual(result['progress'], 71)
        self.assertEqual(before['progress'], 0)


class TestCheckboxAPI(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.storage = DataStorage(self.temp.name)
        self.storage.save_plan('u', 'Java', plan_fixture())
        self.payload = {'user_id': 'u', 'week': '1', 'task_index': 0, 'completed': True,
                        'plan_created_at': 'plan-1'}
        path = Path(__file__).resolve().parents[1] / 'app.py'
        names = {'_sync_checked_plan', '_checked_plan_response', 'toggle_plan_task', 'sync_plan_tasks'}
        nodes = [node for node in ast.parse(path.read_text(encoding='utf-8-sig')).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
        for node in nodes:
            node.decorator_list = []
        self.messages = []
        self.ctx = {'DataStorage': lambda: self.storage, 'jsonify': lambda value: value,
                    'request': SimpleNamespace(get_json=lambda silent=True: self.payload),
                    'logger': logging.getLogger('test'), 'Message': Message, 'MessageType': MessageType,
                    'coordinator': SimpleNamespace(route_message=lambda msg: self.messages.append(msg)),
                    'track_event': Mock(), 'plan_progress_data': plan_progress_data,
                    'summarize_plan_progress': summarize_plan_progress}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), self.ctx)

    def test_five_check_requests_and_sync_keep_count(self):
        for index in range(5):
            self.payload['task_index'] = index
            result = self.ctx['toggle_plan_task']('Java')
            self.assertEqual(result['status'], 'success')
        before = self.storage.get_plan('u', 'Java')
        result = self.ctx['sync_plan_tasks']('Java')
        self.assertEqual(result['data']['progress_summary']['completed_tasks'], 5)
        self.assertEqual(self.storage.get_plan('u', 'Java'), before)
        self.assertTrue(self.messages[-1].content['progress_data']['calculation_basis']['verifiable'])

    def test_validation_and_stale_ui_status(self):
        for field, value, status in [('completed', 'false', 400), ('task_index', 1.5, 400),
                                     ('task_index', 9, 400), ('plan_created_at', 'old-plan', 409)]:
            original = deepcopy(self.payload)
            self.payload[field] = value
            self.assertEqual(self.ctx['toggle_plan_task']('Java')[1], status)
            self.payload = original

    def test_sync_failure_does_not_report_saved_check_as_failed(self):
        self.ctx['coordinator'].route_message = Mock(side_effect=RuntimeError('暂时失败'))
        result = self.ctx['toggle_plan_task']('Java')
        self.assertEqual(result['status'], 'success')
        self.assertTrue(result['sync_warning'])
        self.assertTrue(self.storage.get_plan('u', 'Java')['task_completion']['1'][0])


if __name__ == '__main__':
    unittest.main()
