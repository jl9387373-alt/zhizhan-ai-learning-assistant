"""长计划只细化近期，按周结合真实完成记录推进，短任务不均摊拉长。"""
import os
import sys
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.planner_service import PlannerService
from agents.plan_generation import validate_goal_path, validate_daily_entries, generate_goal_path, generate_learning_week


def rolling_payload(days=61):
    return {'outcome':'完成图书管理项目','goal_feasible':True,'assumptions':['已有基础'],
            'phases':[{'title':'基础','start_day':1,'end_day':14,'objective':'掌握基础','deliverable':'语法练习'},
                      {'title':'项目实践','start_day':15,'end_day':days-7,'objective':'完成主要功能','deliverable':'项目代码'},
                      {'title':'验收','start_day':days-6,'end_day':days,'objective':'测试并演示','deliverable':'演示记录'}],
            'first_week':[{'day':i,'topic':'Java基础','task':f'完成第{i}个语法示例，修改输入并检查输出结果',
                           'hours':1,'deliverable':f'示例{i}及输出记录'} for i in range(1,8)]}


class TestRollingValidation(unittest.TestCase):
    def test_full_period_only_seven_days_detailed(self):
        result=validate_goal_path(rolling_payload(),61,6)
        self.assertTrue(result['rolling'])
        self.assertEqual(len(result['daily_entries']),7)
        self.assertEqual(result['phases'][-1]['end_day'],61)
        self.assertTrue(all(day['estimated_hours']==1 for day in result['daily_entries']))

    def test_rejects_gaps_and_repeated_tasks(self):
        payload=rolling_payload()
        payload['phases'][1]['start_day']=16
        with self.assertRaises(ValueError): validate_goal_path(payload,61,6)
        payload=rolling_payload()
        payload['first_week'][1]['task']=payload['first_week'][0]['task']
        with self.assertRaises(ValueError): validate_goal_path(payload,61,6)
        payload=rolling_payload()
        payload['first_week'][0]['task']='展示项目成果（第1/7部分）'
        with self.assertRaises(ValueError): validate_goal_path(payload,61,6)

    def test_short_task_must_fit_one_day(self):
        payload={'outcome':'完成成果','goal_feasible':True,'milestones':[
            {'topic':'学习','task':'完成一项具体练习并记录结果','hours':2,'deliverable':'练习记录'},
            {'topic':'验收','task':'检查成果是否达到目标并记录差距','hours':1,'deliverable':'验收记录'}]}
        with self.assertRaises(ValueError): validate_goal_path(payload,10,1)

    def test_long_prompt_requests_stages_and_first_week(self):
        with patch('agents.plan_generation.config.get',return_value='test-key'), \
             patch('agents.plan_generation._request_plan_json',return_value=rolling_payload()) as request:
            result=generate_goal_path('Java','完成图书管理项目',61,6,{})
        self.assertTrue(result['rolling'])
        prompt,requirements=request.call_args.args
        self.assertIn('first_week',prompt)
        self.assertIn('不必填满',prompt)
        self.assertEqual(requirements['days'],61)


class TestWeeklyRefinement(unittest.TestCase):
    def setUp(self):
        self.planner=PlannerService.__new__(PlannerService)
        self.store={}
        self.planner.profile_store=SimpleNamespace(get_profile=lambda uid:{'daily_hours':6})
        self.planner.storage=SimpleNamespace(
            save_plan=lambda uid,topic,plan:self.store.update(plan=deepcopy(plan)),
            get_plan=lambda uid,topic:deepcopy(self.store.get('plan')))
        self.planner._attach_dynamic_fields=lambda plan,*args:plan
        self.planner._generate_goal_path=lambda *args:validate_goal_path(rolling_payload(),61,6)
        with patch('agents.planner_service.track_event'):
            self.plan=self.planner.create_learning_plan('Java','u','9周',{'duration_days':61,'goal':'完成图书管理项目','daily_hours':6})

    def week_entries(self, week):
        first=(week-1)*7+1
        last=min(week*7,61)
        return validate_daily_entries([{'day':i,'topic':'Java实践','task':f'编写并检查第{i}个类的属性和构造函数',
                                       'hours':1.5,'deliverable':f'第{i}个类与输出'} for i in range(first,last+1)],first,last,6)

    def test_initial_plan_keeps_future_as_direction(self):
        self.assertEqual(len(self.plan['daily_plan']),7)
        self.assertEqual(len(self.plan['weekly_plan']),9)
        self.assertEqual(self.plan['planning_mode'],'rolling')
        self.assertEqual(self.plan['detailed_weeks'],[1])
        self.assertEqual(self.plan['stage_plan'][-1]['end_day'],61)
        self.assertTrue(all(day['estimated_hours']<6 for day in self.plan['daily_plan']))

    def test_next_week_uses_completed_and_missed_tasks(self):
        self.store['plan']['task_completion']['1'][0]=True
        old_week=deepcopy(self.store['plan']['weekly_plan'][0])
        self.planner._generate_learning_week=Mock(return_value=self.week_entries(2))
        result=self.planner.refine_learning_week('u','Java',2)
        self.assertEqual(result['status'],'success')
        plan=result['plan']
        self.assertEqual(plan['weekly_plan'][0],old_week)
        self.assertTrue(plan['task_completion']['1'][0])
        self.assertEqual(len(plan['daily_plan']),14)
        self.assertEqual(plan['detailed_weeks'],[1,2])
        progress=self.planner._generate_learning_week.call_args.args[2]['previous_tasks']
        self.assertTrue(progress[0]['completed'])
        self.assertFalse(progress[1]['completed'])
        self.assertEqual(plan['daily_plan'][7]['task_refs'][0],{'week':'2','task_index':0})

    def test_does_not_overwrite_completed_week(self):
        self.store['plan']['task_completion']['1'][0]=True
        self.planner._generate_learning_week=Mock()
        self.assertEqual(self.planner.refine_learning_week('u','Java',1)['status'],'error')
        self.planner._generate_learning_week.assert_not_called()

    def test_cannot_skip_weeks_and_failure_preserves_plan(self):
        self.planner._generate_learning_week=Mock(return_value=None)
        self.assertEqual(self.planner.refine_learning_week('u','Java',3)['status'],'error')
        before=deepcopy(self.store['plan'])
        self.assertEqual(self.planner.refine_learning_week('u','Java',2)['status'],'error')
        self.assertEqual(self.store['plan'],before)

    def test_progress_change_during_generation_is_preserved(self):
        def generate(*args):
            self.store['plan']['task_completion']['1'][0]=True
            return self.week_entries(2)
        self.planner._generate_learning_week=generate
        self.assertEqual(self.planner.refine_learning_week('u','Java',2)['status'],'error')
        self.assertTrue(self.store['plan']['task_completion']['1'][0])
        self.assertEqual(len(self.store['plan']['daily_plan']),7)

    def test_main_tasks_leave_twenty_percent_margin(self):
        payload=rolling_payload()
        payload['first_week'][0]['hours']=6
        with self.assertRaises(ValueError): validate_goal_path(payload,61,6)

    def test_followup_prompt_receives_progress(self):
        payload={'days':[{'day':i,'topic':'Java实践','task':f'完成第{i}个具体练习并检查结果','hours':1,
                         'deliverable':f'练习{i}'} for i in range(8,15)]}
        with patch('agents.plan_generation._request_plan_json',return_value=payload) as request:
            result=generate_learning_week(self.plan,2,{'previous_tasks':[{'task':'语法练习','completed':False}]})
        self.assertEqual(len(result),7)
        self.assertFalse(request.call_args.args[1]['progress']['previous_tasks'][0]['completed'])

    def test_short_tasks_are_not_stretched_or_fragmented(self):
        source=[{'topics':['验收'],'tasks':['完整展示成果并记录需要改进的地方'],
                 'task_estimates':[.75],'deliverables':['演示记录']}]
        daily,_,_=self.planner._schedule_with_budget(source,61,6,'展示成果',False)
        active=[d for d in daily if d['tasks']]
        self.assertEqual(len(active),1)
        self.assertEqual(active[0]['estimated_hours'],.75)
        self.assertNotIn('部分',str(active))


if __name__=='__main__': unittest.main()
