"""计划追问、画像时间确认和日历周期回归，不调用模型。"""
import os
import sys
import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.conversation_context import conversation_context
from core.coordinator import AgentCoordinator
from core.message import Message, MessageType
from core.plan_dialog import build_plan_clarification, goal_choices
from agents.planner_service import PlannerService
from nlp.intent_detection import _detect_intent_rules
from nlp.plan_constraints import extract_plan_constraints


class TestCalendarPeriod(unittest.TestCase):
    def test_two_months_are_calendar_months(self):
        params, errors = extract_plan_constraints('预计两个月', reference_date=date(2026, 11, 25))
        self.assertFalse(errors)
        self.assertEqual(params['requested_duration'], '两个月')
        self.assertEqual(params['duration_days'], 61)
        self.assertEqual(params['duration_months'], 2)
        self.assertEqual(params['planned_end_date'], '2027-01-24')

    def test_month_end_and_leap_year(self):
        for start, days, end in [(date(2026, 1, 31), 28, '2026-02-27'), (date(2028, 1, 31), 29, '2028-02-28'),
                                 (date(2026, 12, 31), 31, '2027-01-30')]:
            params, errors = extract_plan_constraints('一个月', reference_date=start)
            self.assertFalse(errors)
            self.assertEqual(params['duration_days'], days)
            self.assertEqual(params['planned_end_date'], end)

    def test_explicit_days_and_hours(self):
        params, errors = extract_plan_constraints('周期10天，每天改成1小时', reference_date=date(2026, 11, 25))
        self.assertFalse(errors)
        self.assertEqual(params['duration_days'], 10)
        self.assertEqual(params['daily_hours'], 1)
        self.assertNotIn('duration_months', params)


class TestDialogFlow(unittest.TestCase):
    def setUp(self):
        self.uid, self.cid = 'plan_dialog_test', self.id()
        conversation_context.reset(self.uid, self.cid)
        self.c = AgentCoordinator.__new__(AgentCoordinator)
        self.c.storage = SimpleNamespace(profiles_store=SimpleNamespace(get_profile=lambda uid: {
            'profile_completed': True, 'daily_hours': 6, 'goal': '无关目标'}),
            get_conversation=lambda cid: {}, save_qa_history=Mock())
        self.messages = []
        def process(msg):
            self.messages.append(msg.content)
            return Message('planner', msg.sender, MessageType.RESPONSE, {
                'status':'success', 'plan': {'subject':msg.content['subject']}})
        self.c.get_agent = lambda name: SimpleNamespace(process=process)
        self.c.detect_intent = lambda text: _detect_intent_rules(text)
        self.memory = patch('core.coordinator.conversation_memory', SimpleNamespace(
            load_from_database=lambda uid: None, get_recent_user_questions=lambda uid, limit=5: [],
            get_relevant_long_term_memory=lambda uid, query: None))
        self.memory.start()
        self.addCleanup(self.memory.stop)
        self.today = patch('nlp.plan_constraints.plan_today', return_value=date(2026,11,25))
        self.today.start()
        self.addCleanup(self.today.stop)
        self.addCleanup(conversation_context.reset, self.uid, self.cid)

    def run_stream(self, text):
        return list(self.c.process_request_stream(text, self.uid, conversation_id=self.cid))[-1]['data']

    def test_screenshot_prompt_does_not_assume_six_hours(self):
        result = self.run_stream('帮我制定一下java学习计划，预计两个月')
        self.assertEqual(result['type'],'plan_clarification')
        self.assertEqual(result['missing_fields'], ['goal','daily_hours'])
        self.assertNotIn('daily_hours', result['plan_params'])
        self.assertEqual(result['plan_params']['suggested_daily_hours'],6)
        self.assertIn('Java，两个月',result['dialog']['intro'])
        self.assertNotIn('8周',result['message'])
        self.assertNotIn('RAG',result['message'])
        time_prompt = result['dialog']['prompts'][-1]
        self.assertEqual(time_prompt['source'],'profile')
        self.assertIn('之前设置',time_prompt['text'])
        self.assertEqual([o['label'] for o in time_prompt['options']],['沿用','修改'])
        self.assertFalse(self.messages)

    def test_goal_then_confirm_profile_generates_without_extra_step(self):
        self.run_stream('制定Java学习计划，两个月')
        result = self.run_stream('目标：能独立做项目')
        self.assertEqual(result['missing_fields'], ['daily_hours'])
        self.assertEqual(len(result['dialog']['prompts']),1)
        self.assertNotIn('你希望',result['message'])
        result = self.run_stream('沿用之前的每天时间')
        self.assertIn('plan',result)
        params = self.messages[-1]['constraints']
        self.assertEqual(params['daily_hours'],6)
        self.assertEqual(params['daily_hours_source'],'profile_confirmed')
        self.assertEqual(params['duration_days'],61)

    def test_combined_plain_reply_replaces_profile_time(self):
        self.run_stream('制定Java学习计划，两个月')
        result = self.run_stream('做项目，每天1小时')
        self.assertIn('plan',result)
        params = self.messages[-1]['constraints']
        self.assertEqual(params['daily_hours'],1)
        self.assertIn('Java',params['goal'])
        self.assertIn('项目',params['goal'])
        self.assertEqual(params['duration_days'],61)

    def test_modify_time_does_not_fill_goal(self):
        self.run_stream('制定Java学习计划，两个月')
        result = self.run_stream('修改每天可用时间')
        self.assertNotIn('goal',result['plan_params'])
        self.assertNotIn('之前设置',result['message'])
        self.assertTrue(any(p['field']=='daily_hours' and p['options'][0]['label']=='30分钟' for p in result['dialog']['prompts']))

    def test_defaults_are_explicit_and_use_two_hours(self):
        first = self.run_stream('制定Java学习计划，两个月')
        self.assertIn('每天2小时',first['dialog']['default_note'])
        self.assertNotIn('周期4周',first['dialog']['default_note'])
        self.assertIn('plan',self.run_stream('其余用默认值'))
        params = self.messages[-1]['constraints']
        self.assertEqual(params['daily_hours'],2)
        self.assertEqual(params['duration_days'],61)

    def test_custom_goal_and_explicit_time(self):
        self.run_stream('制定Java学习计划，两个月')
        result = self.run_stream('自定义')
        self.assertNotIn('goal',result['plan_params'])
        result = self.run_stream('目标：完成图书管理系统，每天45分钟')
        self.assertIn('plan',result)
        self.assertEqual(self.messages[-1]['constraints']['goal'],'完成图书管理系统')

    def test_profile_time_does_not_generate_even_with_complete_goal(self):
        result = self.run_stream('制定Java学习计划，目标：完成作品，周期两个月')
        self.assertEqual(result['missing_fields'],['daily_hours'])
        self.assertFalse(self.messages)

    def test_restart_recovers_clarification(self):
        first = self.run_stream('制定Java学习计划，两个月')
        conversation_context.reset(self.uid,self.cid)
        self.c.storage.get_conversation = lambda cid: {'messages':[
            {'role':'assistant','metadata':{'plan_clarification':first}}]}
        result = self.run_stream('做项目，每天1小时')
        self.assertIn('plan',result)
        self.assertEqual(self.messages[-1]['constraints']['requested_duration'],'两个月')

    def test_new_day_count_replaces_months(self):
        self.run_stream('制定Java学习计划，两个月')
        self.run_stream('做项目，周期10天，每天1小时')
        params = self.messages[-1]['constraints']
        self.assertEqual(params['duration_days'],10)
        self.assertNotIn('duration_months',params)

    def test_bare_time_reply_is_accepted(self):
        self.run_stream('制定Java学习计划，两个月')
        self.run_stream('目标：完成作品')
        result = self.run_stream('45分钟')
        self.assertIn('plan',result)
        self.assertEqual(self.messages[-1]['constraints']['daily_hours'],.75)

    def test_invalid_time_is_not_replaced_with_profile(self):
        self.run_stream('制定Java学习计划，两个月')
        result = self.run_stream('做项目，每天0小时')
        self.assertEqual(result['missing_fields'],['daily_hours'])
        self.assertNotIn('daily_hours',result['plan_params'])
        self.assertNotIn('之前设置',result['message'])
        self.assertIn('大于 0',result['message'])

    def test_declining_profile_time_does_not_become_goal(self):
        self.run_stream('制定Java学习计划，两个月')
        result = self.run_stream('不沿用')
        self.assertNotIn('goal',result['plan_params'])
        self.assertNotIn('daily_hours',result['plan_params'])
        self.assertNotIn('之前设置',result['message'])

    def test_old_text_prompt_can_continue_after_restart(self):
        self.c.storage.get_conversation = lambda cid: {'messages':[
            {'role':'user','content':'帮我制定一下java学习计划，预计两个月'},
            {'role':'assistant','content':'已记住：学习主题（学什么）：java；计划周期：8周；每天可用时间：6.0小时。请补充：学习目标。'}]}
        result = self.run_stream('做项目，每天1小时')
        self.assertIn('plan',result)
        self.assertEqual(self.messages[-1]['constraints']['daily_hours'],1)
        self.assertEqual(self.messages[-1]['constraints']['duration_days'],61)

    def test_complete_message_skips_clarification(self):
        result = self.run_stream('制定Java学习计划，目标：完成图书管理项目，周期两个月，每天1小时')
        self.assertIn('plan',result)
        self.assertEqual(self.messages[-1]['constraints']['duration_days'],61)

    def test_nonstream_uses_same_confirmation(self):
        first = self.c.process_request('制定Java学习计划，两个月',self.uid,intent='plan',conversation_id=self.cid)
        self.assertEqual(first['missing_fields'],['goal','daily_hours'])
        final = self.c.process_request('做项目，每天1小时',self.uid,conversation_id=self.cid)
        self.assertIn('plan',final)
        self.assertEqual(self.messages[-1]['constraints']['daily_hours'],1)

    def test_calendar_period_reaches_saved_daily_plan(self):
        planner = PlannerService.__new__(PlannerService)
        planner.profile_store = SimpleNamespace(get_profile=lambda uid: {'daily_hours': 6})
        planner.storage = SimpleNamespace(save_plan=Mock())
        planner._generate_goal_path = lambda *args: None
        planner._attach_dynamic_fields = lambda plan, *args: plan
        params,_ = extract_plan_constraints('两个月，每天1小时',reference_date=date(2026,11,25))
        params['goal']='独立完成项目'
        with patch('agents.planner_service.plan_today',return_value=date(2026,11,25)), patch('agents.planner_service.track_event'):
            plan = planner.create_learning_plan('Java',self.uid,params['duration'],params)
        self.assertEqual(plan['planned_start_date'],'2026-11-25')
        self.assertEqual(plan['planned_end_date'],'2027-01-24')
        self.assertEqual(len(plan['daily_plan']),7)
        self.assertEqual(plan['stage_plan'][-1]['end_day'],61)
        self.assertEqual(plan['planning_mode'],'rolling')
        self.assertEqual(planner.storage.save_plan.call_args.args[2]['daily_hours'],1)

    def test_new_month_day_is_recalculated_on_generation(self):
        self.run_stream('制定Java学习计划，两个月')
        with patch('nlp.plan_constraints.plan_today',return_value=date(2026,11,26)):
            self.run_stream('做项目，每天1小时')
        params = self.messages[-1]['constraints']
        self.assertEqual(params['planned_start_date'],'2026-11-26')
        self.assertEqual(params['planned_end_date'],'2027-01-25')

    def test_no_profile_time_is_requested_directly(self):
        self.c.storage.profiles_store.get_profile = lambda uid: {'profile_completed':False}
        result = self.run_stream('制定Java学习计划，两个月')
        self.assertNotIn('之前设置',result['message'])
        prompt = next(p for p in result['dialog']['prompts'] if p['field']=='daily_hours')
        self.assertEqual(prompt['options'][0]['label'],'30分钟')

    def test_valid_time_can_be_set_before_goal(self):
        self.run_stream('制定Java学习计划，两个月')
        next_step = self.run_stream('沿用之前的每天时间')
        self.assertEqual(next_step['missing_fields'],['goal'])
        self.assertNotIn('goal',next_step['plan_params'])
        result = self.run_stream('做项目')
        self.assertIn('plan',result)
        self.assertEqual(self.messages[-1]['constraints']['daily_hours'],6)

    def test_every_topic_has_relevant_goal_choices(self):
        for topic in ['Java','Python','RAG','英语口语','产品经理面试','经济学']:
            with self.subTest(topic=topic):
                choices = goal_choices(topic)
                self.assertTrue(all(topic in c['message'] for c in choices))
                self.assertEqual(len(choices),3)


if __name__ == '__main__':
    unittest.main()
