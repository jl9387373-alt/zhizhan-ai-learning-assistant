"""周计划超时重试、格式兼容、约束修正及接口错误状态回归。"""
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import unittest
import urllib.error
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.plan_generation import _request_plan_json, generate_learning_week, normalize_week_response, PlanGenerationError
from agents.planner_service import PlannerService
from core.message import Message, MessageType


def week_payload():
    return {'days':[{'day':day, 'topic':'Java类和对象', 'task':f'编写第{day}个类的构造函数，并创建两个对象检查属性值',
                     'hours':2, 'deliverable':f'第{day}个类与运行记录'} for day in range(8,15)]}


def response(payload):
    res=Mock()
    res.__enter__=Mock(return_value=res)
    res.__exit__=Mock(return_value=False)
    res.read.return_value=json.dumps({'choices':[{'message':{'content':json.dumps(payload)}}]}).encode()
    return res


class TestRequestRecovery(unittest.TestCase):
    def setUp(self):
        settings={'rag.api_key':'test-only', 'rag.base_url':'https://example.invalid/v1',
                  'rag.model_name':'fake', 'rag.request_timeout':30}
        self.config=patch('agents.plan_generation.config.get',lambda key,default=None:settings.get(key,default))
        self.config.start()
        self.addCleanup(self.config.stop)

    def test_timeout_retries_and_succeeds(self):
        payload=week_payload()
        with patch('agents.plan_generation.urllib.request.urlopen',side_effect=[TimeoutError(),response(payload)]) as request:
            result=_request_plan_json('instructions',{},strict=True,attempts=2,max_tokens=2500)
        self.assertEqual(result,payload)
        self.assertEqual(request.call_count,2)
        self.assertEqual(request.call_args.kwargs['timeout'],45)
        self.assertEqual(json.loads(request.call_args.args[0].data)['max_tokens'],2500)

    def test_persistent_timeout_is_reported_as_timeout(self):
        with patch('agents.plan_generation.urllib.request.urlopen',side_effect=TimeoutError()) as request:
            with self.assertRaises(PlanGenerationError) as failure:
                _request_plan_json('instructions',{},strict=True,attempts=2)
        self.assertEqual(request.call_count,2)
        self.assertEqual(failure.exception.code,'model_timeout')
        self.assertEqual(failure.exception.http_status,504)
        self.assertTrue(failure.exception.retryable)

    def test_wrapped_timeout_not_mislabeled_as_format(self):
        with patch('agents.plan_generation.urllib.request.urlopen',side_effect=urllib.error.URLError(TimeoutError())):
            with self.assertRaises(PlanGenerationError) as failure:
                _request_plan_json('instructions',{},strict=True,attempts=1)
        self.assertEqual(failure.exception.code,'model_timeout')

    def test_auth_error_does_not_retry(self):
        error=urllib.error.HTTPError('https://example.invalid',401,'unauthorized',{},None)
        with patch('agents.plan_generation.urllib.request.urlopen',side_effect=error) as request:
            with self.assertRaises(PlanGenerationError) as failure:
                _request_plan_json('instructions',{},strict=True,attempts=2)
        self.assertEqual(request.call_count,1)
        self.assertFalse(failure.exception.retryable)
        self.assertEqual(failure.exception.code,'model_configuration_error')


class TestWeekOutputRecovery(unittest.TestCase):
    def setUp(self):
        self.plan={'subject':'Java','goal':'完成项目','duration_days':61,'daily_hours':6,'stage_plan':[]}

    def test_relative_days_and_common_field_formats(self):
        payload={'daily_plan':[{'day':f'第{i}天','topics':['Java类'], 'tasks':[f'完成示例{i}的类定义并运行检查属性值'],
                              'minutes':'120','deliverables':[f'示例{i}结果']} for i in range(1,8)]}
        entries=normalize_week_response(payload,8,14,6)
        self.assertEqual([item['day'] for item in entries],list(range(8,15)))
        self.assertTrue(all(item['estimated_hours']==2 for item in entries))

    def test_invalid_output_reprompts_with_reason(self):
        invalid=week_payload()
        invalid['days'][0]['hours']=6
        with patch('agents.plan_generation._request_plan_json',side_effect=[invalid,week_payload()]) as request:
            result=generate_learning_week(self.plan,2,{'previous_tasks':[]})
        self.assertEqual(len(result),7)
        self.assertEqual(request.call_count,2)
        self.assertIn('daily_core_hours_limit',request.call_args.args[0])
        self.assertEqual(request.call_args.kwargs['attempts'],1)

    def test_persistent_invalid_output_is_upstream_error(self):
        invalid=week_payload()
        invalid['days'][1]['task']=invalid['days'][0]['task']
        with patch('agents.plan_generation._request_plan_json',return_value=invalid):
            with self.assertRaises(PlanGenerationError) as failure:
                generate_learning_week(self.plan,2,{})
        self.assertEqual(failure.exception.code,'plan_validation_failed')
        self.assertEqual(failure.exception.http_status,502)

    def test_missing_response_does_not_run_content_validation(self):
        with patch('agents.plan_generation._request_plan_json',return_value=None), \
             patch('agents.plan_generation.normalize_week_response') as normalize:
            with self.assertRaises(PlanGenerationError) as failure:
                generate_learning_week(self.plan,2,{})
        normalize.assert_not_called()
        self.assertEqual(failure.exception.code,'model_unavailable')

    def test_budget_and_date_constraints_remain_enforced(self):
        invalid=week_payload()
        invalid['days'][0]['hours']=8
        with self.assertRaises(ValueError): normalize_week_response(invalid,8,14,6)
        invalid=week_payload()
        invalid['days'][0]['day']=9
        with self.assertRaises(ValueError): normalize_week_response(invalid,8,14,6)


class TestGenerationAPI(unittest.TestCase):
    def route(self,result):
        path=Path(__file__).resolve().parents[1]/'app.py'
        tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='generate_plan_week')
        node.decorator_list=[]
        ctx={'request':SimpleNamespace(get_json=lambda silent=True:{'user_id':'u'}),
             'coordinator':SimpleNamespace(route_message=lambda msg:SimpleNamespace(content=result)),
             'jsonify':lambda payload:payload,'Message':Message,'MessageType':MessageType}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ctx)
        return ctx['generate_plan_week']('Java',2)

    def test_upstream_status_is_not_bad_request(self):
        for status,code in [(504,'model_timeout'),(503,'model_unavailable'),(502,'plan_validation_failed')]:
            payload,status_code=self.route({'status':'error','http_status':status,'error_code':code,'message':'请重试'})
            self.assertEqual(status_code,status)
            self.assertEqual(payload['error_code'],code)
        self.assertEqual(self.route({'status':'error','message':'周数不正确'})[1],400)

    def test_success_returns_generated_plan(self):
        plan={'subject':'Java','daily_plan':[{'day':8}]}
        result=self.route({'status':'success','plan':plan,'message':'已安排第2周'})
        self.assertEqual(result['status'],'success')
        self.assertEqual(result['data'],plan)

    def test_failure_preserves_existing_progress(self):
        planner=PlannerService.__new__(PlannerService)
        plan={'subject':'Java','goal':'项目','duration_days':61,'daily_hours':6,'start_date':'2026-11-25T00:00:00',
              'detailed_weeks':[1],'weekly_plan':[{'week':1,'tasks':['语法练习']}],'daily_plan':[{'day':1}],
              'task_completion':{'1':[True]}}
        planner.storage=SimpleNamespace(get_plan=lambda *args:deepcopy(plan),save_plan=Mock())
        planner._generate_learning_week=Mock(side_effect=PlanGenerationError('超时','model_timeout',504))
        result=planner.refine_learning_week('u','Java',2)
        self.assertEqual(result['http_status'],504)
        self.assertEqual(result['error_code'],'model_timeout')
        planner.storage.save_plan.assert_not_called()


if __name__=='__main__': unittest.main()
