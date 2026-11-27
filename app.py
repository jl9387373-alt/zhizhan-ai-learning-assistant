"""
多Agent智能学习助理系统 - 主应用程序
提供Web API接口
"""
import os
from flask import Flask, request, jsonify, render_template_string, send_from_directory, send_file, Response, stream_with_context
import json
from flask_cors import CORS
from core.coordinator import AgentCoordinator
from agents.plan_generation import PlanGenerationError
from core.config import config
from core.logger import logger
from core.storage import DataStorage
from core.plan_progress import plan_progress_data, summarize_plan_progress
from core.message import Message, MessageType
from typing import Optional

from memory.chat_memory import get_conversation_memory
from core.auth import register_user, login_user, delete_user, reset_password
from storage.profiles import ProfilesStore
from storage.quiz import QuizStore
from storage.analytics import track_event
import hashlib

# ... existing code ...



# 初始化对话记忆
conversation_memory = get_conversation_memory()
# Disabled legacy direct child-agent initialization. Coordinator owns this now.

# quiz/review capabilities are managed by coordinator -> ServiceAgent
from rag.content_loader import fetch_url
import json
from datetime import datetime
from nlp.intent_detection import extract_subject as nlp_extract_subject
from nlp.plan_constraints import extract_plan_constraints, extract_plan_subject
from core.plan_dialog import build_plan_clarification

app = Flask(__name__)
CORS(app)

APP_BUILD = "20261126-technical-3"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AVATAR_DIR = os.path.join(BASE_DIR, "data", "avatars")
AVATAR_MAX_BYTES = 2 * 1024 * 1024
AVATAR_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "png", "image/png"),
    (b"\xff\xd8\xff", "jpg", "image/jpeg"),
    (b"GIF87a", "gif", "image/gif"),
    (b"GIF89a", "gif", "image/gif"),
)

CLIENT_TRACK_EVENTS = {'app_opened'}
PROFILE_AVATARS = {
    'default': 'fas fa-user',
    'graduate': 'fas fa-graduation-cap',
    'book': 'fas fa-book-open',
    'code': 'fas fa-code',
    'rocket': 'fas fa-rocket',
    'star': 'fas fa-star',
}

@app.after_request
def add_no_cache_headers(response):
    try:
        if request.path in ['/', '/index.html', '/login.html', '/register.html'] or request.path.startswith('/static/') or request.path.startswith('/api/avatar'):
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
    except Exception:
        pass
    return response

@app.route('/api/version', methods=['GET'])
def api_version():
    return jsonify({'status': 'success', 'build': APP_BUILD})

@app.route('/api/track', methods=['POST'])
def client_track():
    """接收前端埋点事件，要求携带有效 user_id。"""
    try:
        data = request.get_json(silent=True) or {}
        event = str(data.get('event') or '').strip()
        user_id = str(data.get('user_id') or '').strip()
        raw_properties = data.get('properties')
        properties = raw_properties if isinstance(raw_properties, dict) else {}
        session_id = str(data.get('session_id') or '').strip() or None

        if event not in CLIENT_TRACK_EVENTS:
            return jsonify({'status': 'error', 'message': '事件类型不支持'}), 400
        if not user_id or user_id == 'default':
            return jsonify({'status': 'error', 'message': '缺少有效 user_id'}), 400

        track_event(event, user_id, properties, session_id)
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.warning(f"记录前端埋点失败: {e}")
        return jsonify({'status': 'error', 'message': '埋点记录失败'}), 500

# 处理浏览器和工具产生的常见“噪音”请求，保持日志整洁
@app.route('/favicon.ico')
@app.route('/apple-touch-icon.png')
@app.route('/apple-touch-icon-precomposed.png')
@app.route('/.well-known/appspecific/com.chrome.devtools.json')
def silence_noise():
    """返回空响应以静默浏览器特定路径的 404 错误"""
    return '', 204

# 全局错误处理器，确保API始终返回JSON
@app.errorhandler(404)
def not_found(error):
    if request.path.startswith('/api/'):
        return jsonify({'status': 'error', 'message': 'API端点不存在'}), 404
    return error

@app.errorhandler(500)
def internal_error(error):
    if request.path.startswith('/api/'):
        logger.error(f"API内部错误: {str(error)}", exc_info=True)
        return jsonify({'status': 'error', 'message': '服务器内部错误'}), 500
    return error

@app.errorhandler(Exception)
def handle_exception(e):
    if request.path.startswith('/api/'):
        logger.error(f"未处理的异常: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'服务器错误: {str(e)}'}), 500
    raise e

# 初始化协调器
logger.info("初始化多Agent智能学习助理系统...")
coordinator = AgentCoordinator()

# 统一由 coordinator 管理综合服务能力，避免额外持有独立子 agent 实例

logger.info("系统初始化完成")

# 测验和复习Agent已经在文件开头尝试导入
def infer_subject_from_question(question: str, provided_subject: Optional[str] = None) -> str:
    """根据用户问题推断主题，优先使用显式传入的subject"""
    if provided_subject and provided_subject.strip():
        return provided_subject.strip()
    if not question:
        return "通用"
    try:
        inferred = nlp_extract_subject(question)
        return inferred or "通用"
    except Exception as e:
        logger.warning(f"根据问题推断主题失败: {e}")
        return "通用"

@app.route('/')
def index():
    """主页"""
    # 提供静态index.html文件
    static_dir = os.path.join(os.getcwd(), 'static')
    index_path = os.path.join(static_dir, 'index.html')
    
    # 如果index.html不存在，创建静态文件目录和默认文件
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
        logger.info(f"创建静态文件目录: {static_dir}")
    
    return send_from_directory('static', 'index.html')

@app.route('/index.html')
def index_page():
    """主页（显式路径）"""
    return send_from_directory('static', 'index.html')

@app.route('/login.html')
def login_page():
    """登录页"""
    return send_from_directory('static', 'login.html')

@app.route('/register.html')
def register_page():
    """注册页"""
    return send_from_directory('static', 'register.html')

# Flask已有内置的静态文件服务功能，无需手动定义路由

# API端点实现
@app.route('/api/chat', methods=['POST'])
def chat():
    """
    处理用户对话请求
    POST请求体: {
        "message": str, 用户输入的消息
        "subject": str, 学习主题（可选）
        "user_id": str, 用户ID（可选，用于个性化）
        "conversation_id": str, 会话ID（可选，用于维护对话上下文）
    }
    """
    try:
        data = request.get_json(silent=True) or {}
        message = data.get('message', '')
        subject = data.get('subject', '')
        user_id = data.get('user_id', 'default')
        conversation_id = data.get('conversation_id', '')
        
        if not message:
            return jsonify({'status': 'error', 'message': '请输入消息内容'}), 400
        
        # 如果没有会话ID，创建新会话
        storage = DataStorage()
        if not conversation_id:
            conversation_id = storage.create_conversation(user_id)
        
        # 保存用户消息到会话
        storage.add_conversation_message(conversation_id, "user", message)
        
        # 推断主题
        inferred_subject = infer_subject_from_question(message, subject)
        
        # 获取action字段
        action = data.get('action')
        
        # 调用协调器处理请求
        result = coordinator.process_request(message, user_id, action=action, conversation_id=conversation_id)
        
        # 保存助手回复到会话
        assistant_content = None
        metadata = {
            "subject": result.get("subject") or inferred_subject or "通用",
            "source": result.get("source", "default"),
        }
        if result.get("intent"):
            metadata["intent"] = result.get("intent")
        if result.get("context_snippets"):
            metadata["context_snippets"] = result.get("context_snippets")
        if result.get("plan"):
            metadata["plan"] = result.get("plan")
            if isinstance(result.get("plan"), dict) and result["plan"].get("subject"):
                metadata["subject"] = result["plan"]["subject"]
        if result.get("type") == "plan_clarification":
            metadata["plan_clarification"] = result

        if result.get("answer"):
            assistant_content = result.get("answer")
        elif result.get("message"):
            assistant_content = result.get("message")
        elif result.get("plan"):
            plan_subject = metadata.get("subject", "通用")
            assistant_content = f"{plan_subject} 学习计划详情如下："

        if assistant_content:
            storage.add_conversation_message(conversation_id, "assistant", assistant_content, {"metadata": metadata})
        
        # 在返回结果中包含会话ID
        result['conversation_id'] = conversation_id
        
        return jsonify({'status': 'success', 'data': result})
    except Exception as e:
        logger.error(f"处理对话请求失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'处理请求失败: {str(e)}'}), 500

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """流式问答接口（SSE）：同时把用户消息与助手回复落库到会话。"""
    try:
        data = request.get_json(silent=True) or {}
        message = data.get('message', '')
        user_id = data.get('user_id', 'default')
        conversation_id = data.get('conversation_id') or ''
        action = data.get('action')
        image = data.get('image')  # Base64 image
        skip_user_message = bool(data.get('skip_user_message'))

        if not message:
            def empty_stream():
                yield f"data: {json.dumps({'type': 'error', 'message': '请输入消息内容'}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            return Response(stream_with_context(empty_stream()), mimetype='text/event-stream')

        storage = DataStorage()
        # 会话ID失效（被删除/跨端）时重新建一个，避免消息写进不存在的会话
        if conversation_id and not storage.get_conversation(conversation_id):
            conversation_id = ''
        if not conversation_id:
            conversation_id = storage.create_conversation(user_id)
        # 重试/继续生成时前端不会重复展示用户气泡，服务端也不再重复落库
        if not skip_user_message:
            storage.add_conversation_message(conversation_id, "user", message)
        conversation = storage.get_conversation(conversation_id) or {}
        conversation_title = conversation.get("title") or "新对话"

        def generate():
            full_text = ''
            meta = {}
            final_data = None
            stream_failed = False
            try:
                # 先把会话ID发给前端，后续消息才会复用同一个会话
                yield f"data: {json.dumps({'type': 'session', 'data': {'conversation_id': conversation_id, 'title': conversation_title, 'user_id': user_id}}, ensure_ascii=False)}\n\n"
                for chunk in coordinator.process_request_stream(
                    message, user_id, action=action, image=image, conversation_id=conversation_id
                ):
                    if not isinstance(chunk, dict):
                        chunk = {"type": "unknown", "data": str(chunk)}
                    chunk_type = chunk.get("type")
                    if chunk_type == "token":
                        token = chunk.get("data")
                        full_text += token if isinstance(token, str) else json.dumps(token, ensure_ascii=False)
                    elif chunk_type == "meta" and isinstance(chunk.get("data"), dict):
                        meta.update(chunk["data"])
                    elif chunk_type == "complete":
                        final_data = chunk.get("data") if isinstance(chunk.get("data"), dict) else {}
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            except GeneratorExit:
                # 用户中断（暂停/取消）不保存半截回答
                raise
            except Exception as e:
                stream_failed = True
                logger.error(f"Stream generation error: {e}", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

            if not stream_failed:
                try:
                    _persist_stream_answer(storage, conversation_id, message, full_text, meta, final_data)
                except Exception as e:
                    logger.warning(f"保存流式会话消息失败: {e}")
            yield "data: [DONE]\n\n"

        return Response(stream_with_context(generate()), mimetype='text/event-stream')
    except Exception as e:
        logger.error(f"Chat stream error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


def _persist_stream_answer(storage, conversation_id: str, message: str, full_text: str,
                           meta: dict, final_data: dict) -> None:
    """保存一次流式回答。计划类回答保留 plan 元数据，历史会话可还原计划卡片。"""
    final_data = final_data or {}
    full_text = (full_text or "").strip()
    plan = final_data.get("plan") if isinstance(final_data.get("plan"), dict) else None

    if plan and plan.get("weekly_plan"):
        subject = plan.get("subject") or infer_subject_from_question(message, "")
        assistant_content = final_data.get("message") or f"{subject} 学习计划详情如下："
        assistant_meta = {
            "subject": subject,
            "source": "planner",
            "intent": "plan_creation",
            "plan": plan,
        }
    elif final_data.get("type") == "plan_clarification":
        assistant_content = final_data.get("message") or "请补充计划条件。"
        assistant_meta = {"subject": (final_data.get("plan_params") or {}).get("subject") or "通用",
                          "source": "planner", "plan_clarification": final_data}
    else:
        progress = final_data.get("progress") if isinstance(final_data.get("progress"), dict) else {}
        resources = final_data.get("resources") if isinstance(final_data.get("resources"), list) else []
        assistant_content = (
            full_text
            or (final_data.get("answer") or "").strip()
            or (final_data.get("message") or "").strip()
            or (progress.get("message") or "").strip()
        )
        if not assistant_content and resources:
            names = [str(item.get("name") or "").strip() for item in resources if isinstance(item, dict)]
            names = [name for name in names if name]
            if names:
                assistant_content = "为您推荐以下资源：" + "、".join(names[:5])
        if not assistant_content:
            return
        assistant_meta = {
            "subject": final_data.get("subject") or progress.get("subject") or infer_subject_from_question(message, ""),
            "source": meta.get("source") or meta.get("mode") or final_data.get("source") or "default",
        }
        if meta.get("context_snippets"):
            assistant_meta["context_snippets"] = meta["context_snippets"]
        if final_data.get("intent"):
            assistant_meta["intent"] = final_data["intent"]

    storage.add_conversation_message(conversation_id, "assistant", assistant_content, {"metadata": assistant_meta})


@app.route('/api/conversations', methods=['GET', 'POST', 'DELETE'])
def conversations():
    """
    会话管理API
    GET: 获取会话列表或指定会话详情
    POST: 创建新会话
    DELETE: 删除会话
    """
    try:
        user_id = request.args.get('user_id') or (request.get_json() or {}).get('user_id', 'default')
        storage = DataStorage()
        view_all = request.args.get('all') == '1' or bool((request.get_json(silent=True) or {}).get('all'))
        
        if request.method == 'GET':
            # 获取会话列表或指定会话详情
            conversation_id = request.args.get('conversation_id')
            if conversation_id:
                # 获取指定会话的详细信息（全部历史视图允许跨账号打开）
                conversation = storage.get_conversation(conversation_id)
                if conversation and (view_all or conversation.get('user_id') == user_id):
                    try:
                        import re
                        conversation_messages = conversation.get("messages", [])
                        last_assistant_index = next((idx for idx in range(len(conversation_messages) - 1, -1, -1)
                                                     if conversation_messages[idx].get("role") == "assistant"), -1)
                        for message_index, msg in enumerate(conversation_messages):
                            if msg.get("role") != "assistant":
                                continue
                            meta = msg.get("metadata") if isinstance(msg.get("metadata"), dict) else None
                            if meta and meta.get("plan"):
                                snapshot = meta["plan"]
                                live = storage.get_plan(conversation.get('user_id', user_id), snapshot.get('subject', ''))
                                if live and live.get('created_at') == snapshot.get('created_at'):
                                    meta['plan'] = live
                                continue
                            content_text = (msg.get("content") or "").strip()
                            if message_index == last_assistant_index and content_text.startswith("已记住：") and "请补充" in content_text and not (meta or {}).get("plan_clarification"):
                                previous = next((m for m in reversed(conversation_messages[:message_index]) if m.get("role") == "user"), {})
                                original_query = previous.get("content") or ""
                                params, errors = extract_plan_constraints(original_query)
                                params["subject"] = extract_plan_subject(original_query)
                                profile = storage.profiles_store.get_profile(conversation.get("user_id", user_id))
                                if profile.get("profile_completed") and not params.get("daily_hours"):
                                    params["suggested_daily_hours"] = profile.get("daily_hours", 2)
                                params["missing_fields"] = [key for key in ("subject", "goal", "duration", "daily_hours") if not params.get(key)]
                                clarification = build_plan_clarification(params, errors)
                                msg.setdefault("metadata", {})["plan_clarification"] = clarification
                                msg["content"] = clarification["message"]
                                continue
                            if "学习计划" not in content_text:
                                continue
                            m = re.search(r"已为您创建\s*(.+?)\s*的", content_text)
                            if not m:
                                continue
                            subj = m.group(1).strip()
                            if not subj:
                                continue
                            plan = storage.get_plan(conversation.get('user_id', user_id), subj)
                            if not plan:
                                continue
                            if not meta:
                                msg["metadata"] = {}
                                meta = msg["metadata"]
                            meta["plan"] = plan
                            meta.setdefault("subject", subj)
                            meta.setdefault("intent", "plan_creation")
                    except Exception as e:
                        logger.warning(f"回填会话计划信息失败: {e}")
                    return jsonify({'status': 'success', 'conversation': conversation})
                else:
                    return jsonify({'status': 'error', 'message': '会话不存在'}), 404
            else:
                # 获取会话列表（全部历史视图跨账号）
                limit = int(request.args.get('limit', 50))
                conversations_list = (
                    storage.list_all_conversations(limit) if view_all
                    else storage.list_conversations(user_id, limit)
                )
                return jsonify({'status': 'success', 'conversations': conversations_list})
        
        elif request.method == 'POST':
            # 创建新会话
            data = request.get_json(silent=True) or {}
            title = data.get('title')
            conversation_id = storage.create_conversation(user_id, title)
            return jsonify({'status': 'success', 'conversation_id': conversation_id})
        
        elif request.method == 'DELETE':
            # 删除会话：支持单个 conversation_id 或批量 conversation_ids
            data = request.get_json(silent=True) or {}
            conversation_ids = data.get('conversation_ids')
            single_id = data.get('conversation_id')
            if conversation_ids is not None:
                if not isinstance(conversation_ids, list):
                    return jsonify({'status': 'error', 'message': 'conversation_ids 必须是数组'}), 400
                ids = [str(x) for x in conversation_ids if x]
                if not ids:
                    return jsonify({'status': 'success', 'message': '无需删除', 'deleted': 0})
                deleted = sum(1 for cid in ids if storage.delete_conversation_any(cid))
                return jsonify({'status': 'success', 'message': f'已删除 {deleted} 个会话', 'deleted': deleted})
            if not single_id:
                return jsonify({'status': 'error', 'message': '请提供会话ID'}), 400

            success = (
                storage.delete_conversation_any(single_id) if view_all
                else storage.delete_conversation(single_id, user_id)
            )
            if success:
                return jsonify({'status': 'success', 'message': '删除成功'})
            else:
                return jsonify({'status': 'error', 'message': '会话不存在或删除失败'}), 404
        
    except Exception as e:
        logger.error(f"处理会话请求失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'处理请求失败: {str(e)}'}), 500

@app.route('/api/plan', methods=['POST'])
def plan():
    """
    处理学习计划请求（单数形式，兼容前端）
    """
    try:
        data = request.get_json(silent=True) or {}
        action = data.get('action', 'get_all')
        user_id = data.get('user_id', 'default')
        
        if action in ['get_all', 'get']:
            # 获取所有学习计划
            plans = coordinator.get_plans(user_id)
            return jsonify({'status': 'success', 'plans': plans})
        else:
            # 其他操作转发到/api/plans
            return plans()
    except Exception as e:
        logger.error(f"处理学习计划请求失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'处理请求失败: {str(e)}'}), 500

@app.route('/api/plans', methods=['GET', 'POST'])
def plans():
    """
    学习计划管理
    GET: 获取学习计划列表
    POST: 创建新的学习计划
    """
    try:
        if request.method == 'GET':
            # 获取学习计划列表
            user_id = request.args.get('user_id', 'default')
            subject = request.args.get('subject', '')
            plans = coordinator.get_plans(user_id, subject)
            return jsonify({'status': 'success', 'data': plans})
        elif request.method == 'POST':
            # 创建新的学习计划
            data = request.get_json(silent=True) or {}
            user_id = data.get('user_id', 'default')
            subject = data.get('subject', '')
            duration = data.get('duration', 30)  # 默认30天
            goal = data.get('goal', '')
            
            if not subject:
                return jsonify({'status': 'error', 'message': '请提供学习主题'}), 400
            
            try:
                plan = coordinator.create_plan(user_id, subject, duration, goal, data.get('daily_hours'))
            except PlanGenerationError as exc:
                return jsonify({'status': 'error', 'message': str(exc), 'code': exc.code}), exc.http_status
            except (ValueError, TypeError) as exc:
                return jsonify({'status': 'error', 'message': str(exc)}), 400
            if not plan:
                return jsonify({'status': 'error', 'message': '学习计划暂未生成，请稍后重试。'}), 503
            return jsonify({'status': 'success', 'data': plan}), 201
    except Exception as e:
        logger.error(f"处理学习计划请求失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'处理请求失败: {str(e)}'}), 500

@app.route('/api/plans/<plan_id>', methods=['GET', 'PUT', 'DELETE'])
def plan_detail(plan_id):
    """
    学习计划详情管理
    GET: 获取学习计划详情
    PUT: 更新学习计划
    DELETE: 删除学习计划
    """
    try:
        user_id = request.args.get('user_id', 'default')
        
        if request.method == 'GET':
            # 获取学习计划详情
            plan = coordinator.get_plan_detail(user_id, plan_id)
            if not plan:
                return jsonify({'status': 'error', 'message': '学习计划不存在'}), 404
            return jsonify({'status': 'success', 'data': plan})
        elif request.method == 'PUT':
            # 更新学习计划
            data = request.get_json(silent=True) or {}
            plan = coordinator.update_plan(user_id, plan_id, data)
            if not plan:
                return jsonify({'status': 'error', 'message': '学习计划不存在'}), 404
            return jsonify({'status': 'success', 'data': plan})
        elif request.method == 'DELETE':
            # 删除学习计划
            success = coordinator.delete_plan(user_id, plan_id)
            if not success:
                return jsonify({'status': 'error', 'message': '学习计划不存在'}), 404
            return jsonify({'status': 'success', 'message': '学习计划已删除'})
    except Exception as e:
        logger.error(f"处理学习计划详情请求失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'处理请求失败: {str(e)}'}), 500

@app.route('/api/plans/<plan_id>/weeks/<int:week>/generate', methods=['POST'])
def generate_plan_week(plan_id, week):
    data = request.get_json(silent=True) or {}
    response = coordinator.route_message(Message(
        sender='app', receiver='planner', msg_type=MessageType.REQUEST,
        content={'action':'generate_week', 'user_id':data.get('user_id','default'),
                 'subject':plan_id, 'week':week}
    )).content
    if response.get('status') != 'success':
        return jsonify(response), response.get('http_status', 400)
    return jsonify({'status':'success','data':response['plan'],'message':response['message']})


def _sync_checked_plan(storage, user_id, plan_id):
    # 勾选记录已经保存；同步失败不能把成功的勾选显示成未完成。
    try:
        plan = storage.get_plan(user_id, plan_id)
        response = coordinator.route_message(Message(
            sender="app", receiver="progress", msg_type=MessageType.REQUEST,
            content={"action": "update", "user_id": user_id, "subject": plan_id,
                     "progress_data": plan_progress_data(plan), "detail": True}
        ))
        if response and response.content.get('status') != 'success':
            raise RuntimeError(response.content.get('message', '进度同步失败'))
    except Exception:
        logger.error("同步已保存的计划任务失败", exc_info=True)
        return '任务已保存，进度汇总暂未同步，可点击同步进度重试。'
    return None


def _checked_plan_response(storage, user_id, plan_id, warning=None):
    plan = storage.get_plan(user_id, plan_id)
    plan['progress_summary'] = summarize_plan_progress(plan)
    plan['progress'] = plan['progress_summary']['detailed_task_percentage'] or 0
    return jsonify({'status': 'success', 'data': plan, 'sync_warning': warning})


@app.route('/api/plans/<plan_id>/tasks/toggle', methods=['POST'])
def toggle_plan_task(plan_id):
    try:
        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id', 'default')
        week = str(data.get('week', '')).strip()
        task_index = data.get('task_index')
        completed = data.get('completed')
        if not week or type(task_index) is not int or type(completed) is not bool:
            return jsonify({'status': 'error', 'message': '请提供周数、整数任务索引和布尔完成状态'}), 400
        storage = DataStorage()
        try:
            plan = storage.plans_store.set_task_completed(
                user_id, plan_id, week, task_index, completed, data.get('plan_created_at'))
        except ValueError as exc:
            status = 409 if '重新生成' in str(exc) else 400
            return jsonify({'status': 'error', 'message': str(exc)}), status
        if not plan:
            return jsonify({'status': 'error', 'message': '学习计划不存在'}), 404
        track_event('task_completed' if completed else 'task_uncompleted', user_id,
                    {'plan_key': plan_id, 'plan_id': plan_id, 'week': week,
                     'task_index': task_index, 'task_key': f'{plan_id}|{week}|{task_index}',
                     'progress': plan['progress'], 'plan_created_at': plan.get('created_at')})
        warning = _sync_checked_plan(storage, user_id, plan_id)
        return _checked_plan_response(storage, user_id, plan_id, warning)
    except Exception as exc:
        logger.error('更新计划任务失败', exc_info=True)
        return jsonify({'status': 'error', 'message': f'更新失败: {exc}'}), 500


@app.route('/api/plans/<plan_id>/tasks/sync', methods=['POST'])
def sync_plan_tasks(plan_id):
    try:
        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id', 'default')
        storage = DataStorage()
        if not storage.get_plan(user_id, plan_id):
            return jsonify({'status': 'error', 'message': '学习计划不存在'}), 404
        warning = _sync_checked_plan(storage, user_id, plan_id)
        return _checked_plan_response(storage, user_id, plan_id, warning)
    except Exception as exc:
        logger.error('同步计划任务失败', exc_info=True)
        return jsonify({'status': 'error', 'message': f'同步失败: {exc}'}), 500

@app.route('/api/resources', methods=['GET', 'POST'])
def resources():
    """
    学习资源管理
    GET: 获取学习资源列表
    POST: 推荐学习资源或获取所有资源
    """
    try:
        if request.method == 'GET':
            # 获取学习资源列表
            user_id = request.args.get('user_id', 'default')
            subject = request.args.get('subject', '')
            result = coordinator.get_resources(user_id, subject)
            # 返回格式匹配前端期望：{status: 'success', resources: [...]}
            if result.get('status') == 'success' and 'resources' in result:
                return jsonify(result)
            elif result.get('status') == 'success' and 'data' in result:
                # 如果返回的是data格式，提取resources
                data = result.get('data', {})
                if 'resources' in data:
                    return jsonify({'status': 'success', 'resources': data['resources']})
            return jsonify(result)
        elif request.method == 'POST':
            # 处理POST请求
            data = request.get_json(silent=True) or {}
            user_id = data.get('user_id', 'default')
            subject = data.get('subject', '')
            level = data.get('level', 'beginner')  # 默认初级
            
            # 如果subject是'all'或为空，获取所有资源
            if not subject or subject == 'all':
                result = coordinator.get_resources(user_id, '')
            else:
                result = coordinator.recommend_resources(user_id, subject, level)
            
            # 返回格式匹配前端期望：{status: 'success', resources: [...]}
            if result.get('status') == 'success' and 'resources' in result:
                return jsonify(result)
            elif result.get('status') == 'success' and 'data' in result:
                # 如果返回的是data格式，提取resources
                data_obj = result.get('data', {})
                if 'resources' in data_obj:
                    return jsonify({'status': 'success', 'resources': data_obj['resources']})
            return jsonify(result)
    except Exception as e:
        logger.error(f"处理学习资源请求失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'处理请求失败: {str(e)}'}), 500

@app.route('/api/progress', methods=['GET', 'POST'])
@app.route('/api/progress/<subject>', methods=['DELETE'])
def progress(subject=None):
    """
    学习进度管理
    GET: 获取学习进度
    POST: 更新学习进度或获取所有进度
    DELETE: 删除特定主题的学习进度
    """
    try:
        if request.method == 'GET':
            # 获取学习进度
            user_id = request.args.get('user_id', 'default')
            subject = request.args.get('subject', '')
            # 如果没有subject，返回所有进度
            if not subject:
                storage = DataStorage()
                all_progress = storage.get_all_progress(user_id)
                return jsonify({'status': 'success', 'data': all_progress})
            # 有subject，获取特定主题的进度（通过ProgressAgent）
            message = Message(
                sender="api",
                receiver="progress",
                msg_type=MessageType.REQUEST,
                content={
                    "action": "get",
                    "user_id": user_id,
                    "subject": subject
                }
            )
            response = coordinator.route_message(message)
            result = response.content
            if result.get('status') == 'success':
                return jsonify({'status': 'success', 'data': result.get('progress', {})})
            return jsonify(result)
        elif request.method == 'POST':
            data = request.get_json(silent=True) or {}
            user_id = data.get('user_id', 'default')
            action = data.get('action', 'update')
            subject = data.get('subject', '')

            if action in ['get_all', 'get']:
                storage = DataStorage()
                all_progress = storage.get_all_progress(user_id)
                return jsonify({'status': 'success', 'progress': all_progress, 'data': all_progress})

            if action in ['dashboard', 'report']:
                if not subject:
                    return jsonify({'status': 'error', 'message': '请提供学习主题'}), 400
                message = Message(
                    sender="api",
                    receiver="progress",
                    msg_type=MessageType.REQUEST,
                    content={
                        "action": action,
                        "user_id": user_id,
                        "subject": subject,
                        "period": data.get('period', 'weekly')
                    }
                )
                response = coordinator.route_message(message)
                return jsonify(response.content)

            completed_tasks = data.get('completed_tasks', 0)
            total_tasks = data.get('total_tasks', 0)
            study_hours = data.get('study_hours', 0)

            if not subject:
                return jsonify({'status': 'error', 'message': '请提供学习主题'}), 400

            progress_data = {
                "completed_tasks": completed_tasks,
                "total_tasks": total_tasks,
                "study_time": study_hours * 60 if study_hours else 0,
                "date": datetime.now().strftime("%Y-%m-%d"),
                # 手动录入仅当总任务数 > 0 时才计算百分比
                "calculation_basis": {
                    "verifiable": bool(total_tasks and total_tasks > 0),
                    "total_tasks_source": "manual",
                    "completed_tasks_source": "manual",
                },
            }
            message = Message(
                sender="api",
                receiver="progress",
                msg_type=MessageType.REQUEST,
                content={
                    "action": "update",
                    "user_id": user_id,
                    "subject": subject,
                    "progress_data": progress_data
                }
            )
            response = coordinator.route_message(message)
            result = response.content
            if result.get('status') == 'success':
                return jsonify({'status': 'success', 'data': result.get('progress', {}), 'progress': result.get('progress', {})})
            return jsonify(result)
        elif request.method == 'DELETE':
            # 处理DELETE请求：user_id 可能在 query，也可能在 body（无 body 时不能直接读 request.json）
            payload = request.get_json(silent=True) or {}
            user_id = request.args.get('user_id') or payload.get('user_id') or 'default'
            if not subject:
                return jsonify({'status': 'error', 'message': '请提供学习主题'}), 400
            storage = DataStorage()
            success = storage.delete_progress(user_id, subject)
            if success:
                return jsonify({'status': 'success', 'message': '学习进度已删除'})
            return jsonify({'status': 'error', 'message': '学习进度不存在或已删除'}), 404
    except Exception as e:
        logger.error(f"处理学习进度请求失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'处理请求失败: {str(e)}'}), 500

@app.route('/api/notes', methods=['GET', 'POST'])
def notes():
    """
    学习笔记管理
    GET: 获取学习笔记列表
    POST: 创建学习笔记
    """
    try:
        if request.method == 'GET':
            # 获取学习笔记列表
            user_id = request.args.get('user_id', 'default')
            subject = request.args.get('subject', '')
            notes = coordinator.get_notes(user_id, subject)
            return jsonify({'status': 'success', 'data': notes})
        elif request.method == 'POST':
            # 创建学习笔记
            data = request.get_json(silent=True) or {}
            user_id = data.get('user_id', 'default')
            subject = data.get('subject', '')
            title = data.get('title', '')
            content = data.get('content', '')
            
            if not subject or not title:
                return jsonify({'status': 'error', 'message': '请提供学习主题和笔记标题'}), 400
            
            note = coordinator.create_note(user_id, subject, title, content)
            return jsonify({'status': 'success', 'data': note}), 201
    except Exception as e:
        logger.error(f"处理学习笔记请求失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'处理请求失败: {str(e)}'}), 500

@app.route('/api/notes/<note_id>', methods=['GET', 'PUT', 'DELETE'])
def note_detail(note_id):
    """
    学习笔记详情管理
    GET: 获取学习笔记详情
    PUT: 更新学习笔记
    DELETE: 删除学习笔记
    """
    try:
        user_id = request.args.get('user_id', 'default')
        
        if request.method == 'GET':
            # 获取学习笔记详情
            note = coordinator.get_note_detail(user_id, note_id)
            if not note:
                return jsonify({'status': 'error', 'message': '学习笔记不存在'}), 404
            return jsonify({'status': 'success', 'data': note})
        elif request.method == 'PUT':
            # 更新学习笔记
            data = request.get_json(silent=True) or {}
            note = coordinator.update_note(user_id, note_id, data)
            if not note:
                return jsonify({'status': 'error', 'message': '学习笔记不存在'}), 404
            return jsonify({'status': 'success', 'data': note})
        elif request.method == 'DELETE':
            # 删除学习笔记
            success = coordinator.delete_note(user_id, note_id)
            if not success:
                return jsonify({'status': 'error', 'message': '学习笔记不存在'}), 404
            return jsonify({'status': 'success', 'message': '学习笔记已删除'})
    except Exception as e:
        logger.error(f"处理学习笔记详情请求失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'处理请求失败: {str(e)}'}), 500

@app.route('/api/quiz', methods=['POST'])
def quiz():
    """
    知识测验
    POST: 生成或评估测验
    """
    try:
        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id', 'default')
        subject = data.get('subject', '')
        action = data.get('action', 'generate')
        quiz_data = data.get('quiz_data', {})
        
        if not subject:
            return jsonify({'status': 'error', 'message': '请提供学习主题'}), 400

        if not coordinator.get_agent("quiz"):
            return jsonify({'status': 'error', 'message': '测验功能未启用'}), 501

        if action == 'generate':
            difficulty = data.get('difficulty') or data.get('level') or "medium"
            num_questions = int(data.get('num_questions') or data.get('question_count') or data.get('questions_count') or 5)
            result = coordinator.route_message(Message(
                sender="api",
                receiver="quiz",
                msg_type=MessageType.REQUEST,
                content={
                    "action": "generate",
                    "subject": subject,
                    "difficulty": difficulty,
                    "num_questions": num_questions,
                    "user_id": user_id
                }
            )).content
            return jsonify(result)
        elif action in ['submit', 'evaluate']:
            quiz_id = data.get('quiz_id') or quiz_data.get('quiz_id')
            answers = data.get('answers') or quiz_data.get('answers') or []
            result = coordinator.route_message(Message(
                sender="api",
                receiver="quiz",
                msg_type=MessageType.REQUEST,
                content={
                    "action": "submit",
                    "quiz_id": quiz_id,
                    "answers": answers,
                    "user_id": user_id
                }
            )).content
            return jsonify(result)
        elif action == 'get':
            quiz_id = data.get('quiz_id')
            result = coordinator.route_message(Message(
                sender="api",
                receiver="quiz",
                msg_type=MessageType.REQUEST,
                content={
                    "action": "get",
                    "quiz_id": quiz_id,
                    "user_id": user_id
                }
            )).content
            return jsonify(result)
        else:
            return jsonify({'status': 'error', 'message': '无效的操作类型'}), 400
    except Exception as e:
        logger.error(f"处理测验请求失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'处理请求失败: {str(e)}'}), 500


@app.route('/api/quiz/generate', methods=['POST'])
def quiz_generate():
    try:
        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id', 'default')
        subject = (data.get('subject') or "").strip()
        difficulty = data.get('difficulty', 'medium')
        questions_count = int(data.get('questions_count') or data.get('num_questions') or 5)

        if not subject:
            return jsonify({'status': 'error', 'message': '请提供测验主题'}), 400
        if not coordinator.get_agent("quiz"):
            return jsonify({'status': 'error', 'message': '测验功能未启用'}), 501

        result = coordinator.route_message(Message(
            sender="api",
            receiver="quiz",
            msg_type=MessageType.REQUEST,
            content={
                "action": "generate",
                "subject": subject,
                "difficulty": difficulty,
                "num_questions": questions_count,
                "user_id": user_id
            }
        )).content

        if result.get("status") != "success":
            return jsonify(result)

        quiz_id = result.get("quiz_id")
        questions = result.get("questions", []) or []
        ui_questions = []
        for q in questions:
            ui_questions.append({
                "id": q.get("question_id"),
                "text": q.get("question", ""),
                "type": q.get("type", "multiple_choice"),
                "options": q.get("options", [])
            })
        quiz_obj = {
            "id": quiz_id,
            "title": f"测验：{subject}",
            "subject": subject,
            "difficulty": result.get("difficulty", difficulty),
            "questions": ui_questions
        }
        return jsonify({'status': 'success', 'quiz': quiz_obj})
    except Exception as e:
        logger.error(f"测验生成失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'测验生成失败: {str(e)}'}), 500


@app.route('/api/quiz/submit', methods=['POST'])
def quiz_submit():
    try:
        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id', 'default')
        quiz_id = data.get('quiz_id')
        answers_payload = data.get('answers') or []

        if not quiz_id:
            return jsonify({'status': 'error', 'message': '缺少quiz_id'}), 400
        if not coordinator.get_agent("quiz"):
            return jsonify({'status': 'error', 'message': '测验功能未启用'}), 501

        storage = DataStorage()
        quiz_record = storage.get_quiz_by_id(user_id, quiz_id)
        if not quiz_record:
            return jsonify({'status': 'error', 'message': '测验不存在'}), 404

        answers_map = {}
        for a in answers_payload:
            if not isinstance(a, dict):
                continue
            qid = a.get("question_id")
            if qid is None:
                continue
            answers_map[str(qid)] = a

        ordered_answers = []
        for q in quiz_record.get("questions", []) or []:
            qid = q.get("question_id")
            qtype = q.get("type")
            a = answers_map.get(str(qid), {})
            if qtype == "short_answer":
                ordered_answers.append((a.get("user_answer") or "").strip())
            else:
                sel = a.get("selected")
                if sel is None:
                    ordered_answers.append(None)
                else:
                    try:
                        ordered_answers.append(int(sel))
                    except Exception:
                        ordered_answers.append(sel)

        submit_result = coordinator.route_message(Message(
            sender="api",
            receiver="quiz",
            msg_type=MessageType.REQUEST,
            content={
                "action": "submit",
                "quiz_id": quiz_id,
                "answers": ordered_answers,
                "user_id": user_id
            }
        )).content

        if submit_result.get("status") != "success":
            return jsonify(submit_result)

        feedback = submit_result.get("feedback", []) or []
        fb_by_index = {f.get("question_index"): f for f in feedback if isinstance(f, dict)}
        ui_questions = []
        questions = quiz_record.get("questions", []) or []
        for idx, q in enumerate(questions):
            f = fb_by_index.get(idx, {})
            user_answer = ordered_answers[idx] if idx < len(ordered_answers) else None
            item = {
                "id": q.get("question_id"),
                "text": q.get("question", ""),
                "type": q.get("type", "multiple_choice"),
                "options": q.get("options", []),
                "correct": bool(f.get("is_correct")),
                "correct_answer": f.get("correct_answer"),
                "explanation": f.get("explanation", "")
            }
            if item["type"] == "short_answer":
                item["user_answer"] = user_answer if isinstance(user_answer, str) else ""
            else:
                item["selected"] = user_answer
            ui_questions.append(item)

        return jsonify({
            "status": "success",
            "result": {
                "quiz_id": quiz_id,
                "score": submit_result.get("score"),
                "questions": ui_questions
            }
        })
    except Exception as e:
        logger.error(f"提交测验失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'提交测验失败: {str(e)}'}), 500


@app.route('/api/quiz/detail', methods=['GET'])
def quiz_detail():
    try:
        user_id = request.args.get('user_id', 'default')
        quiz_id = request.args.get('quiz_id')
        if not quiz_id:
            return jsonify({'status': 'error', 'message': '缺少quiz_id'}), 400

        storage = DataStorage()
        quiz_record = storage.get_quiz_by_id(user_id, quiz_id)
        if not quiz_record:
            return jsonify({'status': 'error', 'message': '测验不存在'}), 404

        questions = quiz_record.get("questions", []) or []
        answers = quiz_record.get("answers", []) or []
        feedback = quiz_record.get("feedback", []) or []
        fb_by_index = {f.get("question_index"): f for f in feedback if isinstance(f, dict)}

        ui_questions = []
        for idx, q in enumerate(questions):
            qtype = q.get("type", "multiple_choice")
            user_answer = answers[idx] if idx < len(answers) else None
            f = fb_by_index.get(idx, {}) or {}
            item = {
                "id": q.get("question_id"),
                "text": q.get("question", ""),
                "type": qtype,
                "options": q.get("options", []),
                "correct": bool(f.get("is_correct")),
                "correct_answer": f.get("correct_answer"),
                "explanation": f.get("explanation", ""),
            }
            if qtype == "short_answer":
                item["user_answer"] = user_answer if isinstance(user_answer, str) else ""
            else:
                item["selected"] = user_answer
            ui_questions.append(item)

        result = {
            "quiz_id": quiz_id,
            "subject": quiz_record.get("subject"),
            "difficulty": quiz_record.get("difficulty"),
            "score": quiz_record.get("score"),
            "created_at": quiz_record.get("created_at"),
            "completed_at": quiz_record.get("completed_at"),
            "questions": ui_questions,
        }

        return jsonify({'status': 'success', 'result': result})
    except Exception as e:
        logger.error(f"获取测验详情失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'获取测验详情失败: {str(e)}'}), 500

@app.route('/api/review', methods=['POST'])
def review():
    """
    智能复习
    POST: 生成复习内容
    """
    try:
        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id', 'default')
        subject = data.get('subject', '')
        content = data.get('content', '')
        importance = data.get('importance', 'medium')
        
        if not subject:
            return jsonify({'status': 'error', 'message': '请提供学习主题'}), 400
        
        review_agent = coordinator.get_agent('review')
        if review_agent is None:
            return jsonify({'status': 'error', 'message': '复习功能未启用'}), 501
        
        result = review_agent.schedule_review(
            user_id, subject, content, importance, data.get('first_review_at')
        )
        if result.get('status') == 'success':
            return jsonify(result)
        return jsonify(result), 400
    except Exception as e:
        logger.error(f"处理复习请求失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'处理请求失败: {str(e)}'}), 500

@app.route('/api/review/all', methods=['GET'])
def get_all_reviews():
    """
    获取所有复习计划
    GET: 获取用户的所有复习计划
    """
    try:
        user_id = request.args.get('user_id', 'default')
        storage = DataStorage()
        
        # 获取复习历史
        reviews = storage.get_review_history(user_id)
        
        return jsonify({'status': 'success', 'reviews': reviews})
    except Exception as e:
        logger.error(f"获取复习计划失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'获取复习计划失败: {str(e)}'}), 500

@app.route('/api/review/history', methods=['GET'])
def get_review_history():
    """
    获取复习历史
    GET: 获取用户的所有复习历史记录
    """
    try:
        user_id = request.args.get('user_id', 'default')
        limit = int(request.args.get('limit', 50))
        review_agent = coordinator.get_agent('review')
        if review_agent is None:
            return jsonify({'status': 'error', 'message': '复习功能未启用'}), 501
        result = review_agent.get_review_history(user_id, limit)
        if result.get('status') == 'success':
            return jsonify(result)
        return jsonify(result), 400
    except Exception as e:
        logger.error(f"获取复习历史失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'获取复习历史失败: {str(e)}'}), 500

@app.route('/api/clear_memory', methods=['POST'])
def clear_memory():
    """
    清除对话记忆API
    """
    try:
        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id', 'default')
        session_id = data.get('session_id', '')
        
        # 清除对话记忆
        success = conversation_memory.clear(user_id, session_id)
        
        if success:
            return jsonify({'status': 'success', 'message': '对话记忆已清除'})
        else:
            return jsonify({'status': 'error', 'message': '清除对话记忆失败'}), 500
    except Exception as e:
        logger.error(f"清除对话记忆失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'清除对话记忆失败: {str(e)}'}), 500

@app.route('/api/qa/history', methods=['GET', 'DELETE'])
def qa_history():
    """
    获取或删除问答历史
    GET: 获取问答历史
    DELETE: 删除问答历史记录
    """
    try:
        if request.method == 'GET':
            user_id = request.args.get('user_id', 'default')
        else:
            user_id = (request.get_json() or {}).get('user_id', 'default')
        
        if request.method == 'GET':
            limit = int(request.args.get('limit', 50))
            if request.args.get('all') == '1':
                # 全部历史视图：跨账号返回本机所有问答记录
                history = DataStorage().get_qa_history_all(limit)
                return jsonify({'status': 'success', 'history': history})
            result = coordinator.get_qa_history(user_id, limit)
            # 返回格式匹配前端期望：{status: 'success', history: [...]}
            if result.get('status') == 'success' and 'data' in result:
                return jsonify({'status': 'success', 'history': result['data']})
            elif result.get('status') == 'success' and 'history' in result:
                # 如果已经是history字段，直接返回
                return jsonify(result)
            return jsonify(result)
        elif request.method == 'DELETE':
            data = request.get_json(silent=True) or {}
            timestamp = data.get('timestamp')
            question = data.get('question')
            
            if not timestamp:
                return jsonify({'status': 'error', 'message': '请提供时间戳'}), 400
            
            if (request.get_json(silent=True) or {}).get('all'):
                success = DataStorage().delete_qa_history_any(timestamp, question)
                return jsonify({'status': 'success' if success else 'error',
                                'message': '删除成功' if success else '记录不存在或删除失败'})
            result = coordinator.delete_qa_history(user_id, timestamp, question)
            return jsonify(result)
    except Exception as e:
        logger.error(f"处理问答历史请求失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'处理请求失败: {str(e)}'}), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id', 'default')
        conversation_id = data.get('conversation_id', '')
        question = (data.get('question') or '').strip()
        answer = (data.get('answer') or '').strip()
        rating = int(data.get('rating', 0))
        comment = (data.get('comment') or '').strip()
        source = data.get('source') or data.get('mode') or ''
        context_snippets = data.get('context_snippets') or []

        if rating not in (-1, 1):
            return jsonify({'status': 'error', 'message': 'rating 只允许为 1 或 -1'}), 400
        if not question or not answer:
            return jsonify({'status': 'error', 'message': '请提供 question 和 answer'}), 400

        storage = DataStorage()
        item = storage.save_feedback({
            "user_id": user_id,
            "conversation_id": conversation_id,
            "question": question,
            "answer": answer,
            "rating": rating,
            "comment": comment,
            "source": source,
            "context_snippets": context_snippets
        })
        feedback_key = hashlib.sha1(
            f"{user_id}|{question}|{answer}".encode("utf-8")
        ).hexdigest()[:16]
        track_event(
            "qa_helpful" if rating == 1 else "qa_unhelpful",
            user_id,
            {
                "feedback_key": feedback_key,
                "conversation_id": conversation_id,
                "source": source,
                "question_length": len(question),
            },
        )
        return jsonify({'status': 'success', 'data': item})
    except Exception as e:
        logger.error(f"提交反馈失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'提交反馈失败: {str(e)}'}), 500

@app.route('/api/review/<review_id>/reschedule', methods=['POST'])
def reschedule_review_api(review_id):
    """调整复习计划的下次复习时间"""
    try:
        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id', 'default')
        review_agent = coordinator.get_agent('review')
        if not review_agent:
            return jsonify({'status': 'error', 'message': '复习系统未就绪'}), 503

        result = review_agent.reschedule_review(
            user_id,
            review_id,
            next_review_at=data.get('next_review_at'),
            postpone_days=data.get('postpone_days'),
        )
        if result.get('status') == 'success':
            return jsonify(result)
        return jsonify(result), 400
    except Exception as e:
        logger.error(f"调整复习时间失败: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/review/next', methods=['GET'])
def next_review():
    """
    获取下一个复习内容
    """
    try:
        user_id = request.args.get('user_id', 'default')
        review_agent = coordinator.get_agent('review')
        if review_agent is None:
            return jsonify({'status': 'error', 'message': '复习功能未启用'}), 501
        result = review_agent.get_next_review(user_id)
        if result.get('status') == 'success':
            return jsonify(result)
        return jsonify(result), 400
    except Exception as e:
        logger.error(f"处理下一个复习内容请求失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'处理请求失败: {str(e)}'}), 500

@app.route('/api/review/due', methods=['GET'])
def review_due():
    """获取所有到期复习项（主动提醒列表）"""
    try:
        user_id = request.args.get('user_id', 'default')
        from storage.review import ReviewStore
        store = ReviewStore()
        result = store.get_due_reviews(user_id)
        overdue_reviews = result.get('overdue', []) or []
        next_review = result.get('next')
        review_ids = []
        for item in overdue_reviews + ([next_review] if next_review else []):
            review_id = (item.get('review_id') or item.get('id')) if isinstance(item, dict) else item
            if review_id:
                review_ids.append(str(review_id))
        track_event('review_due', user_id, {
            'due_count': len(review_ids),
            'overdue_count': len(overdue_reviews),
            'review_ids': review_ids[:100],
        })
        return jsonify({
            'status': 'success',
            'overdue': overdue_reviews,
            'next': next_review
        })
    except Exception as e:
        logger.error(f"获取到期复习失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/quiz/history', methods=['GET'])
def quiz_history():
    """
    获取测验历史
    """
    try:
        user_id = request.args.get('user_id', 'default')
        limit = int(request.args.get('limit', 5))
        result = coordinator.get_quiz_history(user_id, limit)
        # 返回格式匹配前端期望：{status: 'success', history: [...]}
        if result.get('status') == 'success' and 'data' in result:
            return jsonify({'status': 'success', 'history': result['data']})
        elif result.get('status') == 'success' and 'history' in result:
            # 如果已经是history字段，直接返回
            return jsonify(result)
        return jsonify(result)
    except Exception as e:
        logger.error(f"处理测验历史请求失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'处理请求失败: {str(e)}'}), 500

@app.route('/api/quiz/delete', methods=['POST'])
def quiz_delete():
    """
    删除测验记录
    """
    try:
        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id', 'default')
        quiz_id = data.get('quiz_id')
        
        if not quiz_id:
            return jsonify({'status': 'error', 'message': '缺少quiz_id'}), 400
        
        quiz_store = QuizStore()
        success = quiz_store.delete(user_id, quiz_id)
        if success:
            return jsonify({'status': 'success', 'message': '删除成功'})
        else:
            return jsonify({'status': 'error', 'message': '删除失败'}), 500
    except Exception as e:
        logger.error(f"处理删除测验请求失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'处理请求失败: {str(e)}'}), 500

def _avatar_user_key(user_id: str) -> str:
    """用哈希生成头像文件名，避免 user_id 中的特殊字符进入文件路径。"""
    return hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()[:32]


def _detect_avatar_type(data: bytes):
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return "webp", "image/webp"
    for signature, ext, mimetype in AVATAR_SIGNATURES:
        if data.startswith(signature):
            return ext, mimetype
    return None


def _avatar_path_for(user_id: str, ext: str) -> str:
    return os.path.join(AVATAR_DIR, f"{_avatar_user_key(user_id)}.{ext}")


def _find_avatar_file(user_id: str):
    for ext, mimetype in (("png", "image/png"), ("jpg", "image/jpeg"),
                          ("gif", "image/gif"), ("webp", "image/webp")):
        path = _avatar_path_for(user_id, ext)
        if os.path.exists(path):
            return path, mimetype
    return None, None


def _save_avatar_file(user_id: str, data: bytes, ext: str) -> str:
    os.makedirs(AVATAR_DIR, exist_ok=True)
    target_path = _avatar_path_for(user_id, ext)
    temp_path = target_path + ".tmp"
    with open(temp_path, "wb") as f:
        f.write(data)
    os.replace(temp_path, target_path)

    for old_ext in ("png", "jpg", "gif", "webp"):
        old_path = _avatar_path_for(user_id, old_ext)
        if old_ext != ext and os.path.exists(old_path):
            os.remove(old_path)
    return target_path


@app.route('/api/avatar', methods=['GET'])
def api_avatar():
    """读取用户上传的头像。"""
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'status': 'error', 'message': '缺少user_id'}), 400

    path, mimetype = _find_avatar_file(user_id)
    if not path:
        return jsonify({'status': 'error', 'message': '头像不存在'}), 404
    return send_file(path, mimetype=mimetype)


@app.route('/api/profile/avatar', methods=['POST'])
def api_upload_avatar():
    """上传本地图片作为头像，并更新用户画像。"""
    try:
        user_id = request.form.get('user_id')
        if not user_id:
            return jsonify({'status': 'error', 'message': '缺少user_id'}), 400

        nickname = None
        raw_nickname = request.form.get('nickname')
        if raw_nickname is not None:
            nickname = str(raw_nickname).strip()
            if not nickname or len(nickname) > 20:
                return jsonify({'status': 'error', 'message': '昵称需为 1-20 个字符'}), 400
            if any(ch in nickname for ch in ('\n', '\r', '\t')):
                return jsonify({'status': 'error', 'message': '昵称不能包含换行或制表符'}), 400

        uploaded = request.files.get('file') or request.files.get('avatar')
        if uploaded is None or not uploaded.filename:
            return jsonify({'status': 'error', 'message': '请选择图片'}), 400

        data = uploaded.read(AVATAR_MAX_BYTES + 1)
        if not data:
            return jsonify({'status': 'error', 'message': '图片内容为空'}), 400
        if len(data) > AVATAR_MAX_BYTES:
            return jsonify({'status': 'error', 'message': '头像不能超过 2MB'}), 400

        detected = _detect_avatar_type(data)
        if not detected:
            return jsonify({'status': 'error', 'message': '仅支持 PNG、JPG、GIF、WEBP 图片'}), 400
        ext, _mimetype = detected

        _save_avatar_file(user_id, data, ext)

        store = ProfilesStore()
        profile = store.get_profile(user_id)
        if nickname is not None:
            profile['nickname'] = nickname
        profile['avatar'] = 'custom'
        profile['avatar_version'] = datetime.now().isoformat()
        store.save_profile(user_id, profile)

        return jsonify({'status': 'success', 'message': '头像上传成功', 'data': profile})
    except Exception as e:
        logger.error(f"上传头像失败: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'上传失败: {str(e)}'}), 500


@app.route('/api/auth/profile', methods=['GET', 'POST'])
def api_auth_profile():
    """获取或更新用户画像"""
    try:
        if request.method == 'GET':
            user_id = request.args.get('user_id')
            if not user_id:
                return jsonify({'status': 'error', 'message': '缺少user_id'}), 400
                
            store = ProfilesStore()
            profile = store.get_profile(user_id)
            return jsonify({'status': 'success', 'data': profile})
            
        elif request.method == 'POST':
            data = request.get_json(silent=True) or {}
            user_id = data.get('user_id')
            if not user_id:
                return jsonify({'status': 'error', 'message': '缺少user_id'}), 400

            nickname = None
            if 'nickname' in data:
                nickname = str(data.get('nickname') or '').strip()
                if not nickname or len(nickname) > 20:
                    return jsonify({'status': 'error', 'message': '昵称需为 1-20 个字符'}), 400
                if any(ch in nickname for ch in ('\n', '\r', '\t')):
                    return jsonify({'status': 'error', 'message': '昵称不能包含换行或制表符'}), 400

            avatar = None
            if 'avatar' in data:
                avatar = str(data.get('avatar') or '').strip()
                if avatar not in PROFILE_AVATARS:
                    return jsonify({'status': 'error', 'message': '头像类型不支持'}), 400

            store = ProfilesStore()
            profile = store.get_profile(user_id)

            # 更新画像字段
            if nickname is not None: profile['nickname'] = nickname
            if avatar is not None: profile['avatar'] = avatar
            if 'major' in data: profile['major'] = data['major']
            if 'grade' in data: profile['grade'] = data['grade']
            if 'goal' in data: profile['goal'] = data['goal']
            if 'known_skills' in data: profile['known_skills'] = data['known_skills']
            if 'weak_areas' in data: profile['weak_areas'] = data['weak_areas']
            if 'daily_hours' in data: profile['daily_hours'] = data['daily_hours']
            if 'profile_completed' in data: profile['profile_completed'] = data['profile_completed']

            store.save_profile(user_id, profile)
            return jsonify({'status': 'success', 'message': '画像更新成功', 'data': profile})
    except Exception as e:
        logger.error(f"处理用户画像请求失败: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/auth/register', methods=['POST'])
def register():
    """
    用户注册
    POST: 注册新用户
    """
    try:
        data = request.get_json(silent=True) or {}
        account = data.get('account')
        password = data.get('password')
        
        if not account or not password:
            return jsonify({'status': 'error', 'message': '请提供账号和密码'}), 400
            
        success, message = register_user(account, password)
        if success:
            return jsonify({'status': 'success', 'message': message})
        else:
            return jsonify({'status': 'error', 'message': message}), 400
    except Exception as e:
        logger.error(f"注册失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'注册失败: {str(e)}'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """
    用户登录
    POST: 用户登录
    """
    try:
        data = request.get_json(silent=True) or {}
        account = data.get('account')
        password = data.get('password')
        
        if not account or not password:
            return jsonify({'status': 'error', 'message': '请提供账号和密码'}), 400
            
        success, result = login_user(account, password)
        if success:
            # result is a dict with user_id and role
            return jsonify({
                'status': 'success', 
                'user_id': result['user_id'],
                'role': result['role']
            })
        else:
            return jsonify({'status': 'error', 'message': result}), 401
    except Exception as e:
        logger.error(f"登录失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'登录失败: {str(e)}'}), 500

@app.route('/api/admin/stats', methods=['GET'])
def admin_stats():
    """
    管理员统计信息
    GET: 获取系统统计信息
    """
    try:
        # 鉴权
        # 在实际应用中，这里应该验证session或token中的role字段
        # 为了演示，我们暂时信任客户端（注意：这不安全）
        # 理想情况下应该使用 @login_required 装饰器并检查 current_user.role
        
        # 获取用户统计
        import sqlite3
        conn = sqlite3.connect("data/users.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        conn.close()
        
        # 获取会话统计
        storage = DataStorage()
        conv_stats = storage.get_conversation_stats()
        
        return jsonify({
            'status': 'success',
            'stats': {
                'user_count': user_count,
                'conversation_count': conv_stats.get('conversation_count', 0),
                'message_count': conv_stats.get('message_count', 0),
                'db_path': os.path.abspath("data/users.db")
            }
        })
    except Exception as e:
        logger.error(f"获取管理员统计失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'获取统计失败: {str(e)}'}), 500

@app.route('/api/admin/feedback', methods=['GET'])
def admin_feedback_list():
    try:
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        user_id = request.args.get('user_id')
        rating_raw = request.args.get('rating')
        rating = int(rating_raw) if rating_raw is not None and str(rating_raw).strip() != "" else None
        storage = DataStorage()
        items = storage.list_feedback(limit=limit, offset=offset, user_id=user_id, rating=rating)
        return jsonify({'status': 'success', 'data': items})
    except Exception as e:
        logger.error(f"获取反馈列表失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'获取反馈列表失败: {str(e)}'}), 500

@app.route('/api/admin/feedback/stats', methods=['GET'])
def admin_feedback_stats():
    try:
        storage = DataStorage()
        stats = storage.get_feedback_stats()
        return jsonify({'status': 'success', 'stats': stats})
    except Exception as e:
        logger.error(f"获取反馈统计失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'获取反馈统计失败: {str(e)}'}), 500

@app.route('/api/admin/rag/stats', methods=['GET'])
def admin_rag_stats():
    try:
        from rag.service import get_rag_service
        rag = get_rag_service()
        stats = rag.get_knowledge_base_stats()
        return jsonify({'status': 'success', 'stats': stats})
    except Exception as e:
        logger.error(f"获取知识库统计失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'获取知识库统计失败: {str(e)}'}), 500

@app.route('/api/admin/rag/rebuild', methods=['POST'])
def admin_rag_rebuild():
    try:
        from rag.service import get_rag_service
        rag = get_rag_service()
        ok = rag.rebuild()
        return jsonify({'status': 'success' if ok else 'error', 'rebuild': ok})
    except Exception as e:
        logger.error(f"重建知识库失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'重建知识库失败: {str(e)}'}), 500

@app.route('/api/admin/rag/ingest/text', methods=['POST'])
def admin_rag_ingest_text():
    try:
        data = request.get_json(silent=True) or {}
        text = (data.get('text') or '').strip()
        subject = (data.get('subject') or '通用').strip() or '通用'
        title = (data.get('title') or '').strip()
        if not text or len(text) < 10:
            return jsonify({'status': 'error', 'message': 'text 不能为空且长度需>=10'}), 400

        import uuid
        ingested_dir = os.path.join('data', 'ingested')
        os.makedirs(ingested_dir, exist_ok=True)
        file_id = str(uuid.uuid4())
        file_name = f"ingest_{file_id}.json"
        file_path = os.path.join(ingested_dir, file_name)
        payload = {
            "text": text,
            "meta": {
                "subject": subject,
                "title": title,
                "source": "manual_text",
                "ingested_at": datetime.now().isoformat()
            }
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return jsonify({'status': 'success', 'data': {'file_path': os.path.abspath(file_path), 'file_name': file_name}})
    except Exception as e:
        logger.error(f"导入文本失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'导入文本失败: {str(e)}'}), 500

@app.route('/api/admin/rag/ingest/url', methods=['POST'])
def admin_rag_ingest_url():
    try:
        data = request.get_json(silent=True) or {}
        url = (data.get('url') or '').strip()
        subject = (data.get('subject') or '通用').strip() or '通用'
        if not url:
            return jsonify({'status': 'error', 'message': '请提供 url'}), 400

        text, meta = fetch_url(url)
        if not text or len(text) < 50:
            return jsonify({'status': 'error', 'message': '抓取内容为空或过短'}), 400

        import uuid
        ingested_dir = os.path.join('data', 'ingested')
        os.makedirs(ingested_dir, exist_ok=True)
        file_id = str(uuid.uuid4())
        file_name = f"url_{file_id}.json"
        file_path = os.path.join(ingested_dir, file_name)
        payload = {
            "text": text,
            "meta": {
                "subject": subject,
                "source": "url",
                "url": url,
                "title": meta.get("title") if isinstance(meta, dict) else "",
                "ingested_at": datetime.now().isoformat()
            }
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return jsonify({'status': 'success', 'data': {'file_path': os.path.abspath(file_path), 'file_name': file_name}})
    except Exception as e:
        logger.error(f"导入URL失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'导入URL失败: {str(e)}'}), 500

@app.route('/api/admin/rag/eval/run', methods=['POST'])
def admin_rag_eval_run():
    try:
        data = request.get_json(silent=True) or {}
        dataset_path = (data.get('dataset_path') or os.path.join('data', 'rag_testcases.jsonl')).strip()
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.abspath(dataset_path)
        limit = int(data.get('limit', 50))
        subject = (data.get('subject') or '').strip()

        from rag.eval_lab import create_eval_job, start_eval_job
        job = create_eval_job(dataset_path=dataset_path, limit=limit, subject=subject)
        start_eval_job(job["id"])
        return jsonify({'status': 'success', 'job': {"id": job["id"]}})
    except Exception as e:
        logger.error(f"启动评估失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'启动评估失败: {str(e)}'}), 500

@app.route('/api/admin/rag/eval/jobs', methods=['GET'])
def admin_rag_eval_jobs():
    try:
        from rag.eval_lab import list_eval_jobs
        jobs = list_eval_jobs()
        return jsonify({'status': 'success', 'data': jobs})
    except Exception as e:
        logger.error(f"获取评估任务列表失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'获取评估任务列表失败: {str(e)}'}), 500

@app.route('/api/admin/rag/eval/jobs/<job_id>', methods=['GET'])
def admin_rag_eval_job_detail(job_id):
    try:
        from rag.eval_lab import get_eval_job
        job = get_eval_job(job_id)
        if not job:
            return jsonify({'status': 'error', 'message': '任务不存在'}), 404
        return jsonify({'status': 'success', 'data': job})
    except Exception as e:
        logger.error(f"获取评估任务详情失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'获取评估任务详情失败: {str(e)}'}), 500

@app.route('/api/admin/users', methods=['GET'])
def admin_users():
    """
    管理员：获取用户列表
    """
    try:
        # 简单鉴权
        import sqlite3
        conn = sqlite3.connect("data/users.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, account, role, created_at FROM users ORDER BY created_at DESC")
        users = []
        for row in cursor.fetchall():
            users.append({
                'id': row[0],
                'account': row[1],
                'role': row[2],
                'created_at': row[3]
            })
        
        conn.close()
        return jsonify({'status': 'success', 'users': users})
    except Exception as e:
        logger.error(f"获取用户列表失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'获取用户列表失败: {str(e)}'}), 500

@app.route('/api/admin/users/<user_id>', methods=['DELETE'])
def admin_delete_user(user_id):
    """
    管理员：删除用户
    """
    try:
        success, message = delete_user(user_id)
        if success:
            return jsonify({'status': 'success', 'message': message})
        else:
            return jsonify({'status': 'error', 'message': message}), 400
    except Exception as e:
        logger.error(f"删除用户失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'删除用户失败: {str(e)}'}), 500

@app.route('/api/admin/users/<user_id>/reset_password', methods=['POST'])
def admin_reset_password(user_id):
    """
    管理员：重置用户密码
    """
    try:
        data = request.get_json(silent=True) or {}
        new_password = data.get('password')
        if not new_password:
             return jsonify({'status': 'error', 'message': '请提供新密码'}), 400
             
        success, message = reset_password(user_id, new_password)
        if success:
            return jsonify({'status': 'success', 'message': message})
        else:
            return jsonify({'status': 'error', 'message': message}), 400
    except Exception as e:
        logger.error(f"重置密码失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'重置密码失败: {str(e)}'}), 500

@app.route('/api/admin/users/<user_id>/stats', methods=['GET'])
def admin_user_stats(user_id):
    """
    管理员：获取单个用户详细统计
    """
    try:
        storage = DataStorage()
        
        # 1. 获取会话统计
        conversations = storage.list_conversations(user_id, limit=1000)
        conv_count = len(conversations)
        msg_count = sum(c.get('message_count', 0) for c in conversations)
        
        # 2. 获取学习计划
        plans = storage.get_all_plans(user_id)
        plan_count = len(plans.get('plans', [])) if isinstance(plans, dict) else 0
        
        # 3. 获取学习进度
        progress = storage.get_all_progress(user_id)
        # progress is dict {subject: {...}}
        progress_count = len(progress) if isinstance(progress, dict) else 0
        
        return jsonify({
            'status': 'success',
            'stats': {
                'conversation_count': conv_count,
                'message_count': msg_count,
                'plan_count': plan_count,
                'progress_count': progress_count
            }
        })
    except Exception as e:
        logger.error(f"获取用户详细统计失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'获取用户详细统计失败: {str(e)}'}), 500



@app.route('/api/admin/system/backup', methods=['POST'])
def admin_backup_db():
    """管理员：备份数据库"""
    import shutil
    import time
    try:
        source_db = "data/users.db"
        if not os.path.exists(source_db):
            return jsonify({'status': 'error', 'message': '数据库文件不存在'}), 404
            
        backup_dir = "data/backups"
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"users_backup_{timestamp}.db")
        
        shutil.copy2(source_db, backup_path)
        
        return jsonify({
            'status': 'success', 
            'message': f'数据库已备份至 {backup_path}',
            'path': backup_path
        })
    except Exception as e:
        logger.error(f"备份数据库失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'备份失败: {str(e)}'}), 500

@app.route('/api/admin/system/logs', methods=['GET'])
def admin_export_logs():
    """管理员：获取最新日志"""
    try:
        log_file = "app.log" # 假设日志文件在根目录
        if not os.path.exists(log_file):
            # 尝试在logs目录查找
            log_file = "logs/app.log"
            if not os.path.exists(log_file):
                 return jsonify({'status': 'error', 'message': '日志文件不存在'}), 404
        
        # 读取最后100行
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            last_lines = lines[-100:]
            
        return jsonify({
            'status': 'success',
            'logs': ''.join(last_lines)
        })
    except Exception as e:
        logger.error(f"导出日志失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'导出日志失败: {str(e)}'}), 500

# --- 用户画像与错题本 API ---

@app.route('/api/profile', methods=['GET', 'PUT'])
def profile_api():
    """用户画像 API"""
    user_id = request.args.get('user_id', 'default')
    
    if request.method == 'GET':
        profile = coordinator.get_user_profile(user_id)
        return jsonify({'status': 'success', 'profile': profile})
        
    elif request.method == 'PUT':
        data = request.get_json(silent=True) or {}
        coordinator.update_user_profile(user_id, data)
        return jsonify({'status': 'success', 'message': '画像已更新'})

@app.route('/api/checkin', methods=['POST'])
def checkin_api():
    """每日签到"""
    data = request.get_json(silent=True) or {}
    user_id = data.get('user_id', 'default')
    result = coordinator.check_in(user_id)
    return jsonify({'status': 'success', 'data': result})

@app.route('/api/recommendations', methods=['GET'])
def recommendations_api():
    try:
        user_id = request.args.get('user_id', 'default')
        storage = DataStorage()

        plans = coordinator.get_plans(user_id) or []
        plan_subjects = [p.get("subject") for p in plans if isinstance(p, dict) and p.get("subject")]

        progress_all = storage.get_all_progress(user_id) or {}
        progress_subjects = list(progress_all.keys()) if isinstance(progress_all, dict) else []

        quiz_history = storage.get_quiz_history(user_id, limit=200) or []
        quiz_subjects = [q.get("subject") for q in quiz_history if isinstance(q, dict) and q.get("subject")]

        due_cards = storage.get_due_flashcards(user_id) or []
        due_by_subject = {}
        for c in due_cards:
            if not isinstance(c, dict):
                continue
            subj = c.get("subject") or "通用"
            due_by_subject[subj] = due_by_subject.get(subj, 0) + 1

        subjects = []
        for s in plan_subjects + progress_subjects + quiz_subjects:
            if not s:
                continue
            if s not in subjects:
                subjects.append(s)
        if not subjects:
            subjects = ["通用"]

        scored = []
        for subj in subjects:
            prog = progress_all.get(subj) if isinstance(progress_all, dict) else None
            progress_pct = int((prog or {}).get("progress_percentage") or 0)
            perf = storage.get_quiz_performance(user_id, subj) or {}
            quiz_avg = float(perf.get("average_score") or 0.0)
            due_cnt = int(due_by_subject.get(subj, 0))
            score = (100 - progress_pct) * 0.4 + (100 - quiz_avg) * 0.6 + min(due_cnt, 10) * 2
            scored.append({
                "subject": subj,
                "score": score,
                "progress_percentage": progress_pct,
                "quiz_average": quiz_avg,
                "due_flashcards": due_cnt,
                "has_plan": subj in plan_subjects
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:3]

        actions = []
        if due_cards:
            actions.append({
                "type": "flashcards",
                "title": f"复习闪卡（{len(due_cards)}张待复习）",
                "subject": "",
                "priority": "high"
            })
        for item in top:
            subj = item["subject"]
            if item["due_flashcards"] > 0:
                actions.append({
                    "type": "flashcards",
                    "title": f"复习 {subj} 闪卡（{item['due_flashcards']}张）",
                    "subject": subj,
                    "priority": "high"
                })
            actions.append({
                "type": "quiz",
                "title": f"做一次 {subj} 测验",
                "subject": subj,
                "priority": "medium"
            })
            if item.get("has_plan") and item.get("progress_percentage", 0) < 100:
                actions.append({
                    "type": "plan",
                    "title": f"完成 {subj} 计划任务（同步进度）",
                    "subject": subj,
                    "priority": "medium"
                })

        actions = actions[:6]
        return jsonify({
            "status": "success",
            "data": {
                "top_subjects": top,
                "actions": actions
            }
        })
    except Exception as e:
        logger.error(f"获取推荐失败: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'获取推荐失败: {str(e)}'}), 500

# 移除错题本相关API

# --- 闪卡 API ---

@app.route('/api/flashcards', methods=['GET', 'POST'])
def flashcards_api():
    """闪卡管理"""
    user_id = request.args.get('user_id', 'default')
    
    if request.method == 'GET':
        mode = (request.args.get('mode') or 'due').strip().lower()
        if mode == 'all':
            cards = coordinator.get_all_flashcards(user_id)
        else:
            # 获取待复习的闪卡
            cards = coordinator.get_due_flashcards(user_id)
        return jsonify({'status': 'success', 'cards': cards})
        
    elif request.method == 'POST':
        # 创建闪卡
        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id', user_id)
        card = coordinator.create_flashcard(
            user_id,
            data.get('subject', '通用'),
            data.get('front'),
            data.get('back')
        )
        return jsonify({'status': 'success', 'card': card})

@app.route('/api/flashcards/<card_id>/review', methods=['POST'])
def review_flashcard_api(card_id):
    """复习闪卡"""
    data = request.get_json(silent=True) or {}
    user_id = data.get('user_id', 'default')
    success = data.get('success', False)
    
    card = coordinator.review_flashcard(user_id, card_id, success)
    if card:
        return jsonify({'status': 'success', 'card': card})
    return jsonify({'status': 'error', 'message': '操作失败'}), 404

@app.route('/api/flashcards/<card_id>', methods=['DELETE'])
def delete_flashcard_api(card_id):
    """删除闪卡"""
    user_id = request.args.get('user_id') or (request.get_json() or {}).get('user_id', 'default')
    success = coordinator.delete_flashcard(user_id, card_id)
    if success:
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': '删除失败'}), 404

# --- 复习计划管理 API ---

@app.route('/api/review/schedule', methods=['POST'])
def schedule_review_api():
    """安排新的复习任务"""
    try:
        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id', 'default')
        subject = data.get('subject', '通用')
        content = data.get('content', '')
        importance = data.get('importance', 'medium')
        
        # 调用 ReviewAgent 安排复习
        review_agent = coordinator.get_agent('review')
        if not review_agent:
            return jsonify({'status': 'error', 'message': '复习系统未就绪'}), 503
            
        result = review_agent.schedule_review(
            user_id, subject, content, importance, data.get('first_review_at')
        )
        if result.get('status') == 'success':
            return jsonify(result)
        return jsonify(result), 400
    except Exception as e:
        logger.error(f"安排复习失败: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/review/next', methods=['GET'])
def get_next_review_api():
    """获取下一个复习任务"""
    try:
        user_id = request.args.get('user_id', 'default')
        review_agent = coordinator.get_agent('review')
        if not review_agent:
            return jsonify({'status': 'error', 'message': '复习系统未就绪'}), 503
            
        result = review_agent.get_next_review(user_id)
        if result.get('status') == 'success':
            return jsonify(result)
        return jsonify(result), 400
    except Exception as e:
        logger.error(f"获取复习任务失败: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/review/history', methods=['GET'])
def get_review_history_api():
    """获取复习历史"""
    try:
        user_id = request.args.get('user_id', 'default')
        limit = int(request.args.get('limit', 50))
        
        review_agent = coordinator.get_agent('review')
        if not review_agent:
            return jsonify({'status': 'error', 'message': '复习系统未就绪'}), 503
            
        result = review_agent.get_review_history(user_id, limit)
        if result.get('status') == 'success':
            return jsonify(result)
        return jsonify(result), 400
    except Exception as e:
        logger.error(f"获取复习历史失败: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/review/complete', methods=['POST'])
def complete_review_api():
    """完成复习任务"""
    try:
        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id', 'default')
        review_id = data.get('review_id')
        performance = data.get('performance', {})
        
        review_agent = coordinator.get_agent('review')
        if not review_agent:
            return jsonify({'status': 'error', 'message': '复习系统未就绪'}), 503
            
        result = review_agent.complete_review(user_id, review_id, performance)
        if result.get('status') == 'success':
            return jsonify(result)
        return jsonify(result), 400
    except Exception as e:
        logger.error(f"提交复习结果失败: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/review/get', methods=['GET'])
def get_review_api():
    """获取复习计划详情"""
    try:
        user_id = request.args.get('user_id', 'default')
        review_id = request.args.get('review_id')
        
        if not review_id:
            return jsonify({'status': 'error', 'message': '缺少review_id'}), 400
            
        review_agent = coordinator.get_agent('review')
        if not review_agent:
            return jsonify({'status': 'error', 'message': '复习系统未就绪'}), 503
            
        result = review_agent.get_review(user_id, review_id)
        return jsonify(result)
    except Exception as e:
        logger.error(f"获取复习详情失败: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/review/delete', methods=['POST'])
def delete_review_api():
    """删除复习计划"""
    try:
        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id', 'default')
        review_id = data.get('review_id')
        
        review_agent = coordinator.get_agent('review')
        if not review_agent:
            return jsonify({'status': 'error', 'message': '复习系统未就绪'}), 503
            
        result = review_agent.delete_review(user_id, review_id)
        if result.get('status') == 'success':
            return jsonify(result)
        return jsonify(result), 404
    except Exception as e:
        logger.error(f"删除复习计划失败: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# --- 高级功能 API ---

@app.route('/api/mindmap', methods=['POST'])
def generate_mindmap_api():
    """生成思维导图"""
    data = request.get_json(silent=True) or {}
    topic = data.get('topic')
    
    try:
        from rag.rag_chain import get_rag_helper
        rag_helper = get_rag_helper()
        mermaid_code = rag_helper.generate_mindmap(topic)
    except Exception as e:
        logger.error(f"调用RAG生成思维导图失败: {e}")
        mermaid_code = f"""mindmap
  root(({topic}))
    基础概念
      定义
      历史
      应用场景
    核心技术
      算法
      数据结构
    最佳实践
      设计模式
      性能优化
    资源
      官方文档
      社区教程
"""

    return jsonify({'status': 'success', 'mermaid_code': mermaid_code})

# ---------------- 今日一个概念 ----------------
@app.route('/api/daily-concept', methods=['GET'])
def daily_concept_api():
    """从本地知识库随机抽一个「概念」，供首屏展示（不调模型）。"""
    import random
    import re as _re

    try:
        doc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "documents")
        _skip = ("求职", "面试", "简历", "作品集")
        files = [f for f in os.listdir(doc_dir)
                 if f.endswith(".md") and not any(k in f for k in _skip)] if os.path.isdir(doc_dir) else []
        if not files:
            return jsonify({'status': 'error', 'message': '知识库暂无文档'}), 404

        random.shuffle(files)
        for name in files[:20]:
            path = os.path.join(doc_dir, name)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            # 取所有二级标题及其后的第一段
            blocks = _re.split(r"\n(?=##\s)", text)
            sections = []
            for block in blocks:
                lines = [ln.strip() for ln in block.splitlines()]
                if not lines or not lines[0].startswith("## "):
                    continue
                heading = lines[0][3:].strip()
                heading = _re.sub(r"^\d+[.、)]\s*", "", heading)
                body = []
                for ln in lines[1:]:
                    if not ln:
                        if body:
                            break
                        continue
                    if ln.startswith("#"):
                        break
                    body.append(ln)
                _bad = ("常见坑", "误区", "总结", "小结", "习题", "练习", "参考", "延伸", "本章", "要点回顾")
                if heading and any(b in heading for b in _bad):
                    continue
                if heading and body:
                    cleaned = []
                    for ln in body:
                        ln = _re.sub(r"^[-*+]\s+", "", ln)
                        ln = _re.sub(r"^\d+[.、)]\s*", "", ln)
                        ln = _re.sub(r"[*`>]", "", ln).strip()
                        if ln:
                            cleaned.append(ln)
                    para = " · ".join(cleaned)
                    sections.append((heading, para))

            if not sections:
                continue
            heading, para = random.choice(sections)
            snippet = para[:110] + ("…" if len(para) > 110 else "")
            return jsonify({
                "status": "success",
                "data": {
                    "doc": name.replace(".md", ""),
                    "title": heading,
                    "snippet": snippet,
                },
            })

        return jsonify({'status': 'error', 'message': '未抽到合适的概念'}), 404
    except Exception as exc:
        logger.error(f"抽取今日概念失败: {exc}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(exc)}), 500



@app.route('/api/activity-week', methods=['GET'])
def activity_week_api():
    """近 7 天活动量（来自会话消息与笔记创建时间），供首屏柱状图。"""
    import json as _json
    from datetime import datetime as _dt, timedelta as _td

    user_id = request.args.get('user_id', 'default')
    base = os.path.dirname(os.path.abspath(__file__))
    today = _dt.now().date()
    days = [(today - _td(days=i)) for i in range(6, -1, -1)]
    counts = {d.isoformat(): 0 for d in days}

    def _bump(ts):
        try:
            d = _dt.fromisoformat(str(ts)).date().isoformat()
        except Exception:
            return
        if d in counts:
            counts[d] += 1

    for name in ('conversations.json',):
        p = os.path.join(base, 'data', name)
        if not os.path.exists(p):
            continue
        try:
            with open(p, 'r', encoding='utf-8') as f:
                raw = _json.load(f)
            convs = raw if isinstance(raw, list) else raw.get('conversations', [])
            if isinstance(convs, dict):
                convs = list(convs.values())
            for c in convs:
                if str(c.get('user_id')) != str(user_id):
                    continue
                for m in (c.get('messages') or []):
                    _bump(m.get('timestamp'))
        except Exception:
            continue

    p = os.path.join(base, 'data', 'notes.json')
    if os.path.exists(p):
        try:
            with open(p, 'r', encoding='utf-8') as f:
                raw = _json.load(f)
            for n in (raw.get('notes') or {}).values():
                if str(n.get('user_id')) == str(user_id):
                    _bump(n.get('created_at'))
        except Exception:
            pass

    week = ['一', '二', '三', '四', '五', '六', '日']
    out = [{
        'date': d.isoformat(),
        'label': week[d.weekday()],
        'count': counts[d.isoformat()],
        'is_today': d == today,
    } for d in days]
    return jsonify({'status': 'success', 'data': {'days': out, 'total': sum(counts.values())}})



@app.route('/api/debug/retrieve', methods=['GET'])
def debug_retrieve():
    """诊断用：只跑检索与过滤，不调模型。返回意图、命中的片段标题与开头。"""
    q = request.args.get('q', '').strip()
    subject = request.args.get('subject', '通用').strip() or '通用'
    if not q:
        return jsonify({'status': 'error', 'message': '缺少 q 参数'}), 400
    try:
        # RAGHelper 是包装层，真正的检索/意图方法在 RAGService 上
        from rag.rag_chain import get_rag_helper
        helper = coordinator.get_rag_helper() if hasattr(coordinator, 'get_rag_helper') else get_rag_helper()
        rag = getattr(helper, "_service", None) or helper
        rag.ensure_ready()
        intent = rag._question_intent(q)
        retrieve_k = 10 if intent == 'define' else None
        retrieved, focus = rag._retrieve(q, subject, "", k=retrieve_k)
        def _brief(doc):
            meta = getattr(doc, 'metadata', {}) or {}
            return {
                'chunk_title': meta.get('chunk_title') or meta.get('title'),
                'file_name': meta.get('file_name'),
                'head': (getattr(doc, 'page_content', '') or '')[:70].replace('\n', ' '),
            }

        # 过滤前：全部召回 + 是否通过意图过滤
        retrieved_view = []
        for d in retrieved:
            passed = rag._doc_answers_intent(q, d, subject)
            item = _brief(d)
            item['passed'] = passed
            retrieved_view.append(item)

        kept = [d for d in retrieved if rag._doc_answers_intent(q, d, subject)]
        if intent == 'define' and len(kept) > 1:
            kept = rag._prefer_topic_chunks(kept, q, subject)
        items = [_brief(d) for d in kept]
        return jsonify({'status': 'success', 'data': {
            'query': q, 'subject': subject, 'intent': intent,
            'retrieved': len(retrieved), 'kept': len(kept),
            'focus_terms': focus, 'items': items, 'retrieved_items': retrieved_view,
        }})
    except Exception as exc:
        logger.error(f"诊断检索失败: {exc}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(exc)}), 500


if __name__ == '__main__':
    # 在后台线程中预初始化 RAG 知识库
    def init_rag_background():
        try:
            from rag.rag_chain import get_rag_helper
            logger.info("后台初始化 RAG 知识库...")
            rag = get_rag_helper()
            rag.ensure_ready()
            logger.info("后台 RAG 知识库初始化完成")
        except Exception as e:
            logger.error(f"后台初始化 RAG 失败: {e}")

    import threading
    init_thread = threading.Thread(target=init_rag_background, daemon=True)
    init_thread.start()

    # 启动Flask应用
    host = config.get('flask', {}).get('host', '127.0.0.1')
    port = config.get('flask', {}).get('port', 5000)
    debug = True  # 临时开启调试模式以获取详细错误信息
    
    logger.info(f"启动多Agent智能学习助理系统，监听地址: {host}:{port}")
    app.run(host=host, port=port, debug=debug)
