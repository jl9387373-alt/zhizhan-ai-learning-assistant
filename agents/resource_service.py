"""
资源推荐Agent
负责推荐学习资源
"""

from agents.base_agent import BaseAgent
from core.message import Message, MessageType
from core.config import config
from core.logger import logger
from typing import Dict, Any, List, Optional
import requests
from tools.web_search import web_search
from storage.profiles import ProfilesStore


class ResourceService(BaseAgent):
    """资源推荐Agent"""
    
    def __init__(self):
        super().__init__("resource", "资源推荐Agent")
        self.resources = self._load_resources()
        self.module_config = config.get("modules", {}).get("resource", {})
        self.profile_store = ProfilesStore()
        
        # 注册 Web Search 工具
        self.register_tool(
            name="web_search",
            func=web_search,
            description="在互联网上搜索最新的学习资源、文档或教程",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "最大结果数量",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        )
    
    def _load_resources(self) -> Dict[str, List[Dict[str, Any]]]:
        """加载资源库"""
        return {
            "python": [
                {"name": "Python官方文档", "type": "文档", "url": "https://docs.python.org/zh-cn/3/", "difficulty": "初级", "description": "Python官方中文文档，最权威的参考资料"},
                {"name": "Python编程：从入门到实践", "type": "书籍", "url": "", "difficulty": "初级", "description": "Eric Matthes著，适合零基础学习"},
                {"name": "廖雪峰Python教程", "type": "教程", "url": "https://www.liaoxuefeng.com/wiki/1016959663602400", "difficulty": "初级", "description": "中文Python教程，内容全面"},
                {"name": "Real Python", "type": "网站", "url": "https://realpython.com/", "difficulty": "中级", "description": "英文Python教程网站，质量很高"},
                {"name": "Python Cookbook", "type": "书籍", "url": "", "difficulty": "中级", "description": "Python进阶必读书籍"},
                {"name": "流畅的Python", "type": "书籍", "url": "", "difficulty": "高级", "description": "深入理解Python特性"},
                {"name": "Python Tricks", "type": "书籍", "url": "", "difficulty": "中级", "description": "Python技巧和最佳实践"},
                {"name": "Python Crash Course", "type": "书籍", "url": "", "difficulty": "初级", "description": "快速入门Python"},
                {"name": "LeetCode Python题解", "type": "练习", "url": "https://leetcode.cn/", "difficulty": "中级", "description": "算法练习平台"},
                {"name": "Python Weekly", "type": "资讯", "url": "https://www.pythonweekly.com/", "difficulty": "通用", "description": "Python周报，了解最新动态"},
                {"name": "PyPI", "type": "平台", "url": "https://pypi.org/", "difficulty": "通用", "description": "Python包索引，查找第三方库"},
                {"name": "Awesome Python", "type": "资源", "url": "https://github.com/vinta/awesome-python", "difficulty": "通用", "description": "Python资源大全"},
            ],
            "机器学习": [
                {"name": "机器学习（周志华）", "type": "书籍", "url": "", "difficulty": "中级", "description": "机器学习经典教材，中文版"},
                {"name": "Coursera机器学习课程", "type": "课程", "url": "https://www.coursera.org/learn/machine-learning", "difficulty": "中级", "description": "Andrew Ng的经典课程"},
                {"name": "Scikit-learn文档", "type": "文档", "url": "https://scikit-learn.org/stable/", "difficulty": "中级", "description": "Python机器学习库官方文档"},
                {"name": "Kaggle Learn", "type": "课程", "url": "https://www.kaggle.com/learn", "difficulty": "初级", "description": "免费机器学习课程和实践"},
                {"name": "Fast.ai", "type": "课程", "url": "https://www.fast.ai/", "difficulty": "中级", "description": "实用深度学习课程"},
                {"name": "TensorFlow官方教程", "type": "教程", "url": "https://www.tensorflow.org/tutorials", "difficulty": "中级", "description": "TensorFlow深度学习框架教程"},
                {"name": "PyTorch官方教程", "type": "教程", "url": "https://pytorch.org/tutorials/", "difficulty": "中级", "description": "PyTorch深度学习框架教程"},
                {"name": "Hands-On Machine Learning", "type": "书籍", "url": "", "difficulty": "中级", "description": "实用机器学习指南"},
                {"name": "Deep Learning (Ian Goodfellow)", "type": "书籍", "url": "", "difficulty": "高级", "description": "深度学习领域经典教材"},
                {"name": "Papers With Code", "type": "资源", "url": "https://paperswithcode.com/", "difficulty": "高级", "description": "论文和代码集合"},
                {"name": "Kaggle竞赛", "type": "实践", "url": "https://www.kaggle.com/competitions", "difficulty": "中级", "description": "机器学习竞赛平台"},
            ],
            "java": [
                {"name": "Java官方文档", "type": "文档", "url": "https://docs.oracle.com/javase/", "difficulty": "通用", "description": "Java官方API文档"},
                {"name": "Java核心技术", "type": "书籍", "url": "", "difficulty": "中级", "description": "Java经典教材"},
                {"name": "Effective Java", "type": "书籍", "url": "", "difficulty": "高级", "description": "Java最佳实践"},
                {"name": "Spring官方文档", "type": "文档", "url": "https://spring.io/docs", "difficulty": "中级", "description": "Spring框架文档"},
                {"name": "Java并发编程实战", "type": "书籍", "url": "", "difficulty": "高级", "description": "Java并发编程经典"},
            ],
            "产品经理": [
                {"name": "人人都是产品经理", "type": "社区", "url": "https://www.woshipm.com/", "difficulty": "初级", "description": "中文产品社区，案例多，适合建立产品感"},
                {"name": "SVPG（Marty Cagan）", "type": "博客", "url": "https://www.svpg.com/articles/", "difficulty": "高级", "description": "《启示录》作者博客，讲产品发现与团队"},
                {"name": "Lenny's Newsletter", "type": "资讯", "url": "https://www.lennysnewsletter.com/", "difficulty": "中级", "description": "增长与产品实践，案例偏实操"},
                {"name": "Mind the Product", "type": "社区", "url": "https://www.mindtheproduct.com/", "difficulty": "中级", "description": "全球产品社区，方法论与访谈"},
                {"name": "《启示录》", "type": "书籍", "url": "", "difficulty": "初级", "description": "产品经理入门经典，讲清角色与流程"},
                {"name": "《精益创业》", "type": "书籍", "url": "", "difficulty": "中级", "description": "最小可行产品与验证式学习"},
                {"name": "《用户故事与敏捷方法》", "type": "书籍", "url": "", "difficulty": "中级", "description": "把需求写成可执行条目"},
                {"name": "《金字塔原理》", "type": "书籍", "url": "", "difficulty": "初级", "description": "结构化表达，写 PRD 和汇报都用得上"},
                {"name": "Figma 社区", "type": "平台", "url": "https://www.figma.com/community", "difficulty": "初级", "description": "看别人的原型稿，学信息结构"},
                {"name": "产品沉思录", "type": "资讯", "url": "https://pmthinking.com/", "difficulty": "中级", "description": "中文产品思考合集"},
            ],
            "职场": [
                {"name": "《金字塔原理》", "type": "书籍", "url": "", "difficulty": "初级", "description": "结论先行的表达框架"},
                {"name": "《沟通的方法》", "type": "书籍", "url": "", "difficulty": "初级", "description": "脱不花著，中文职场沟通实操"},
                {"name": "《非暴力沟通》", "type": "书籍", "url": "", "difficulty": "初级", "description": "把情绪换成需求的表达方式"},
                {"name": "《高效能人士的七个习惯》", "type": "书籍", "url": "", "difficulty": "初级", "description": "时间管理与自我管理的基础"},
                {"name": "牛客网", "type": "平台", "url": "https://www.nowcoder.com/", "difficulty": "初级", "description": "校招笔面经与岗位信息"},
                {"name": "Harvard Business Review", "type": "资讯", "url": "https://hbr.org/", "difficulty": "高级", "description": "管理与协作类文章，英文"},
                {"name": "LinkedIn Learning", "type": "课程", "url": "https://www.linkedin.com/learning/", "difficulty": "中级", "description": "职场技能短期课程"},
                {"name": "《麦肯锡工作法》", "type": "书籍", "url": "", "difficulty": "中级", "description": "结构化解决问题的方法"},
            ],
            "通用": [
                {"name": "GitHub", "type": "平台", "url": "https://github.com/", "difficulty": "通用", "description": "代码托管和开源项目平台"},
                {"name": "Stack Overflow", "type": "社区", "url": "https://stackoverflow.com/", "difficulty": "通用", "description": "程序员问答社区"},
                {"name": "Coursera", "type": "平台", "url": "https://www.coursera.org/", "difficulty": "通用", "description": "在线课程平台"},
                {"name": "edX", "type": "平台", "url": "https://www.edx.org/", "difficulty": "通用", "description": "在线课程平台"},
                {"name": "Udemy", "type": "平台", "url": "https://www.udemy.com/", "difficulty": "通用", "description": "在线课程平台"},
                {"name": "MDN Web Docs", "type": "文档", "url": "https://developer.mozilla.org/", "difficulty": "通用", "description": "Web开发文档"},
                {"name": "FreeCodeCamp", "type": "课程", "url": "https://www.freecodecamp.org/", "difficulty": "初级", "description": "免费编程课程"},
                {"name": "Codecademy", "type": "课程", "url": "https://www.codecademy.com/", "difficulty": "初级", "description": "交互式编程学习"},
            ]
        }
    
    def process(self, message: Message) -> Message:
        """处理资源推荐请求"""
        content = message.content
        action = content.get("action", "recommend")
        subject = content.get("subject", "通用")
        difficulty = content.get("difficulty", "all")
        resource_type = content.get("type", "all")
        count = content.get("count", 5)
        
        if action == "get_all":
            # 如果是获取所有资源，返回所有资源，不限制数量
            # 合并所有主题的资源
            all_resources = []
            for topic, resources in self.resources.items():
                if not subject or subject == topic:
                    all_resources.extend(resources)
            
            # 如果指定了主题，只返回该主题的资源
            if subject and subject != "通用":
                all_resources = self.resources.get(subject, [])
                if not all_resources:
                    all_resources = self.resources.get("通用", [])
            
            # 应用过滤
            filtered_resources = all_resources
            if difficulty != "all":
                filtered_resources = [r for r in filtered_resources if r.get("difficulty") == difficulty]
            if resource_type != "all":
                filtered_resources = [r for r in filtered_resources if r.get("type") == resource_type]
            
            # 丰富资源信息
            resources = []
            for resource in filtered_resources:
                resources.append(self._enhance_resource(resource, subject))
            
        else:
            # 正常推荐逻辑
            user_id = content.get("user_id", "default")
            resources = self.recommend_resources(subject, difficulty, resource_type, count, user_id=user_id)
        
        response_content = {
            "status": "success",
            "subject": subject,
            "resources": resources,
            "count": len(resources)
        }
        
        return self.send_message(
            receiver=message.sender,
            msg_type=MessageType.RESPONSE,
            content=response_content
        )
    
    def recommend_resources(self, subject: str, difficulty: str = "all", 
                           resource_type: str = "all", count: int = 5,
                           user_id: str = "default", personalized: bool = True) -> List[Dict[str, Any]]:
        """推荐资源（增强版：个性化推荐）"""
        subject_resources = list(self.resources.get(subject, []))
        if not subject_resources:
            subject_resources = self.resources.get("通用", [])
            subject_resources = list(subject_resources)
        
        # 尝试拉取外部资源
        external_resources = self._fetch_external_resources(subject, count)
        if external_resources:
            subject_resources.extend(external_resources)

        online_resources = self._search_online_resources(subject, count)
        if online_resources:
            subject_resources.extend(online_resources)
        
        # 过滤资源
        filtered_resources = subject_resources
        
        if difficulty != "all":
            filtered_resources = [r for r in filtered_resources if r.get("difficulty") == difficulty]
        
        if resource_type != "all":
            filtered_resources = [r for r in filtered_resources if r.get("type") == resource_type]
        
        # 个性化评分和排序
        if personalized:
            from core.storage import DataStorage
            storage = DataStorage()
            user_progress = storage.get_progress(user_id, subject) if user_id != "default" else None
            user_profile = self.profile_store.get_profile(user_id) if user_id != "default" else None
            
            scored_resources = []
            for resource in filtered_resources:
                score = self._calculate_resource_score(resource, subject, user_progress, user_profile)
                scored_resources.append((resource, score))
            
            # 按分数排序
            scored_resources.sort(key=lambda x: x[1], reverse=True)
            filtered_resources = [r for r, _ in scored_resources]
        
        selected_resources = filtered_resources[:count]
        web_candidates = [r for r in filtered_resources if r.get("source") == "web_search"]
        if web_candidates and count > 0:
            min_web_count = 2 if count >= 4 else 1
            min_web_count = min(min_web_count, len(web_candidates), count)
            selected_web = web_candidates[:min_web_count]
            selected_keys = {(r.get("name"), r.get("url")) for r in selected_web}
            selected_resources = list(selected_web)
            for resource in filtered_resources:
                key = (resource.get("name"), resource.get("url"))
                if key in selected_keys:
                    continue
                selected_resources.append(resource)
                if len(selected_resources) >= count:
                    break

        enriched = []
        for resource in selected_resources:
            enriched.append(self._enhance_resource(resource, subject))
        return enriched
    
    def _search_online_resources(self, subject: str, limit: int) -> List[Dict[str, Any]]:
        query = f"{subject} 学习 教程 资料"
        try:
            results = self.execute_tool("web_search", {"query": query, "max_results": limit})
        except Exception as e:
            logger.warning(f"调用 web_search 工具失败: {e}")
            return []
        
        if not results:
            return []
        
        resources = []
        for item in results:
            if not isinstance(item, dict):
                continue
            if item.get("error"):
                continue
            resources.append({
                "name": item.get("title") or "在线学习资源",
                "type": "网页",
                "url": item.get("url"),
                "difficulty": "通用",
                "description": (item.get("content") or "")[:120],
                "source": "web_search"
            })
        return resources
    
    def _calculate_resource_score(self, resource: Dict[str, Any], subject: str, user_progress: Dict[str, Any] = None, user_profile: Dict[str, Any] = None) -> float:
        """计算资源推荐分数"""
        score = 0.5  # 基础分
        
        # 1. 资源类型权重
        resource_type = resource.get("type", "")
        type_scores = {
            "书籍": 0.9,  # 书籍通常较系统
            "文档": 0.8,
            "课程": 0.85, # 课程交互性好
            "教程": 0.75,
            "网站": 0.7,
            "实践": 0.8,  # 实践类资源价值高
            "练习": 0.75
        }
        score += type_scores.get(resource_type, 0.6)
        
        # 2. 难度匹配分数（结合进度和用户画像）
        difficulty = resource.get("difficulty", "通用")
        progress_percentage = user_progress.get("progress_percentage", 0) if user_progress else 0
        
        # 如果有用户画像，优先使用用户画像中的已知技能进行推断
        if user_profile and user_profile.get("known_skills"):
            is_known = any(subject.lower() in skill.lower() for skill in user_profile["known_skills"])
            if is_known:
                # 已经掌握的基础，不推荐初级
                if difficulty == "初级":
                    score -= 0.3
                elif difficulty == "高级" or difficulty == "中级":
                    score += 0.3
            else:
                # 未知领域，结合进度判断
                if progress_percentage < 30:
                    if difficulty == "初级": score += 0.3
                    elif difficulty == "高级": score -= 0.2
                elif progress_percentage < 70:
                    if difficulty == "中级": score += 0.3
                else:
                    if difficulty == "高级" or difficulty == "中级": score += 0.3
        else:
            # 仅依赖进度判断
            if progress_percentage < 30:
                if difficulty == "初级": score += 0.3
                elif difficulty == "高级": score -= 0.2
            elif progress_percentage < 70:
                if difficulty == "中级": score += 0.3
            else:
                if difficulty == "高级" or difficulty == "中级": score += 0.3
                
        # 3. 薄弱方向加分
        if user_profile and user_profile.get("weak_areas"):
            is_weak = any(subject.lower() in weak.lower() for weak in user_profile["weak_areas"])
            if is_weak and (resource_type == "练习" or resource_type == "实践"):
                score += 0.2 # 针对薄弱点多推荐练习和实践
        
        # 4. URL可用性（有URL的资源更易访问）
        if resource.get("url"):
            score += 0.1
        
        # 5. 描述质量（描述详细的资源可能更好）
        description = resource.get("description", "")
        if len(description) > 20:
            score += 0.1
        
        return min(score, 1.8)  # 提高上限以容纳更多加分项

    # === 新增能力 ===
    def _fetch_external_resources(self, subject: str, limit: int) -> List[Dict[str, Any]]:
        api_url = self.module_config.get("external_api")
        if not api_url:
            return []
        params = {"subject": subject, "limit": limit}
        headers = {}
        api_key = self.module_config.get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            resp = requests.get(api_url, params=params, headers=headers, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            resources = []
            for item in data.get("resources", []):
                normalized = {
                    "name": item.get("name") or item.get("title", "外部资源"),
                    "type": item.get("type", "外部"),
                    "url": item.get("url"),
                    "difficulty": item.get("difficulty", "通用"),
                    "description": item.get("description", ""),
                    "provider": item.get("provider"),
                    "source": "external_api"
                }
                resources.append(normalized)
            return resources
        except Exception as e:
            logger.warning(f"外部资源接口调用失败: {e}")
            return []

    def _enhance_resource(self, resource: Dict[str, Any], subject: str) -> Dict[str, Any]:
        enriched = dict(resource)
        enriched["tags"] = self._generate_tags(resource, subject)
        enriched["cluster"] = self._cluster_topic(resource)
        enriched["reason"] = self._generate_reason(resource, subject)
        return enriched

    def _generate_tags(self, resource: Dict[str, Any], subject: str) -> List[str]:
        tags = set()
        resource_type = resource.get("type")
        difficulty = resource.get("difficulty")
        if subject:
            tags.add(subject.lower())
        if resource_type:
            tags.add(resource_type)
        if difficulty and difficulty != "通用":
            tags.add(f"{difficulty}难度")
        description = (resource.get("description") or "").lower()
        keywords = ["项目", "实战", "基础", "进阶", "算法", "工具", "框架", "练习", "课程"]
        for kw in keywords:
            if kw in description:
                tags.add(kw)
        return list(tags)

    def _cluster_topic(self, resource: Dict[str, Any]) -> str:
        difficulty = (resource.get("difficulty") or "").lower()
        resource_type = (resource.get("type") or "").lower()
        if "初" in difficulty:
            return "基础入门"
        if "高" in difficulty or "进阶" in difficulty:
            return "进阶强化"
        if resource_type in {"项目", "实践", "练习"}:
            return "项目实践"
        if resource_type in {"课程", "教程"}:
            return "系统课程"
        return "综合资源"

    def _generate_reason(self, resource: Dict[str, Any], subject: str) -> str:
        name = resource.get("name", "该资源")
        resource_type = resource.get("type", "资源")
        difficulty = resource.get("difficulty", "通用")
        description = resource.get("description", "")
        summary = description[:60] + ("..." if len(description) > 60 else "")
        return (
            f"推荐理由：{name} 属于{difficulty}水平的{resource_type}，"
            f"适合学习 {subject} 时巩固『{summary or '核心知识点'}』。"
        )
