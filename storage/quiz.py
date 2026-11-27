"""
测验存储模块
"""

import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime


class QuizStore:
    """测验存储"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = os.path.join(data_dir, "quiz")
        os.makedirs(self.data_dir, exist_ok=True)
    
    def save(self, user_id: str, quiz_data: Dict[str, Any]) -> None:
        """保存测验记录"""
        quiz_id = quiz_data.get("quiz_id")
        if not quiz_id:
            return
        
        user_dir = os.path.join(self.data_dir, user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        file_path = os.path.join(user_dir, f"quiz_{quiz_id}.json")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(quiz_data, f, ensure_ascii=False, indent=2)
        
        # 更新索引
        self._update_index(user_id, quiz_id, quiz_data)
    
    def get_quiz_by_id(self, user_id: str, quiz_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取测验"""
        file_path = os.path.join(self.data_dir, user_id, f"quiz_{quiz_id}.json")
        
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    
    def get_user_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """获取用户测验历史"""
        index_file = os.path.join(self.data_dir, user_id, "quiz_index.json")
        
        if not os.path.exists(index_file):
            return []
        
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            
            # 按创建时间排序
            quizzes = list(index.values())
            quizzes.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
            return quizzes[:limit]
        except Exception:
            return []
    
    def get_subject_history(self, user_id: str, subject: str, limit: int = 50) -> List[Dict[str, Any]]:
        """获取特定主题的测验历史"""
        all_quizzes = self.get_user_history(user_id, limit=1000)
        
        subject_quizzes = [
            q for q in all_quizzes 
            if q.get("subject", "").lower() == subject.lower()
        ]
        
        return subject_quizzes[:limit]
    
    def get_performance_stats(self, user_id: str, subject: Optional[str] = None) -> Dict[str, Any]:
        """获取测验表现统计"""
        if subject:
            history = self.get_subject_history(user_id, subject, limit=1000)
        else:
            history = self.get_user_history(user_id, limit=1000)
        
        completed = [q for q in history if q.get("status") == "completed"]
        
        if not completed:
            return {
                "total_quizzes": 0,
                "completed_quizzes": 0,
                "average_score": 0
            }
        
        scores = [q.get("score", 0) for q in completed if q.get("score") is not None]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return {
            "total_quizzes": len(history),
            "completed_quizzes": len(completed),
            "average_score": round(avg_score, 2)
        }
    
    def delete(self, user_id: str, quiz_id: str) -> bool:
        """删除测验记录"""
        try:
            # 删除测验文件
            file_path = os.path.join(self.data_dir, user_id, f"quiz_{quiz_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # 从索引中删除
            index_file = os.path.join(self.data_dir, user_id, "quiz_index.json")
            if os.path.exists(index_file):
                with open(index_file, 'r', encoding='utf-8') as f:
                    index = json.load(f)
                
                if quiz_id in index:
                    del index[quiz_id]
                    
                    with open(index_file, 'w', encoding='utf-8') as f:
                        json.dump(index, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"删除测验失败: {e}")
            return False
    
    def _update_index(self, user_id: str, quiz_id: str, quiz_data: Dict[str, Any]) -> None:
        """更新索引"""
        index_file = os.path.join(self.data_dir, user_id, "quiz_index.json")
        
        if os.path.exists(index_file):
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
        else:
            index = {}
        
        # 只保存索引信息（不包含完整题目）
        index_entry = {
            "quiz_id": quiz_id,
            "title": quiz_data.get("title", ""),
            "subject": quiz_data.get("subject"),
            "difficulty": quiz_data.get("difficulty"),
            "num_questions": quiz_data.get("num_questions"),
            "status": quiz_data.get("status"),
            "score": quiz_data.get("score"),
            "created_at": quiz_data.get("created_at"),
            "completed_at": quiz_data.get("completed_at")
        }
        
        index[quiz_id] = index_entry
        
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

