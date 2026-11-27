"""
复习存储模块
"""

import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime


class ReviewStore:
    """复习存储"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = os.path.join(data_dir, "review")
        os.makedirs(self.data_dir, exist_ok=True)
    
    def save_plan(self, user_id: str, review_data: Dict[str, Any]) -> None:
        """保存复习计划"""
        review_id = review_data.get("review_id")
        if not review_id:
            return
        
        user_dir = os.path.join(self.data_dir, user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        file_path = os.path.join(user_dir, f"review_{review_id}.json")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(review_data, f, ensure_ascii=False, indent=2)
        
        # 更新索引
        self._update_index(user_id, review_id, review_data)
    
    def get_next_review(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取下一次复习内容（最早到期的）"""
        index_file = os.path.join(self.data_dir, user_id, "review_index.json")
        
        if not os.path.exists(index_file):
            return None
        
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            
            # 找到所有活跃的复习计划
            active_reviews = [
                r for r in index.values()
                if r.get("status") == "active"
            ]
            
            if not active_reviews:
                return None
            
            now = datetime.now()
            overdue_reviews = []
            upcoming_reviews = []
            
            for review in active_reviews:
                next_review_at_str = review.get("next_review_at")
                if next_review_at_str:
                    try:
                        next_review_at = datetime.fromisoformat(next_review_at_str)
                        if next_review_at <= now:
                            overdue_reviews.append((next_review_at, review))
                        else:
                            upcoming_reviews.append((next_review_at, review))
                    except:
                        continue
            
            target_review = None
            
            # 优先返回已过期的（按时间最早的优先）
            if overdue_reviews:
                overdue_reviews.sort(key=lambda x: x[0])
                _, target_review = overdue_reviews[0]
            # 如果没有过期的，返回最近即将到来的（用于前端展示倒计时）
            elif upcoming_reviews:
                upcoming_reviews.sort(key=lambda x: x[0])
                _, target_review = upcoming_reviews[0]
            
            if target_review:
                review_id = target_review.get("review_id")
                if review_id:
                    return self.get_review_by_id(user_id, review_id)
            
            return None
        except Exception as e:
            print(f"获取下次复习失败: {str(e)}")
            return None

    def get_due_reviews(self, user_id: str) -> Dict[str, Any]:
        """获取所有到期复习项（主动提醒列表），并返回最近一次未到期项用于'等待中'提示"""
        index_file = os.path.join(self.data_dir, user_id, "review_index.json")
        if not os.path.exists(index_file):
            return {"overdue": [], "next": None}
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            active = [r for r in index.values() if r.get("status") == "active"]
            now = datetime.now()
            overdue = []
            upcoming = []
            for r in active:
                s = r.get("next_review_at")
                if not s:
                    continue
                try:
                    t = datetime.fromisoformat(s)
                except Exception:
                    continue
                if t <= now:
                    overdue.append(r)
                else:
                    upcoming.append((t, r))
            overdue.sort(key=lambda x: x.get("next_review_at", ""))
            overdue_full = [self.get_review_by_id(user_id, r.get("review_id")) or r for r in overdue]
            next_item = None
            if upcoming:
                upcoming.sort(key=lambda x: x[0])
                nid = upcoming[0][1].get("review_id")
                next_item = self.get_review_by_id(user_id, nid) or upcoming[0][1]
            return {"overdue": overdue_full, "next": next_item}
        except Exception as e:
            print(f"获取到期复习失败: {str(e)}")
            return {"overdue": [], "next": None}

    def get_review_by_id(self, user_id: str, review_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取复习计划"""
        # 1. 先尝试直接读取文件
        file_path = os.path.join(self.data_dir, user_id, f"review_{review_id}.json")
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        
        # 2. 如果文件不存在或读取失败，尝试从索引中查找并构建
        # 这种情况通常发生在索引有记录但文件丢失的情况下，可以作为降级方案
        index_file = os.path.join(self.data_dir, user_id, "review_index.json")
        if os.path.exists(index_file):
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    index = json.load(f)
                
                review_data = index.get(review_id)
                if review_data:
                    # 从索引中恢复基本信息
                    review_data["content"] = review_data.get("content", "内容加载失败，请删除后重新创建")
                    return review_data
            except Exception:
                pass
                
        return None
    
    def complete_review(self, user_id: str, review_id: str, 
                       performance: Dict[str, Any]) -> bool:
        """完成复习"""
        try:
            review = self.get_review_by_id(user_id, review_id)
            if not review:
                return False
            
            # 更新复习计划（已在ReviewAgent中完成，这里只是保存）
            self.save_plan(user_id, review)
            return True
        except Exception as e:
            print(f"完成复习失败: {str(e)}")
            return False
    
    def get_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """获取复习历史"""
        index_file = os.path.join(self.data_dir, user_id, "review_index.json")
        
        if not os.path.exists(index_file):
            return []
        
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            
            # 按创建时间排序
            reviews = list(index.values())
            reviews.sort(key=lambda x: x.get("scheduled_at", ""), reverse=True)
            
            return reviews[:limit]
        except Exception:
            return []
    
    def delete_plan(self, user_id: str, review_id: str) -> bool:
        """删除复习计划"""
        try:
            # 删除复习计划文件
            file_path = os.path.join(self.data_dir, user_id, f"review_{review_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # 从索引中删除
            index_file = os.path.join(self.data_dir, user_id, "review_index.json")
            if os.path.exists(index_file):
                with open(index_file, 'r', encoding='utf-8') as f:
                    index = json.load(f)
                
                if review_id in index:
                    del index[review_id]
                    
                    with open(index_file, 'w', encoding='utf-8') as f:
                        json.dump(index, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"删除复习计划失败: {str(e)}")
            return False
    
    def _update_index(self, user_id: str, review_id: str, review_data: Dict[str, Any]) -> None:
        """更新索引"""
        index_file = os.path.join(self.data_dir, user_id, "review_index.json")
        
        if os.path.exists(index_file):
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
        else:
            index = {}
        
        # 只保存索引信息
        index_entry = {
            "review_id": review_id,
            "subject": review_data.get("subject"),
            "importance": review_data.get("importance"),
            "status": review_data.get("status"),
            "scheduled_at": review_data.get("scheduled_at"),
            "next_review_at": review_data.get("next_review_at"),
            "review_count": review_data.get("review_count", 0),
            "mastery_level": review_data.get("mastery_level", 0.0),
            "last_reviewed_at": review_data.get("last_reviewed_at")
        }
        
        index[review_id] = index_entry
        
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

