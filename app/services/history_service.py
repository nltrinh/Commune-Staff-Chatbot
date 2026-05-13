import logger_config
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from app.core.database import get_history_col

logger = logger_config.get_logger(__name__)

class HistoryService:
    def __init__(self):
        self.history_col = get_history_col()

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        doc = self.history_col.find_one({"session_id": session_id})
        return doc.get("messages", []) if doc else []

    def save_history(self, session_id: str, messages: List[Dict[str, str]], username: str = None) -> bool:
        try:
            self.history_col.update_one(
                {"session_id": session_id},
                {
                    "$set": {
                        "messages": messages,
                        "updated_at": datetime.now(timezone.utc),
                        "username": username
                    }
                },
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Error saving history: {e}")
            return False

    def delete_session(self, session_id: str) -> bool:
        try:
            result = self.history_col.delete_one({"session_id": session_id})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            return False

    def get_user_sessions(self, username: str) -> List[Dict[str, Any]]:
        # This is a simplified version, usually you want titles etc.
        docs = self.history_col.find({"username": username}).sort("updated_at", -1)
        sessions = []
        for d in docs:
            # Generate a title from the first message if not exists
            title = "Hội thoại mới"
            if d.get("messages") and len(d["messages"]) > 0:
                first_msg = d["messages"][0]["content"]
                title = (first_msg[:40] + '...') if len(first_msg) > 40 else first_msg
            
            sessions.append({
                "id": d["session_id"],
                "title": title,
                "updated_at": d.get("updated_at")
            })
        return sessions

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        docs = self.history_col.find().sort("updated_at", -1)
        sessions = []
        for d in docs:
            title = "Hội thoại mới"
            if d.get("messages") and len(d["messages"]) > 0:
                first_msg = d["messages"][0]["content"]
                title = (first_msg[:40] + '...') if len(first_msg) > 40 else first_msg
            
            sessions.append({
                "id": d["session_id"],
                "username": d.get("username", "Unknown"),
                "title": title,
                "updated_at": d.get("updated_at")
            })
        return sessions
