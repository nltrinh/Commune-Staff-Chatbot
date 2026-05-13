import logger_config
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from app.core.database import get_users_col, get_depts_col, get_db, get_files_col, get_history_col
from app.core.config import settings

logger = logger_config.get_logger(__name__)

class AdminService:
    def __init__(self):
        self.users_col = get_users_col()
        self.depts_col = get_depts_col()

    def create_department(self, dept_data: Dict[str, Any]) -> bool:
        try:
            if self.depts_col.find_one({"code": dept_data["code"]}):
                return False
            dept_data.setdefault("created_at", datetime.now(timezone.utc))
            self.depts_col.insert_one(dept_data)
            return True
        except Exception as e:
            logger.error(f"Error creating department: {e}")
            return False

    def get_all_users(self) -> List[Dict[str, Any]]:
        users = list(self.users_col.find({}, {"hashed_password": 0}).sort("created_at", -1))
        for u in users:
            u["_id"] = str(u["_id"])
        return users

    def create_user(self, user_data: Dict[str, Any]) -> bool:
        try:
            if self.users_col.find_one({"username": user_data["username"]}):
                return False
            # Ensure default permissions
            user_data.setdefault("can_upload_file", True)
            user_data.setdefault("restrict_chatbot_dept", True)
            user_data.setdefault("disabled", False)
            self.users_col.insert_one(user_data)
            return True
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return False

    def update_user(self, username: str, update_data: Dict[str, Any]) -> bool:
        try:
            result = self.users_col.update_one({"username": username}, {"$set": update_data})
            return result.matched_count > 0
        except Exception as e:
            logger.error(f"Error updating user: {e}")
            return False

    def delete_user(self, username: str) -> bool:
        if username == "admin": return False
        try:
            result = self.users_col.delete_one({"username": username})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting user: {e}")
            return False

    def get_all_departments(self) -> List[Dict[str, Any]]:
        depts = list(self.depts_col.find().sort("name", 1))
        for d in depts:
            d["_id"] = str(d["_id"])
            d["num_users"] = self.users_col.count_documents({"department": d["code"]})
        return depts

    def update_department(self, code: str, dept_data: Dict[str, Any]) -> bool:
        try:
            # If code changed, update users too
            new_code = dept_data.get("code")
            if new_code and new_code != code:
                self.users_col.update_many({"department": code}, {"$set": {"department": new_code}})
            
            result = self.depts_col.update_one({"code": code}, {"$set": dept_data})
            return result.matched_count > 0
        except Exception as e:
            logger.error(f"Error updating department: {e}")
            return False

    def delete_department(self, code: str) -> bool:
        try:
            # Check for users
            if self.users_col.count_documents({"department": code}) > 0:
                return False
            result = self.depts_col.delete_one({"code": code})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting department: {e}")
            return False

    def get_system_stats(self) -> Dict[str, Any]:
        try:
            db = get_db()
            total_files = get_files_col().count_documents({})
            total_chunks = db[settings.COLLECTION_DOCUMENTS].count_documents({})
            
            # User activity
            pipeline = [
                {"$unwind": "$messages"},
                {"$match": {"messages.role": "user"}},
                {"$group": {"_id": "$username", "total_questions": {"$sum": 1}}},
                {"$sort": {"total_questions": -1}}
            ]
            activity = list(db[settings.COLLECTION_CHAT_HISTORY].aggregate(pipeline))
            
            return {
                "total_files": total_files,
                "total_chunks": total_chunks,
                "user_activity": activity
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"total_files": 0, "total_chunks": 0, "user_activity": []}
