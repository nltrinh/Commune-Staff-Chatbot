import logger_config
from typing import List, Dict, Any, Optional
from app.core.database import get_files_col
from app.core.config import settings

logger = logger_config.get_logger(__name__)

class DocumentService:
    def __init__(self):
        self.files_col = get_files_col()

    def get_all_files(self, filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        files = list(self.files_col.find(filter_dict or {}).sort("created_at", -1))
        for f in files:
            f["_id"] = str(f["_id"])
        return files

    def save_file_metadata(self, metadata: Dict[str, Any]) -> bool:
        try:
            self.files_col.insert_one(metadata)
            return True
        except Exception as e:
            logger.error(f"Error saving file metadata: {e}")
            return False

    def delete_file_metadata(self, file_id: str) -> bool:
        try:
            result = self.files_col.delete_one({"file_id": file_id})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting file metadata: {e}")
            return False

    def get_file_by_id(self, file_id: str) -> Optional[Dict[str, Any]]:
        file = self.files_col.find_one({"file_id": file_id})
        if file:
            file["_id"] = str(file["_id"])
        return file

    def update_file_metadata(self, file_id: str, update_data: Dict[str, Any]) -> bool:
        try:
            self.files_col.update_one({"file_id": file_id}, {"$set": update_data})
            return True
        except Exception as e:
            logger.error(f"Error updating file metadata: {e}")
            return False

    def update_file_status(self, file_id: str, status: str, error: str = None, chunks_count: int = None) -> bool:
        try:
            update_data = {"status": status}
            if error: update_data["error"] = error
            if chunks_count is not None: update_data["chunks_count"] = chunks_count
            self.files_col.update_one({"file_id": file_id}, {"$set": update_data})
            return True
        except Exception as e:
            logger.error(f"Error updating file status: {e}")
            return False
