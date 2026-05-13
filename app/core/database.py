from pymongo import MongoClient
from app.core.config import settings

class Database:
    _client: MongoClient = None

    @classmethod
    def get_client(cls) -> MongoClient:
        if cls._client is None:
            cls._client = MongoClient(settings.MONGO_URI)
        return cls._client

    @classmethod
    def get_db(cls):
        return cls.get_client()[settings.MONGO_DB_NAME]

    @classmethod
    def get_collection(cls, name: str):
        return cls.get_db()[name]

def get_db(): return Database.get_db()
def get_users_col(): return Database.get_collection(settings.COLLECTION_USERS)
def get_history_col(): return Database.get_collection(settings.COLLECTION_CHAT_HISTORY)
def get_files_col(): return Database.get_collection(settings.COLLECTION_UPLOADED_FILES)
def get_depts_col(): return Database.get_collection(settings.COLLECTION_DEPARTMENTS)
