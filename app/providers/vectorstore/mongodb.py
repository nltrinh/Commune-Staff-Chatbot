import time
import logger_config
from pymongo import MongoClient
from typing import List, Dict, Any
from app.core.config import settings
from app.core.interfaces import VectorStoreProvider

logger = logger_config.get_logger(__name__)

class MongoDBVectorProvider(VectorStoreProvider):
    def __init__(self):
        self.client = MongoClient(settings.MONGO_URI)
        self.db = self.client[settings.MONGO_DB_NAME]
        self.collection = self.db[settings.COLLECTION_DOCUMENTS]

    def save_document(self, doc_id: str, content: str, embedding: List[float], metadata: Dict[str, Any]) -> bool:
        try:
            mongo_doc = {
                "doc_id": doc_id,
                "content": content,
                "embedding": embedding,
                "metadata": metadata,
                "created_at": metadata.get("ingested_at")
            }
            self.collection.insert_one(mongo_doc)
            return True
        except Exception as e:
            logger.error(f"Error saving to MongoDB: {e}")
            return False

    def search(self, query_vector: List[float], top_k: int, filter_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            if settings.MONGODB_VECTOR_MODE == "atlas":
                pipeline = self._get_atlas_pipeline(query_vector, top_k, filter_dict)
            else:
                # Production-ready native vector search for MongoDB 8.2+
                pipeline = self._get_native_pipeline(query_vector, top_k, filter_dict)
                
            return list(self.collection.aggregate(pipeline))
        except Exception as e:
            logger.warning(f"Vector Search ({settings.MONGODB_VECTOR_MODE}) failed: {e}")
            # Robust fallback: exact match filtering (no semantic score)
            results = list(self.collection.find(filter_dict or {}).sort("created_at", -1).limit(top_k))
            return [{"content": r["content"], "metadata": r["metadata"], "score": 0.0} for r in results]

    def _get_atlas_pipeline(self, query_vector: List[float], top_k: int, filter_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {
                "$vectorSearch": {
                    "index": settings.VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": top_k * 10,
                    "limit": top_k,
                    "filter": filter_dict if filter_dict else None
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "content": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            }
        ]

    def _get_native_pipeline(self, query_vector: List[float], top_k: int, filter_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Using native $vectorSearch operator (standard in MongoDB 8.2+)
        # This structure is designed to be compatible with local Replica Set deployments
        return [
            {
                "$vectorSearch": {
                    "queryVector": query_vector,
                    "path": "embedding",
                    "numCandidates": top_k * 10,
                    "limit": top_k,
                    "index": settings.VECTOR_INDEX_NAME,
                    "filter": filter_dict if filter_dict else {}
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "content": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

    def delete_file(self, file_id: str) -> bool:
        try:
            self.collection.delete_many({"metadata.file_id": file_id})
            return True
        except Exception as e:
            logger.error(f"Error deleting file from MongoDB: {e}")
            return False
