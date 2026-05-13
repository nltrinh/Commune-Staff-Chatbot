from abc import ABC, abstractmethod
from typing import Any, List, Dict, Generator

class AIProvider(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def stream_chat(self, query: str, context: str, history: List[Dict[str, str]]) -> Generator[str, None, None]:
        pass

class VectorStoreProvider(ABC):
    @abstractmethod
    def save_document(self, doc_id: str, content: str, embedding: List[float], metadata: Dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def search(self, query_vector: List[float], top_k: int, filter_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete_file(self, file_id: str) -> bool:
        pass

class DocumentProcessor(ABC):
    @abstractmethod
    def extract_text(self, content: bytes, file_type: str) -> List[Dict[str, Any]]:
        pass
